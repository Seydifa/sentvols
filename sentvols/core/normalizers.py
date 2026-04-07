"""Financial text normalization layer for FinancialVADERAnnotator.

Provides a pluggable LLM pre-processing stage that can be placed *before*
annotation to handle long articles, author-style noise, and long-range
negation.  The annotator itself is never modified — the normalizer is an
optional upstream component in the pipeline.

Exported to ``sentvols.utils``:
    NormalizationResult
    NormalizerBackend
    OpenAIBackend
    ReasoningBackend
    AnthropicBackend
    TransformersBackend
    LangChainBackend
    FinancialTextNormalizer
    FinancialRAGNormalizer

Design principles
-----------------
* **User-owned clients** — the library never stores credentials.  Users
  pass their own authenticated client object (``openai.OpenAI()``, etc.).
* **Lazy heavy imports** — ``openai``, ``anthropic``, and ``transformers``
  are imported only inside ``call()`` on first use, so ``import sentvols``
  never fails for users who haven't installed those packages.
* **Audit-first** — every ``normalize()`` call returns a
  ``NormalizationResult`` that carries the original text, the model used,
  the exact prompt, and the reasoning trace (when available).  Nothing is
  discarded silently.
* **reasoning_available vs reasoning_trace** — ``reasoning_available`` is a
  static property of the backend class (can it ever return a trace?).
  ``reasoning_trace`` is the runtime value (``None`` when the backend ran
  but produced no trace).  Both are surfaced so audit logs are unambiguous.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .exports import registration

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_PROMPTS: dict[str, str] = {
    "extract": (
        "You are a financial analyst assistant.  Extract only the key financial "
        "facts from the article below as a short, direct list.  Focus exclusively on: "
        "earnings, revenue, guidance, dividends, debt levels, credit ratings, "
        "legal or regulatory actions, management changes, M&A activity.  "
        "Remove author style, opinion, speculation, and boilerplate.  "
        "Return only the extracted facts.\n\nArticle: {text}"
    ),
    "summarize": (
        "Summarize this financial news article in 2-3 sentences of neutral, "
        "factual financial reporting language.  Include only material events "
        "and quantitative outcomes.  Omit author opinion and style.\n\nArticle: {text}"
    ),
    "rewrite": (
        # v11 — engineered to resolve sarcasm and long-range negation while
        # preserving the three baseline cases (hedged positive, buried negative,
        # double negation).  Key findings from iterative testing:
        #   • A system-level persona + diverse few-shot examples outperforms
        #     rule-based instructions alone on small (≤1B) decoder models.
        #   • The sarcasm example must share surface tokens with real targets
        #     ('record losses', 'burning through cash') for 0.5B models to
        #     pattern-match.
        #   • Long-range negation ('do not expect X') requires an explicit
        #     example that rewrites to 'No X is anticipated' so VADER's
        #     three-token window can catch the negator.
        "SYSTEM: You translate financial prose into literal one-sentence financial "
        "facts for a sentiment classifier. Sarcasm → literal negative. "
        "Long negation → 'No X anticipated'.\n\n"
        "IN:  Oh great, another record quarter — record losses, that is. "
        "Management helpfully reminded investors that burning through cash at "
        "twice the rate is part of the plan.\n"
        "OUT: The company posted record losses and is burning through cash at "
        "twice the prior rate.\n\n"
        "IN:  Analysts do not expect the firm to file for bankruptcy protection.\n"
        "OUT: No bankruptcy filing is anticipated.\n\n"
        "IN:  While headwinds remain, the company posted stronger earnings "
        "and raised the dividend.\n"
        "OUT: The company posted stronger earnings and raised dividends.\n\n"
        "IN:  {text}\n"
        "OUT:"
    ),
}

_VALID_MODES: frozenset[str] = frozenset(_PROMPTS)

# Prefix injected for reasoning models (o1 / o3 / Claude extended thinking
# via prompt workaround).  The actual Anthropic ThinkingBlock path does NOT
# use this — it is only for OpenAI reasoning models that need an in-band tag.
_REASONING_WRAP: str = (
    "Think step by step about the key financial content before writing your output.\n"
    "<reasoning> your analysis </reasoning>\n"
    "<output> your final result </output>\n\n"
)

_REASONING_TAG_RE: re.Pattern = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)
_OUTPUT_TAG_RE: re.Pattern = re.compile(
    r"<output>(.*?)</output>", re.DOTALL | re.IGNORECASE
)


# ---------------------------------------------------------------------------
# NormalizationResult — the audit carrier
# ---------------------------------------------------------------------------


@registration(module="utils")
@dataclass
class NormalizationResult:
    """Carries the full provenance of a single normalization call.

    Attributes
    ----------
    normalized_text : str
        The text that should be passed to the annotator.
    original_text : str
        The raw input before any LLM transformation.
    backend : str
        Short name of the backend class used (e.g. ``"OpenAIBackend"``).
        ``"passthrough"`` when the text was short enough to skip the LLM.
    model : str
        Model identifier as reported by the backend (e.g. ``"gpt-4o-mini"``).
        ``"none"`` for passthrough.
    mode : str
        Normalization mode: ``"extract"``, ``"summarize"``, or ``"rewrite"``.
    prompt_used : str
        Exact prompt submitted to the model — enables reproducibility audits.
        Empty string for passthrough.
    reasoning_trace : str or None
        Chain-of-thought reasoning returned by the backend.  ``None`` either
        because the backend class cannot produce traces
        (``reasoning_available=False``) or because the backend ran but
        returned nothing.
    reasoning_available : bool
        Static property of the backend — ``True`` if the backend *can* return
        reasoning traces (even if ``reasoning_trace`` is ``None`` for this
        call).
    llm_used : bool
        ``False`` when text was short enough to pass through without an LLM call.
    score_delta : float or None
        Difference between the score on ``normalized_text`` and the score on
        ``original_text``.  Filled externally by the pipeline after annotation;
        ``None`` until then.  A large ``|score_delta|`` flags potential LLM
        drift.
    """

    normalized_text: str
    original_text: str
    backend: str
    model: str
    mode: str
    prompt_used: str
    reasoning_trace: str | None
    reasoning_available: bool
    llm_used: bool
    score_delta: float | None = field(default=None)


# ---------------------------------------------------------------------------
# NormalizerBackend Protocol
# ---------------------------------------------------------------------------


@registration(module="utils")
@runtime_checkable
class NormalizerBackend(Protocol):
    """Structural protocol that all backend classes must satisfy.

    Any object with the right attributes and ``call()`` method works —
    including user-written backends for custom providers — without requiring
    inheritance from this class.

    Methods
    -------
    call(prompt) -> (normalized_text, reasoning_trace_or_None)
    """

    @property
    def model(self) -> str:
        """Model identifier string (used in NormalizationResult)."""
        ...

    @property
    def reasoning_available(self) -> bool:
        """True if this backend can structurally return a reasoning trace."""
        ...

    def call(self, prompt: str) -> tuple[str, str | None]:
        """Submit *prompt* and return ``(normalized_text, reasoning_trace)``.

        ``reasoning_trace`` is ``None`` when unavailable.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAIBackend  (GPT-4o, GPT-4o-mini, etc.)
# ---------------------------------------------------------------------------


@registration(module="utils")
class OpenAIBackend:
    """Backend for standard OpenAI chat-completion models (GPT-4o family).

    Parameters
    ----------
    client :
        An authenticated ``openai.OpenAI()`` (or ``AsyncOpenAI``) instance.
        The library never creates or stores credentials.
    model : str, default ``"gpt-4o-mini"``
        Any OpenAI chat-completion model identifier.
    temperature : float, default ``0.0``
        Set to 0 for deterministic, audit-friendly outputs.
    max_tokens : int, default ``512``
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return False

    def call(self, prompt: str) -> tuple[str, None]:
        """Call the OpenAI completions API.  Returns ``(text, None)``."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a financial text processing assistant.  "
                        "Follow the user's instructions precisely and return "
                        "only the requested output."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content.strip(), None

    def __repr__(self) -> str:
        return f"OpenAIBackend(model={self._model!r}, temperature={self._temperature})"


# ---------------------------------------------------------------------------
# ReasoningBackend  (o1, o3, o3-mini)
# ---------------------------------------------------------------------------


@registration(module="utils")
class ReasoningBackend:
    """Backend for OpenAI reasoning models (o1 / o3 / o3-mini).

    Key differences from ``OpenAIBackend``:

    * **No system message** — o1 and o3 models reject ``role="system"``.
      The entire context is sent as a single ``user`` turn.
    * **In-band reasoning trace** — the prompt is wrapped with
      ``<reasoning>`` / ``<output>`` tags so chain-of-thought is extractable
      from the response text.  This is a prompt workaround, not a native API
      feature (OpenAI does not expose internal reasoning tokens).

    Parameters
    ----------
    client :
        An authenticated ``openai.OpenAI()`` instance.
    model : str, default ``"o3-mini"``
    reasoning_effort : str, default ``"medium"``
        ``"low"``, ``"medium"``, or ``"high"`` passed via the API parameter.
    max_tokens : int, default ``2048``
    """

    def __init__(
        self,
        client,
        model: str = "o3-mini",
        reasoning_effort: str = "medium",
        max_tokens: int = 2048,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return True

    def call(self, prompt: str) -> tuple[str, str | None]:
        """Call an OpenAI reasoning model.

        Returns ``(normalized_text, reasoning_trace)``.  Both are extracted
        from ``<output>`` and ``<reasoning>`` tags in the response; falls back
        to the full response text as ``normalized_text`` when tags are absent.
        """
        full_prompt = _REASONING_WRAP + prompt
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": full_prompt}],
            max_completion_tokens=self._max_tokens,
            # reasoning_effort is passed through extra_body for models that
            # support it; silently ignored otherwise.
            extra_body={"reasoning_effort": self._reasoning_effort},
        )
        raw = response.choices[0].message.content or ""

        reasoning_match = _REASONING_TAG_RE.search(raw)
        output_match = _OUTPUT_TAG_RE.search(raw)

        reasoning_trace = reasoning_match.group(1).strip() if reasoning_match else None
        normalized_text = output_match.group(1).strip() if output_match else raw.strip()
        return normalized_text, reasoning_trace

    def __repr__(self) -> str:
        return (
            f"ReasoningBackend(model={self._model!r}, "
            f"reasoning_effort={self._reasoning_effort!r})"
        )


# ---------------------------------------------------------------------------
# AnthropicBackend  (Claude 3.7+ with extended thinking)
# ---------------------------------------------------------------------------


@registration(module="utils")
class AnthropicBackend:
    """Backend for Anthropic Claude models with extended thinking enabled.

    Extended thinking provides **native** chain-of-thought reasoning via
    ``ThinkingBlock`` objects in the response — no prompt engineering required.

    Parameters
    ----------
    client :
        An authenticated ``anthropic.Anthropic()`` instance.
    model : str, default ``"claude-3-7-sonnet-20250219"``
    thinking_budget : int, default ``8000``
        Token budget allocated to the thinking process.
    max_tokens : int, default ``16000``
        Must be strictly greater than ``thinking_budget``.

    Raises
    ------
    ValueError
        If ``max_tokens <= thinking_budget``.
    """

    def __init__(
        self,
        client,
        model: str = "claude-3-7-sonnet-20250219",
        thinking_budget: int = 8000,
        max_tokens: int = 16000,
    ) -> None:
        if max_tokens <= thinking_budget:
            raise ValueError(
                f"max_tokens ({max_tokens}) must be greater than "
                f"thinking_budget ({thinking_budget})."
            )
        self._client = client
        self._model = model
        self._thinking_budget = thinking_budget
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return True

    def call(self, prompt: str) -> tuple[str, str | None]:
        """Call the Anthropic API with extended thinking enabled.

        Iterates ``response.content``:

        * ``block.type == "thinking"`` → ``reasoning_trace``
        * ``block.type == "text"``     → ``normalized_text``
        """
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            thinking={
                "type": "enabled",
                "budget_tokens": self._thinking_budget,
            },
            messages=[{"role": "user", "content": prompt}],
        )

        reasoning_trace: str | None = None
        normalized_text: str = ""

        for block in response.content:
            if block.type == "thinking":
                reasoning_trace = block.thinking
            elif block.type == "text":
                normalized_text = block.text.strip()

        return normalized_text, reasoning_trace

    def __repr__(self) -> str:
        return (
            f"AnthropicBackend(model={self._model!r}, "
            f"thinking_budget={self._thinking_budget})"
        )


# ---------------------------------------------------------------------------
# TransformersBackend  (fully local inference)
# ---------------------------------------------------------------------------


@registration(module="utils")
class TransformersBackend:
    """Backend for local HuggingFace Transformers text2text models.

    The pipeline is instantiated lazily on the first ``call()`` and cached
    on the instance, so construction is fast and ``import sentvols`` never
    triggers a model download.

    Parameters
    ----------
    model : str, default ``"google/flan-t5-base"``
        Any ``text2text-generation`` compatible model identifier.
    device : str, default ``"cpu"``
        ``"cpu"``, ``"cuda"``, or ``"mps"``.
    max_new_tokens : int, default ``256``
    """

    def __init__(
        self,
        model: str = "google/flan-t5-base",
        device: str = "cpu",
        max_new_tokens: int = 256,
        torch_dtype=None,
        device_map: str | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._torch_dtype = torch_dtype
        self._device_map = device_map
        self._pipe = None  # lazy-loaded on first call

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return False

    def call(self, prompt: str) -> tuple[str, None]:
        """Run local inference using AutoModelForSeq2SeqLM (T5, BART, …).

        Uses the Auto classes directly rather than ``transformers.pipeline``
        so the code works across transformers versions (≥4.x including 4.50+
        which dropped ``text2text-generation`` from the pipeline registry).
        Supports ``torch_dtype`` (e.g. ``torch.bfloat16``) and
        ``device_map="auto"`` for large models that span multiple devices.
        Returns ``(text, None)`` — no reasoning trace for local models.
        """
        if self._pipe is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self._model)
            _load_kwargs: dict = {}
            if self._torch_dtype is not None:
                _load_kwargs["torch_dtype"] = self._torch_dtype
            if self._device_map is not None:
                _load_kwargs["device_map"] = self._device_map
                mdl = AutoModelForSeq2SeqLM.from_pretrained(self._model, **_load_kwargs)
            else:
                mdl = AutoModelForSeq2SeqLM.from_pretrained(self._model, **_load_kwargs)
                device_obj = torch.device(self._device)
                mdl = mdl.to(device_obj)
            mdl.eval()
            self._pipe = (tok, mdl)

        tok, mdl = self._pipe
        import torch

        _input_device = next(mdl.parameters()).device
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            _input_device
        )
        with torch.no_grad():
            out_ids = mdl.generate(**inputs, max_new_tokens=self._max_new_tokens)
        return tok.decode(out_ids[0], skip_special_tokens=True).strip(), None

    def __repr__(self) -> str:
        return f"TransformersBackend(model={self._model!r}, device={self._device!r})"


# ---------------------------------------------------------------------------
# OllamaBackend  (local server via ollama daemon)
# ---------------------------------------------------------------------------


@registration(module="utils")
class OllamaBackend:
    """Backend for models served by a running Ollama daemon.

    Ollama manages quantised GGUF models (llama.cpp under the hood) and
    exposes them through a tiny HTTP server.  Any model available through
    ``ollama pull`` can be used here.  Recommended small models for CPU:

    * ``qwen2.5:0.5b``  — 400 MB Q4_K_M, fast, good instruction following
    * ``qwen2.5:1.5b``  — 1 GB, better quality
    * ``llama3.2:1b``   — 800 MB, solid general purpose
    * ``phi3.5:mini``   — 2.2 GB, high quality, needs ~4 GB RAM

    Parameters
    ----------
    model : str, default ``"qwen2.5:0.5b"``
        Any Ollama model tag (``name:size`` format).
    host : str or None, default None
        Override the server URL, e.g. ``"http://localhost:11434"``.
        ``None`` uses the Ollama default (``OLLAMA_HOST`` env var or
        ``http://localhost:11434``).

    Raises
    ------
    RuntimeError
        On the first ``call()`` if the Ollama daemon is not reachable.

    Examples
    --------
    >>> # Start the server first: `ollama serve` in a terminal
    >>> # Pull a model:          `ollama pull qwen2.5:0.5b`
    >>> backend = OllamaBackend(model="qwen2.5:0.5b")
    >>> norm = FinancialTextNormalizer(backend=backend)
    >>> result = norm.normalize(long_article, mode="extract")
    """

    def __init__(self, model: str = "qwen2.5:0.5b", host: str | None = None) -> None:
        self._model = model
        self._host = host

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return False

    def call(self, prompt: str) -> tuple[str, None]:
        """Send a chat-completion request to the Ollama daemon.

        Returns ``(text, None)`` — Ollama does not expose chain-of-thought
        traces through its standard API.
        """
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "The 'ollama' Python package is required for OllamaBackend.  "
                "Install it with:  pip install ollama"
            ) from exc

        try:
            client = _ollama.Client(host=self._host) if self._host else _ollama.Client()
            resp = client.chat(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
        except Exception as exc:
            raise RuntimeError(
                f"OllamaBackend: failed to reach the Ollama daemon "
                f"(model={self._model!r}, host={self._host!r}).  "
                f"Make sure Ollama is installed and `ollama serve` is running.\n"
                f"Original error: {exc}"
            ) from exc

        return resp.message.content.strip(), None

    def __repr__(self) -> str:
        host_part = f", host={self._host!r}" if self._host else ""
        return f"OllamaBackend(model={self._model!r}{host_part})"


# ---------------------------------------------------------------------------
# LlamaCppBackend  (direct GGUF inference — no server needed)
# ---------------------------------------------------------------------------


@registration(module="utils")
class LlamaCppBackend:
    """Backend for local GGUF models via llama-cpp-python.

    Loads a quantised ``.gguf`` model directly in-process using llama.cpp's
    optimised CPU kernels (AVX2/AVX-512).  No server or internet connection
    is required after the model file has been downloaded.

    Why this is faster than ``TransformersBackend`` on CPU:
    * llama.cpp uses 4-bit or 8-bit integer arithmetic (GGUF quantisation)
      instead of 32-bit float.
    * Typical speed-up: **4–10× faster** with models of equivalent quality.

    Recommended small GGUF files for CPU (download from HuggingFace Hub):

    * ``Qwen2.5-0.5B-Instruct-Q4_K_M.gguf``   — ~350 MB, very fast
    * ``Qwen2.5-1.5B-Instruct-Q4_K_M.gguf``   — ~1 GB, good quality
    * ``Llama-3.2-1B-Instruct-Q4_K_M.gguf``   — ~800 MB

    Parameters
    ----------
    model_path : str
        Absolute or relative path to the ``.gguf`` model file.
    n_ctx : int, default 2048
        Context window size (tokens).
    n_threads : int or None, default None
        Number of CPU threads.  ``None`` lets llama.cpp choose automatically.
    max_tokens : int, default 256
        Maximum tokens to generate per call.

    Examples
    --------
    >>> backend = LlamaCppBackend("/models/qwen2.5-0.5b-instruct-q4_k_m.gguf")
    >>> norm = FinancialTextNormalizer(backend=backend)
    >>> result = norm.normalize(long_article, mode="extract")
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int | None = None,
        max_tokens: int = 256,
    ) -> None:
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_threads = n_threads
        self._max_tokens = max_tokens
        self._llm = None  # lazy-loaded on first call

    @property
    def model(self) -> str:
        """Returns the GGUF file path (used as the model identifier)."""
        return self._model_path

    @property
    def reasoning_available(self) -> bool:
        return False

    def call(self, prompt: str) -> tuple[str, None]:
        """Run inference against the loaded GGUF model.

        The ``Llama`` object is instantiated on the first call and cached
        for subsequent calls, so the expensive model load happens only once.
        Returns ``(text, None)`` — no reasoning trace.
        """
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise ImportError(
                    "The 'llama-cpp-python' package is required for LlamaCppBackend.  "
                    "Install it with:  pip install llama-cpp-python"
                ) from exc

            self._llm = Llama(
                model_path=self._model_path,
                n_ctx=self._n_ctx,
                n_threads=self._n_threads,
                verbose=False,
            )

        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self._max_tokens,
            temperature=0.0,
        )
        return resp["choices"][0]["message"]["content"].strip(), None

    def __repr__(self) -> str:
        import os

        return (
            f"LlamaCppBackend(model={os.path.basename(self._model_path)!r}, "
            f"n_ctx={self._n_ctx}, n_threads={self._n_threads})"
        )


# ---------------------------------------------------------------------------
# VLLMBackend  (vLLM offline engine — PagedAttention, continuous batching)
# ---------------------------------------------------------------------------


@registration(module="utils")
class VLLMBackend:
    """Backend using vLLM's offline inference engine.

    vLLM provides **PagedAttention** and **continuous batching** which
    typically gives 3–10× higher throughput than the vanilla Transformers
    ``generate()`` path for the same model.

    Works best with decoder-only instruct models (LLaMA, Mistral, Qwen, Phi …).
    Install with: ``pip install vllm``.

    Parameters
    ----------
    model : str
        HuggingFace repo ID or local path.  Recommended for Colab L4 24 GB
        (after freeing other models through ``del model; gc.collect()``):

        * ``"Qwen/Qwen2.5-3B-Instruct"``        — 3 B, ~6 GB fp16, very fast
        * ``"Qwen/Qwen2.5-7B-Instruct"``        — 7 B, ~14 GB fp16
        * ``"meta-llama/Llama-3.2-3B-Instruct"``— 3 B, ~6 GB fp16

    max_new_tokens : int, default 16
        Keep very small for pure scoring (output is a number like ``0.7``).
    temperature : float, default 0.0
        0 = greedy / deterministic.
    gpu_memory_utilization : float, default 0.90
        Fraction of GPU memory vLLM allocates for its KV-cache.
    dtype : str, default ``"auto"``
        ``"auto"``, ``"float16"``, or ``"bfloat16"``.
    use_chat_template : bool, default ``True``
        Wraps prompts in the model's Jinja chat template via
        ``AutoTokenizer.apply_chat_template``.  This is essential for
        instruct models (Qwen, LLaMA-Instruct, …) to follow the scoring
        instruction reliably.
    trust_remote_code : bool, default ``True``
        Required for Qwen models.

    Notes
    -----
    ``batch_call(prompts)`` submits all prompts as a single vLLM request —
    this is the primary performance advantage: vLLM fills idle KV-cache slots
    across requests (continuous batching) whereas sequential ``call()`` calls
    cannot benefit from this.
    """

    def __init__(
        self,
        model: str,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "auto",
        use_chat_template: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._gpu_memory_utilization = gpu_memory_utilization
        self._dtype = dtype
        self._use_chat_template = use_chat_template
        self._trust_remote_code = trust_remote_code
        self._engine = None  # lazy-init on first call
        self._tokenizer = None
        self._sampling_params = None

    @property
    def model(self) -> str:
        return self._model

    @property
    def reasoning_available(self) -> bool:
        return False

    # ------------------------------------------------------------------
    # Engine initialisation (lazy)
    # ------------------------------------------------------------------

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise ImportError(
                "The 'vllm' package is required for VLLMBackend.  "
                "Install it with:  pip install vllm"
            ) from exc

        self._engine = LLM(
            model=self._model,
            gpu_memory_utilization=self._gpu_memory_utilization,
            dtype=self._dtype,
            trust_remote_code=self._trust_remote_code,
            enforce_eager=True,  # disable CUDA graph capture → faster first call
        )
        self._sampling_params = SamplingParams(
            max_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )
        if self._use_chat_template:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model, trust_remote_code=self._trust_remote_code
            )

    # ------------------------------------------------------------------
    # Prompt formatting (chat template)
    # ------------------------------------------------------------------

    def _format(self, prompt: str) -> str:
        """Apply the model's chat template if requested."""
        if not (self._use_chat_template and self._tokenizer is not None):
            return prompt
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a financial sentiment scorer.  "
                    "Always respond with ONLY a single decimal number between "
                    "-1.0 and 1.0.  No words, no explanation, just the number."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def call(self, prompt: str) -> tuple[str, None]:
        """Single-sample inference.  Returns ``(score_text, None)``."""
        self._ensure_engine()
        outputs = self._engine.generate([self._format(prompt)], self._sampling_params)
        return outputs[0].outputs[0].text.strip(), None

    def batch_call(self, prompts: list[str]) -> list[tuple[str, None]]:
        """Batch inference — submits all prompts in **one** vLLM request.

        vLLM's PagedAttention engine fills idle KV-cache slots across the
        batch (continuous batching) which gives 3–8× higher throughput than
        looping over ``call()`` for the same number of inputs.

        Parameters
        ----------
        prompts : list[str]
            Raw (pre-chat-template) prompt strings.

        Returns
        -------
        list of (score_text, None) tuples, in input order.
        """
        self._ensure_engine()
        formatted = [self._format(p) for p in prompts]
        outputs = self._engine.generate(formatted, self._sampling_params)
        return [(o.outputs[0].text.strip(), None) for o in outputs]

    def __repr__(self) -> str:
        return (
            f"VLLMBackend(model={self._model!r}, "
            f"gpu_memory_utilization={self._gpu_memory_utilization}, "
            f"dtype={self._dtype!r})"
        )


# ---------------------------------------------------------------------------
# FinancialTextNormalizer
# ---------------------------------------------------------------------------


@registration(module="utils")
class FinancialTextNormalizer:
    """LLM-based pre-processing stage for ``FinancialVADERAnnotator``.

    The normalizer is entirely optional and sits *upstream* of the annotator.
    It accepts any :class:`NormalizerBackend`-compatible object so users can
    swap providers without touching the annotator.

    Parameters
    ----------
    backend : NormalizerBackend
        Any object satisfying the ``NormalizerBackend`` protocol:
        ``OpenAIBackend``, ``ReasoningBackend``, ``AnthropicBackend``,
        ``TransformersBackend``, or a user-written custom backend.

    Raises
    ------
    TypeError
        If *backend* does not satisfy the ``NormalizerBackend`` protocol.

    Examples
    --------
    >>> import openai
    >>> backend = OpenAIBackend(client=openai.OpenAI(), model="gpt-4o-mini")
    >>> normalizer = FinancialTextNormalizer(backend)
    >>> result = normalizer.normalize(long_article, mode="extract")
    >>> print(result.normalized_text)
    >>> print(result.reasoning_trace)   # None for GPT-4o backends
    >>> score = ann.score(result.normalized_text)
    """

    def __init__(self, backend) -> None:
        if not isinstance(backend, NormalizerBackend):
            raise TypeError(
                f"backend must satisfy the NormalizerBackend protocol, "
                f"got {type(backend).__name__!r}."
            )
        self._backend = backend

    def normalize(self, text: str, mode: str = "extract") -> NormalizationResult:
        """Normalize *text* using the configured backend.

        Parameters
        ----------
        text : str
            Raw article or document text.
        mode : str, default ``"extract"``
            ``"extract"``   — key-fact extraction (best for scoring accuracy).
            ``"summarize"`` — 2-3 sentence neutral summary (best for display).
            ``"rewrite"``   — direct declarative rewrite (resolves long-range
                              negation and author hedging).

        Returns
        -------
        NormalizationResult
            Full audit record including normalized text, prompt, reasoning
            trace, and backend metadata.

        Raises
        ------
        ValueError
            If *mode* is not one of the three valid modes.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}."
            )
        prompt = _PROMPTS[mode].format(text=text)
        normalized, trace = self._backend.call(prompt)
        return NormalizationResult(
            normalized_text=normalized,
            original_text=text,
            backend=type(self._backend).__name__,
            model=self._backend.model,
            mode=mode,
            prompt_used=prompt,
            reasoning_trace=trace,
            reasoning_available=self._backend.reasoning_available,
            llm_used=True,
        )

    def normalize_if_needed(
        self,
        text: str,
        threshold_chars: int = 300,
        mode: str = "extract",
    ) -> NormalizationResult:
        """Normalize only when *text* exceeds *threshold_chars*.

        Short texts (headlines, brief paragraphs) pass through unchanged —
        the LLM adds latency with no benefit at that scale.

        Parameters
        ----------
        text : str
        threshold_chars : int, default 300
            Texts shorter than this bypass the LLM entirely.
        mode : str, default ``"extract"``

        Returns
        -------
        NormalizationResult
            ``llm_used=False`` and ``backend="passthrough"`` when bypassed.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}."
            )
        if len(text) < threshold_chars:
            return NormalizationResult(
                normalized_text=text,
                original_text=text,
                backend="passthrough",
                model="none",
                mode=mode,
                prompt_used="",
                reasoning_trace=None,
                reasoning_available=False,
                llm_used=False,
            )
        return self.normalize(text, mode=mode)

    def normalize_batch(
        self,
        texts: list[str],
        mode: str = "extract",
        workers: int = 1,
    ) -> list[NormalizationResult]:
        """Normalize a list of texts.

        Parameters
        ----------
        texts : list[str]
        mode : str, default ``"extract"``
        workers : int, default 1
            Number of threads for parallel execution (``ThreadPoolExecutor``).
            Useful for large batches against API-hosted models.

        Returns
        -------
        list[NormalizationResult]
            Results in the same order as *texts*.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}."
            )
        _norm = lambda t: self.normalize(t, mode=mode)  # noqa: E731
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_norm, texts))
        return [_norm(t) for t in texts]

    def __repr__(self) -> str:
        return (
            f"FinancialTextNormalizer("
            f"backend={self._backend!r}, "
            f"reasoning_available={self._backend.reasoning_available})"
        )


# ---------------------------------------------------------------------------
# LangChainBackend  (thin adapter — zero hard dependency)
# ---------------------------------------------------------------------------


@registration(module="utils")
class LangChainBackend:
    """Adapter that wraps any LangChain ``BaseChatModel`` as a ``NormalizerBackend``.

    Users who already have LangChain installed can plug **any** supported
    provider (``ChatOpenAI``, ``ChatAnthropic``, ``ChatOllama``, ``ChatGroq``,
    ``ChatMistralAI``, …) into :class:`FinancialTextNormalizer` or
    :class:`FinancialRAGNormalizer` without reimplementing the call protocol.

    ``langchain_core`` is imported **lazily inside call()** — constructing this
    class does not require LangChain to be installed, and ``import sentvols``
    never fails for users without it.

    Parameters
    ----------
    llm :
        Any ``langchain_core.language_models.BaseChatModel`` instance,
        already configured with credentials, model name, and temperature.

    Examples
    --------
    >>> from langchain_openai import ChatOpenAI
    >>> from sentvols.utils import LangChainBackend, FinancialTextNormalizer
    >>> lc_backend = LangChainBackend(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    >>> norm = FinancialTextNormalizer(lc_backend)
    >>> result = norm.normalize(long_article, mode="rewrite")
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    # --- NormalizerBackend protocol -------------------------------------------

    @property
    def model(self) -> str:
        """Model identifier — reads ``model_name`` or ``model`` from the LLM."""
        return getattr(
            self._llm,
            "model_name",
            getattr(self._llm, "model", type(self._llm).__name__),
        )

    @property
    def reasoning_available(self) -> bool:
        # LangChain's standard API does not surface internal reasoning traces.
        return False

    def call(self, prompt: str) -> tuple[str, None]:
        """Invoke the LangChain LLM with *prompt* as a single human message.

        Returns ``(response_text, None)`` — no reasoning trace.

        Raises
        ------
        ImportError
            If ``langchain_core`` is not installed.
        """
        try:
            from langchain_core.messages import HumanMessage
        except ImportError as exc:
            raise ImportError(
                "The 'langchain-core' package is required for LangChainBackend.  "
                "Install it with:  pip install langchain-core"
            ) from exc
        response = self._llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip(), None

    def __repr__(self) -> str:
        return f"LangChainBackend(llm={self._llm!r})"


# ---------------------------------------------------------------------------
# LM-lexicon RAG corpus (module-level, built once on first access)
# ---------------------------------------------------------------------------
# We store the corpus as a module-level tuple so multiple
# FinancialRAGNormalizer instances built in the same process share the same
# numpy arrays without recomputing embeddings every time.

_RAG_CORPUS: tuple | None = None  # (terms: list[str], valences: list[float],
#                                     embeddings: np.ndarray, encoder)


def _get_rag_corpus(embed_model: str):
    """Build or return the shared RAG lexicon corpus.

    The corpus is built **once per process** from ``_LM_LEXICON``,
    ``CONTEXT_PHRASES``, and ``_HEADLINE_PATCH``.  The embedding model is
    loaded lazily so ``import sentvols`` is never slowed down.

    Building takes ~0.5 s for 4 000 terms on CPU (``all-MiniLM-L6-v2``
    produces 384-dim embeddings in a single batch pass).  Results are cached
    on ``_RAG_CORPUS`` so subsequent instances reuse the same arrays.

    Parameters
    ----------
    embed_model : str
        SentenceTransformer model identifier.

    Returns
    -------
    tuple of (terms, valences, embeddings, encoder)
    """
    global _RAG_CORPUS
    if _RAG_CORPUS is not None:
        return _RAG_CORPUS

    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "The 'sentence-transformers' and 'numpy' packages are required for "
            "FinancialRAGNormalizer.  Install them with:\n"
            "  pip install 'sentvols[rag]'\n"
            "or:  pip install sentence-transformers numpy"
        ) from exc

    # Deferred import to avoid circular dependency at module load time.
    from .annotators import CONTEXT_PHRASES, _HEADLINE_PATCH, _LM_LEXICON

    # Merge all term→valence sources; skip neutral entries (valence == 0)
    merged: dict[str, float] = {}
    merged.update({t: v for t, v in _LM_LEXICON.items() if v != 0.0})
    merged.update({t: v for t, v in _HEADLINE_PATCH.items() if v != 0.0})
    merged.update({t: v for t, v in CONTEXT_PHRASES.items() if v != 0.0})

    terms = list(merged.keys())
    valences = list(merged.values())

    encoder = SentenceTransformer(embed_model)
    # L2-normalised embeddings: cosine similarity == dot product (faster lookup)
    embeddings = encoder.encode(
        terms,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=256,  # single GPU/CPU pass over the small corpus
    )
    _RAG_CORPUS = (terms, valences, np.array(embeddings, dtype="float32"), encoder)
    return _RAG_CORPUS


# RAG-augmented rewrite prompt — vocabulary hint layer injected between the
# few-shot examples and the live input so the LLM stays grounded in the exact
# words that VADER+LM can score.
#
# IMPORTANT: the hint section uses intentionally conservative language so the
# LLM does NOT introduce sentiment absent from the source text.  Hints are
# only injected when retrieval similarity exceeds `sim_threshold` — genuinely
# neutral text skips this prompt entirely and falls back to the base class.
_RAG_REWRITE_PROMPT: str = (
    "SYSTEM: You translate financial prose into literal one-sentence financial "
    "facts for a sentiment classifier. Sarcasm → literal negative. "
    "Long negation → 'No X anticipated'.\n"
    "CRITICAL RULE: preserve only sentiment that is already present in the "
    "source text. Never introduce evaluative words that the source does not "
    "express, even if they appear in the hints below.\n\n"
    "IN:  Oh great, another record quarter — record losses, that is. "
    "Management helpfully reminded investors that burning through cash at "
    "twice the rate is part of the plan.\n"
    "OUT: The company posted record losses and is burning through cash at "
    "twice the prior rate.\n\n"
    "IN:  Analysts do not expect the firm to file for bankruptcy protection.\n"
    "OUT: No bankruptcy filing is anticipated.\n\n"
    "IN:  While headwinds remain, the company posted stronger earnings "
    "and raised the dividend.\n"
    "OUT: The company posted stronger earnings and raised dividends.\n\n"
    "VOCABULARY HINTS — if the original text already expresses the "
    "corresponding concept, prefer these LM-lexicon terms in your rewrite so "
    "the downstream scorer can detect them.  Do NOT use a hint term unless "
    "its concept is directly present in the source sentence.\n"
    "  Positive (use only if positive concept is in source): {pos_hints}\n"
    "  Negative (use only if negative concept is in source): {neg_hints}\n\n"
    "IN:  {text}\n"
    "OUT:"
)


@registration(module="utils")
class FinancialRAGNormalizer(FinancialTextNormalizer):
    """LM-lexicon retrieval-augmented normalizer for the VADER+LM pipeline.

    Extends :class:`FinancialTextNormalizer` by **injecting vocabulary hints**
    into the ``"rewrite"`` prompt before each LLM call.  The hints are the
    top-K LM-lexicon terms most semantically similar to the input text, split
    by polarity.  This closes the gap between what the LLM writes and what
    VADER+LM can score: even if the LLM paraphrases correctly, it now has
    explicit cues to use high-valence words like *"impairment"*, *"delinquent"*,
    or *"raised guidance"* that VADER+LM recognises.

    ``"extract"`` and ``"summarize"`` modes are passed through unchanged to
    the parent class — RAG only augments the ``"rewrite"`` path.

    Runtime optimisations
    ---------------------
    * **Shared corpus**: the embedding index is built once per process and
      cached at module level (``_RAG_CORPUS``).  Subsequent instances reuse
      the same numpy arrays — zero redundant model loads.
    * **Lazy build**: the encoder is loaded only on the first ``normalize()``
      call with ``mode="rewrite"``, so construction is instant and
      ``import sentvols`` is never slowed down.
    * **Brute-force cosine**: with ~4 000 terms at 384 dim a single ``@``
      dot-product takes < 1 ms on CPU — no vector-DB overhead needed.
    * **Query-level batch**: the query embedding is computed in a single
      ``encoder.encode([query])`` call; no unnecessary batching.

    Parameters
    ----------
    backend : NormalizerBackend
        Any object satisfying the :class:`NormalizerBackend` protocol.
    top_k : int, default 8
        Total vocabulary hints injected (split evenly between positive and
        negative polarity, i.e. ``top_k // 2`` each).
    embed_model : str, default ``"all-MiniLM-L6-v2"``
        SentenceTransformer model used for retrieval.  Loaded lazily on first
        ``normalize()`` call and cached for subsequent calls.  The default
        model is ~80 MB and runs on CPU in < 50 ms per query.

    Examples
    --------
    >>> from sentvols.utils import FinancialRAGNormalizer, OllamaBackend
    >>> backend = OllamaBackend("qwen2.5:0.5b")
    >>> norm = FinancialRAGNormalizer(backend, top_k=8)
    >>> result = norm.normalize(
    ...     "Oh great, another record quarter — record losses, that is.",
    ...     mode="rewrite",
    ... )
    >>> print(result.normalized_text)
    >>> print(result.prompt_used)     # shows injected vocabulary hints
    """

    def __init__(
        self,
        backend,
        *,
        top_k: int = 8,
        embed_model: str = "all-MiniLM-L6-v2",
        sim_threshold: float = 0.40,
    ) -> None:
        super().__init__(backend)
        if top_k < 2:
            raise ValueError(f"top_k must be >= 2, got {top_k}.")
        if not 0.0 < sim_threshold < 1.0:
            raise ValueError(f"sim_threshold must be in (0, 1), got {sim_threshold}.")
        self._top_k = top_k
        self._embed_model = embed_model
        self._sim_threshold = sim_threshold

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> tuple[list[str], list[str]]:
        """Return (positive_hints, negative_hints) via cosine retrieval.

        Only terms whose similarity to *query* exceeds ``sim_threshold`` are
        considered.  If no term clears the threshold (i.e. the text uses
        vocabulary not represented in the LM lexicon, which is the typical
        signature of genuinely neutral prose) **both lists are empty**.  The
        caller (:meth:`normalize`) treats empty lists as a signal to bypass
        hint injection entirely and fall back to the base-class prompt so the
        LLM is not pushed toward unwarranted sentiment.
        """
        import numpy as np

        terms, valences, embeddings, encoder = _get_rag_corpus(self._embed_model)

        q_emb = encoder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0].astype("float32")

        # cosine similarity = dot product (embeddings are already L2-normalised)
        sims = embeddings @ q_emb
        # Over-fetch so we have enough to fill both polarity buckets
        candidate_idx = np.argsort(sims)[::-1][: self._top_k * 3]

        half = max(1, self._top_k // 2)
        pos: list[str] = []
        neg: list[str] = []
        for i in candidate_idx:
            if float(sims[i]) < self._sim_threshold:
                # Candidates are sorted descending — all remaining are below threshold.
                # Stop: the text has no sufficiently similar LM-lexicon vocabulary.
                break
            if valences[i] > 0 and len(pos) < half:
                pos.append(terms[i])
            elif valences[i] < 0 and len(neg) < half:
                neg.append(terms[i])
            if len(pos) >= half and len(neg) >= half:
                break

        return pos, neg

    # ------------------------------------------------------------------
    # Override normalize() for the "rewrite" mode only
    # ------------------------------------------------------------------

    def normalize(self, text: str, mode: str = "extract") -> NormalizationResult:
        """Normalize *text*, injecting LM-lexicon vocabulary hints for ``mode="rewrite"``.

        For ``"extract"`` and ``"summarize"`` the call is forwarded to the
        parent :class:`FinancialTextNormalizer` without modification.

        Parameters
        ----------
        text : str
        mode : str, default ``"extract"``

        Returns
        -------
        NormalizationResult
            ``prompt_used`` contains the full RAG-augmented prompt so the
            injected hints are fully auditable.
        """
        if mode not in _VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_VALID_MODES)}, got {mode!r}."
            )

        if mode != "rewrite":
            return super().normalize(text, mode=mode)

        pos_hints, neg_hints = self._retrieve(text)

        # If retrieval returned nothing it means no LM-lexicon term cleared
        # sim_threshold — the text is generic/neutral enough that injecting
        # hints would corrupt rather than improve the score.  Fall back to the
        # base-class rewrite prompt which makes no vocabulary assumptions.
        if not pos_hints and not neg_hints:
            return super().normalize(text, mode=mode)

        prompt = _RAG_REWRITE_PROMPT.format(
            pos_hints=", ".join(pos_hints) if pos_hints else "—",
            neg_hints=", ".join(neg_hints) if neg_hints else "—",
            text=text,
        )
        normalized, trace = self._backend.call(prompt)
        return NormalizationResult(
            normalized_text=normalized,
            original_text=text,
            backend=type(self._backend).__name__,
            model=self._backend.model,
            mode=mode,
            prompt_used=prompt,
            reasoning_trace=trace,
            reasoning_available=self._backend.reasoning_available,
            llm_used=True,
        )

    def __repr__(self) -> str:
        return (
            f"FinancialRAGNormalizer("
            f"backend={self._backend!r}, "
            f"top_k={self._top_k}, "
            f"embed_model={self._embed_model!r}, "
            f"sim_threshold={self._sim_threshold})"
        )

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
    FinancialTextNormalizer

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
    ) -> None:
        self._model = model
        self._device = device
        self._max_new_tokens = max_new_tokens
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
        Returns ``(text, None)`` — no reasoning trace for local models.
        """
        if self._pipe is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self._model)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(self._model)
            device_obj = torch.device(self._device)
            mdl = mdl.to(device_obj)
            mdl.eval()
            # Store as a callable that mirrors the old pipeline interface
            self._pipe = (tok, mdl, device_obj)

        tok, mdl, device_obj = self._pipe
        import torch

        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(
            device_obj
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

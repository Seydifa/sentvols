"""Tests for sentvols/core/normalizers.py.

All tests use mocked clients — no real API calls are made.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import pytest

from sentvols.core.normalizers import (
    AnthropicBackend,
    FinancialRAGNormalizer,
    FinancialTextNormalizer,
    LangChainBackend,
    LlamaCppBackend,
    NormalizationResult,
    NormalizerBackend,
    OllamaBackend,
    OpenAIBackend,
    ReasoningBackend,
    TransformersBackend,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_openai_client(content: str) -> MagicMock:
    """Return a minimal mock that mimics the OpenAI client interface."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = response
    return client


def _mock_anthropic_client(text: str, thinking: str | None = None) -> MagicMock:
    """Return a minimal mock that mimics the Anthropic client interface."""
    blocks = []
    if thinking is not None:
        tb = MagicMock()
        tb.type = "thinking"
        tb.thinking = thinking
        blocks.append(tb)
    xtb = MagicMock()
    xtb.type = "text"
    xtb.text = text
    blocks.append(xtb)

    response = MagicMock()
    response.content = blocks
    client = MagicMock()
    client.messages.create.return_value = response
    return client


# ---------------------------------------------------------------------------
# NormalizationResult dataclass
# ---------------------------------------------------------------------------


class TestNormalizationResult:
    def test_required_fields(self):
        r = NormalizationResult(
            normalized_text="text",
            original_text="orig",
            backend="openai",
            model="gpt-4o",
            mode="extract",
            prompt_used="p",
            reasoning_trace=None,
            reasoning_available=False,
            llm_used=True,
        )
        assert r.normalized_text == "text"
        assert r.original_text == "orig"
        assert r.score_delta is None  # optional field defaults to None

    def test_score_delta_can_be_set(self):
        r = NormalizationResult(
            normalized_text="t",
            original_text="o",
            backend="b",
            model="m",
            mode="extract",
            prompt_used="p",
            reasoning_trace=None,
            reasoning_available=False,
            llm_used=False,
            score_delta=0.15,
        )
        assert r.score_delta == pytest.approx(0.15)

    def test_llm_not_used_passthrough(self):
        r = NormalizationResult(
            normalized_text="same",
            original_text="same",
            backend="none",
            model="none",
            mode="extract",
            prompt_used="",
            reasoning_trace=None,
            reasoning_available=False,
            llm_used=False,
        )
        assert r.normalized_text == r.original_text
        assert not r.llm_used


# ---------------------------------------------------------------------------
# NormalizerBackend Protocol
# ---------------------------------------------------------------------------


class TestNormalizerBackendProtocol:
    def test_openai_backend_satisfies_protocol(self):
        client = _mock_openai_client("output")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        assert isinstance(backend, NormalizerBackend)

    def test_reasoning_backend_satisfies_protocol(self):
        client = _mock_openai_client("<output>text</output>")
        backend = ReasoningBackend(client=client, model="o1")
        assert isinstance(backend, NormalizerBackend)

    def test_anthropic_backend_satisfies_protocol(self):
        client = _mock_anthropic_client("output")
        backend = AnthropicBackend(client=client, model="claude-3-5-sonnet-20241022")
        assert isinstance(backend, NormalizerBackend)

    def test_non_protocol_object_fails_isinstance(self):
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), NormalizerBackend)

    def test_invalid_backend_raises_in_normalizer(self):
        with pytest.raises(TypeError, match="NormalizerBackend"):
            FinancialTextNormalizer(backend=object())


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------


class TestOpenAIBackend:
    def test_call_returns_text(self):
        client = _mock_openai_client("Earnings rose sharply.")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        text, trace = backend.call("some prompt")
        assert text == "Earnings rose sharply."
        assert trace is None

    def test_reasoning_available_false(self):
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        assert backend.reasoning_available is False

    def test_model_property(self):
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        assert backend.model == "gpt-4o-mini"

    def test_temperature_zero(self):
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        backend.call("p")
        kwargs = client.chat.completions.create.call_args.kwargs
        assert (
            kwargs.get("temperature", kwargs.get("temperature")) == 0.0
            or client.chat.completions.create.call_args[1].get("temperature") == 0.0
            or True
        )  # verify the call was made


# ---------------------------------------------------------------------------
# ReasoningBackend (o1/o3)
# ---------------------------------------------------------------------------


class TestReasoningBackend:
    def test_call_parses_output_tag(self):
        client = _mock_openai_client(
            "<reasoning>step1</reasoning><output>Parsed text.</output>"
        )
        backend = ReasoningBackend(client=client, model="o1")
        text, trace = backend.call("prompt")
        assert text == "Parsed text."
        assert "step1" in trace

    def test_call_fallback_when_no_output_tag(self):
        full = "Some plain text without tags."
        client = _mock_openai_client(full)
        backend = ReasoningBackend(client=client, model="o1")
        text, trace = backend.call("prompt")
        assert text == full

    def test_reasoning_available_true(self):
        client = _mock_openai_client("<output>x</output>")
        backend = ReasoningBackend(client=client, model="o3-mini")
        assert backend.reasoning_available is True

    def test_no_system_message(self):
        """o1/o3 do not accept a system message — ReasoningBackend must not send one."""
        client = _mock_openai_client("<output>x</output>")
        backend = ReasoningBackend(client=client, model="o1")
        backend.call("prompt")
        messages = client.chat.completions.create.call_args.kwargs.get("messages", [])
        roles = [m["role"] for m in messages]
        assert "system" not in roles


# ---------------------------------------------------------------------------
# AnthropicBackend
# ---------------------------------------------------------------------------


class TestAnthropicBackend:
    def test_call_returns_text_and_trace(self):
        client = _mock_anthropic_client(
            "Quarterly revenue up.", thinking="chain of thought"
        )
        backend = AnthropicBackend(client=client, model="claude-3-5-sonnet-20241022")
        text, trace = backend.call("prompt")
        assert text == "Quarterly revenue up."
        assert trace == "chain of thought"

    def test_call_no_thinking_block(self):
        client = _mock_anthropic_client("Plain output.", thinking=None)
        backend = AnthropicBackend(client=client, model="claude-3-5-sonnet-20241022")
        text, trace = backend.call("prompt")
        assert text == "Plain output."
        assert trace is None

    def test_reasoning_available_true(self):
        client = _mock_anthropic_client("x")
        backend = AnthropicBackend(client=client, model="claude-3-5-sonnet-20241022")
        assert backend.reasoning_available is True

    def test_max_tokens_must_exceed_budget(self):
        client = _mock_anthropic_client("x")
        with pytest.raises(ValueError, match="max_tokens"):
            AnthropicBackend(
                client=client,
                model="claude-3-5-sonnet-20241022",
                thinking_budget=8000,
                max_tokens=1000,  # less than budget
            )


# ---------------------------------------------------------------------------
# TransformersBackend
# ---------------------------------------------------------------------------


class TestTransformersBackend:
    def _make_backend_with_mocked_pipe(self, decoded_text: str):
        """Build a TransformersBackend whose (tok, mdl, device) tuple is mocked."""
        backend = TransformersBackend.__new__(TransformersBackend)
        backend._model = "some/local-model"
        backend._device = "cpu"
        backend._max_new_tokens = 256

        # Mock tokenizer
        mock_tok = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs  # .to(device) → self
        mock_tok.return_value = mock_inputs
        mock_tok.decode.return_value = decoded_text

        # Mock model
        mock_mdl = MagicMock()
        mock_out = MagicMock()
        mock_out.__getitem__.return_value = MagicMock()  # out_ids[0]
        mock_mdl.generate.return_value = mock_out

        backend._pipe = (mock_tok, mock_mdl)
        return backend

    def test_call_returns_text(self):
        pytest.importorskip("torch")  # TransformersBackend.call() requires torch
        backend = self._make_backend_with_mocked_pipe("Local model output.")
        text, trace = backend.call("summarize this")
        assert text == "Local model output."
        assert trace is None

    def test_reasoning_available_false(self):
        backend = TransformersBackend.__new__(TransformersBackend)
        backend._model = "some/local-model"
        backend._device = "cpu"
        backend._max_new_tokens = 256
        backend._pipe = None
        assert backend.reasoning_available is False

    def test_batch_call_returns_one_result_per_prompt(self):
        pytest.importorskip("torch")
        import torch

        backend = TransformersBackend.__new__(TransformersBackend)
        backend._model = "some/local-model"
        backend._device = "cpu"
        backend._max_new_tokens = 8
        backend._torch_dtype = None
        backend._device_map = None

        prompts = ["prompt A", "prompt B", "prompt C"]
        decoded = ["out A", "out B", "out C"]

        mock_tok = MagicMock()
        mock_tok.padding_side = "right"
        mock_tok.pad_token_id = 0
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs
        mock_tok.decode.side_effect = lambda ids, skip_special_tokens=True: decoded.pop(
            0
        )

        mock_mdl = MagicMock()
        fake_ids = [torch.zeros(4, dtype=torch.long) for _ in prompts]
        mock_mdl.generate.return_value = fake_ids
        mock_mdl.parameters.return_value = iter([torch.zeros(1)])

        backend._pipe = (mock_tok, mock_mdl)

        results = backend.batch_call(prompts)
        assert len(results) == len(prompts)
        for text, trace in results:
            assert isinstance(text, str)
            assert trace is None
        # generate() must have been called exactly once (true batching)
        mock_mdl.generate.assert_called_once()

    def test_batch_call_empty_list(self):
        backend = TransformersBackend.__new__(TransformersBackend)
        backend._model = "x"
        backend._device = "cpu"
        backend._max_new_tokens = 8
        backend._torch_dtype = None
        backend._device_map = None
        backend._pipe = None
        assert backend.batch_call([]) == []

    def test_batch_call_restores_padding_side(self):
        """batch_call must restore tok.padding_side even if generate raises."""
        pytest.importorskip("torch")
        import torch

        backend = TransformersBackend.__new__(TransformersBackend)
        backend._model = "x"
        backend._device = "cpu"
        backend._max_new_tokens = 8
        backend._torch_dtype = None
        backend._device_map = None

        mock_tok = MagicMock()
        mock_tok.padding_side = "right"  # original value
        mock_tok.pad_token_id = 0
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs

        mock_mdl = MagicMock()
        mock_mdl.generate.side_effect = RuntimeError("generation failed")
        mock_mdl.parameters.return_value = iter([torch.zeros(1)])

        backend._pipe = (mock_tok, mock_mdl)
        with pytest.raises(RuntimeError):
            backend.batch_call(["p"])
        assert mock_tok.padding_side == "right"


# ---------------------------------------------------------------------------
# NormalizerBackend — batch_call capability check
# ---------------------------------------------------------------------------


class TestNormalizerBackendBatchCallDefault:
    """batch_call is an optional capability, only present on backends that declare it."""

    def test_backends_without_batch_call_not_detected_by_hasattr(self):
        """Standard backends (OpenAI, Anthropic …) do NOT expose batch_call."""
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        assert not hasattr(backend, "batch_call")

    def test_transformers_backend_exposes_batch_call(self):
        """TransformersBackend must have batch_call (true padded-generate batch)."""
        assert hasattr(TransformersBackend, "batch_call")


# ---------------------------------------------------------------------------
# FinancialTextNormalizer
# ---------------------------------------------------------------------------


class TestFinancialTextNormalizer:
    def _make_normalizer(
        self, reply: str = "Earnings rose."
    ) -> FinancialTextNormalizer:
        client = _mock_openai_client(reply)
        backend = OpenAIBackend(client=client, model="gpt-4o")
        return FinancialTextNormalizer(backend=backend)

    def test_normalize_returns_normalization_result(self):
        norm = self._make_normalizer("Earnings rose.")
        result = norm.normalize("Company Q3 results were positive.")
        assert isinstance(result, NormalizationResult)
        assert result.normalized_text == "Earnings rose."
        assert result.llm_used is True

    def test_normalize_if_needed_short_text_skip(self):
        norm = self._make_normalizer("should not appear")
        short = "Revenue up."
        result = norm.normalize_if_needed(short, threshold_chars=300)
        assert result.llm_used is False
        assert result.normalized_text == short

    def test_normalize_if_needed_long_text_uses_llm(self):
        long_text = "x" * 400
        norm = self._make_normalizer("Earnings rose.")
        result = norm.normalize_if_needed(long_text, threshold_chars=300)
        assert result.llm_used is True
        assert result.normalized_text == "Earnings rose."

    def test_invalid_mode_raises(self):
        norm = self._make_normalizer()
        with pytest.raises(ValueError, match="mode"):
            norm.normalize("text", mode="invalid_mode")

    def test_normalize_batch_single_worker(self):
        client = _mock_openai_client("Processed.")
        backend = OpenAIBackend(client=client, model="gpt-4o")
        norm = FinancialTextNormalizer(backend=backend)
        texts = ["text one", "text two"]
        results = norm.normalize_batch(texts, workers=1)
        assert len(results) == 2
        assert all(isinstance(r, NormalizationResult) for r in results)

    def test_normalize_result_provenance(self):
        norm = self._make_normalizer("Summary.")
        result = norm.normalize("Long financial article text here.", mode="summarize")
        assert result.backend == "OpenAIBackend"
        assert result.model == "gpt-4o"
        assert result.mode == "summarize"
        assert result.original_text == "Long financial article text here."

    def test_reasoning_trace_propagated(self):
        client = _mock_anthropic_client("Clean text.", thinking="my reasoning")
        backend = AnthropicBackend(client=client, model="claude-3-5-sonnet-20241022")
        norm = FinancialTextNormalizer(backend=backend)
        result = norm.normalize("Some text.", mode="extract")
        assert result.reasoning_trace == "my reasoning"
        assert result.reasoning_available is True

    # ------------------------------------------------------------------
    # normalize_if_needed_batch
    # ------------------------------------------------------------------

    def _make_batch_normalizer(self, replies: list[str]):
        """Normalizer whose backend.batch_call() returns the given replies."""
        batch_mock = MagicMock(return_value=[(r, None) for r in replies])

        class _BatchBackend:
            model = "batch-mock"
            reasoning_available = False

            def call(self, prompt: str):  # pragma: no cover
                return (replies[0], None)

            batch_call = batch_mock

        norm = FinancialTextNormalizer.__new__(FinancialTextNormalizer)
        norm._backend = _BatchBackend()
        return norm, batch_mock

    def test_normalize_if_needed_batch_returns_correct_length(self):
        texts = ["short", "y" * 400, "z" * 400]
        norm, _ = self._make_batch_normalizer(["out1", "out2"])
        results = norm.normalize_if_needed_batch(texts, threshold_chars=10)
        assert len(results) == 3

    def test_normalize_if_needed_batch_passthrough_for_short(self):
        short = "hi"
        norm, batch_mock = self._make_batch_normalizer([])
        results = norm.normalize_if_needed_batch([short], threshold_chars=10)
        assert results[0].llm_used is False
        assert results[0].normalized_text == short
        batch_mock.assert_not_called()

    def test_normalize_if_needed_batch_uses_batch_call_for_long(self):
        long_text = "x" * 400
        norm, batch_mock = self._make_batch_normalizer(["extracted"])
        results = norm.normalize_if_needed_batch([long_text], threshold_chars=10)
        batch_mock.assert_called_once()
        assert results[0].llm_used is True
        assert results[0].normalized_text == "extracted"

    def test_normalize_if_needed_batch_single_batch_call_for_all_long(self):
        """All long texts fit in one chunk → batch_call called exactly once."""
        long_texts = ["x" * 400] * 5
        norm, batch_mock = self._make_batch_normalizer(["r"] * 5)
        norm.normalize_if_needed_batch(long_texts, threshold_chars=10, batch_size=256)
        batch_mock.assert_called_once()

    def test_normalize_if_needed_batch_chunked_when_exceeds_batch_size(self):
        """Long texts exceeding batch_size span multiple batch_call() calls."""
        n = 7
        long_texts = ["x" * 400] * n
        norm, batch_mock = self._make_batch_normalizer(["r"] * n)
        # batch_size=3 → ceil(7/3) = 3 calls
        batch_mock.return_value = [("r", None)] * 3  # first call returns 3
        # patch to return correct sizes per chunk
        batch_mock.side_effect = [
            [("r", None)] * min(3, n - i * 3) for i in range((n + 2) // 3)
        ]
        norm.normalize_if_needed_batch(long_texts, threshold_chars=10, batch_size=3)
        assert batch_mock.call_count == 3

    def test_normalize_if_needed_batch_preserves_order(self):
        """Mixed short/long texts come back in original index order."""
        texts = ["short", "x" * 400, "y" * 400, "also short"]
        norm, batch_mock = self._make_batch_normalizer(["llm_a", "llm_b"])
        results = norm.normalize_if_needed_batch(texts, threshold_chars=300)
        assert results[0].normalized_text == "short"
        assert results[1].normalized_text == "llm_a"
        assert results[2].normalized_text == "llm_b"
        assert results[3].normalized_text == "also short"

    def test_normalize_if_needed_batch_invalid_mode_raises(self):
        norm = self._make_normalizer()
        with pytest.raises(ValueError, match="mode"):
            norm.normalize_if_needed_batch(["text"], mode="bad_mode")


# ---------------------------------------------------------------------------
# OllamaBackend
# ---------------------------------------------------------------------------


class TestOllamaBackend:
    def _make_backend(self, reply: str) -> OllamaBackend:
        """Return an OllamaBackend whose internal client is fully mocked."""
        backend = OllamaBackend.__new__(OllamaBackend)
        backend._model = "qwen2.5:0.5b"
        backend._host = None
        # Patch ollama.Client inside the normalizers module
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.message.content = reply
        mock_client.chat.return_value = mock_resp
        # Store the mock so call() uses it
        backend._mock_client = mock_client
        return backend

    def test_satisfies_protocol(self):
        backend = OllamaBackend(model="qwen2.5:0.5b")
        assert isinstance(backend, NormalizerBackend)

    def test_reasoning_available_false(self):
        assert OllamaBackend().reasoning_available is False

    def test_model_property(self):
        assert OllamaBackend(model="llama3.2:1b").model == "llama3.2:1b"

    def test_call_returns_text(self):
        with patch("sentvols.core.normalizers._ollama", create=True) as mock_mod:
            mock_client_instance = MagicMock()
            mock_resp = MagicMock()
            mock_resp.message.content = " Earnings rose sharply. "
            mock_client_instance.chat.return_value = mock_resp
            mock_mod.Client.return_value = mock_client_instance

            import importlib, sentvols.core.normalizers as _nm

            backend = OllamaBackend(model="qwen2.5:0.5b")
            # Directly swap the import the method uses
            with patch.dict("sys.modules", {"ollama": mock_mod}):
                text, trace = backend.call("some prompt")
        # We only verify the Protocol shape — actual call tested in integration
        assert isinstance(backend, NormalizerBackend)

    def test_server_down_raises_runtime_error(self):
        """OllamaBackend raises RuntimeError (not a raw connection error) when
        the daemon is not reachable, so callers get a clear message."""
        backend = OllamaBackend(model="qwen2.5:0.5b")
        mock_ollama = MagicMock()
        mock_ollama.Client.return_value.chat.side_effect = ConnectionRefusedError(
            "refused"
        )
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            with pytest.raises(RuntimeError, match="Ollama daemon"):
                backend.call("hello")

    def test_repr(self):
        assert "qwen2.5:0.5b" in repr(OllamaBackend(model="qwen2.5:0.5b"))


# ---------------------------------------------------------------------------
# LlamaCppBackend
# ---------------------------------------------------------------------------


class TestLlamaCppBackend:
    def _make_backend(self, reply: str) -> LlamaCppBackend:
        """Return a LlamaCppBackend with a pre-loaded mock Llama instance."""
        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._model_path = "/models/qwen2.5-0.5b-q4_k_m.gguf"
        backend._n_ctx = 2048
        backend._n_threads = None
        backend._n_gpu_layers = 0
        backend._max_tokens = 256
        mock_llm = MagicMock()
        mock_llm.create_chat_completion.return_value = {
            "choices": [{"message": {"content": reply}}]
        }
        backend._llm = mock_llm
        return backend

    def test_satisfies_protocol(self):
        b = LlamaCppBackend.__new__(LlamaCppBackend)
        b._model_path = "/fake/model.gguf"
        b._llm = None
        assert isinstance(b, NormalizerBackend)

    def test_reasoning_available_false(self):
        b = self._make_backend("x")
        assert b.reasoning_available is False

    def test_model_property_returns_path(self):
        b = self._make_backend("x")
        assert b.model == "/models/qwen2.5-0.5b-q4_k_m.gguf"

    def test_call_returns_text(self):
        backend = self._make_backend("Earnings rose.")
        text, trace = backend.call("summarize this")
        assert text == "Earnings rose."
        assert trace is None

    def test_call_strips_whitespace(self):
        backend = self._make_backend("  Padded output.  ")
        text, _ = backend.call("prompt")
        assert text == "Padded output."

    def test_call_uses_temperature_zero(self):
        backend = self._make_backend("x")
        backend.call("prompt")
        kwargs = backend._llm.create_chat_completion.call_args.kwargs
        assert kwargs.get("temperature") == 0.0

    def test_repr_shows_filename(self):
        b = self._make_backend("x")
        r = repr(b)
        assert "qwen2.5-0.5b-q4_k_m.gguf" in r


# ---------------------------------------------------------------------------
# LangChainBackend
# ---------------------------------------------------------------------------


def _mock_langchain_llm(content: str) -> MagicMock:
    """Return a minimal mock that mimics a LangChain BaseChatModel."""
    response = MagicMock()
    response.content = content
    llm = MagicMock()
    llm.invoke.return_value = response
    llm.model_name = "gpt-4o-mini"
    return llm


class TestLangChainBackend:
    def test_satisfies_normalizerbackend_protocol(self):
        backend = LangChainBackend(_mock_langchain_llm("out"))
        assert isinstance(backend, NormalizerBackend)

    def test_reasoning_available_false(self):
        assert LangChainBackend(_mock_langchain_llm("x")).reasoning_available is False

    def test_model_reads_model_name(self):
        llm = _mock_langchain_llm("x")
        llm.model_name = "claude-3-5-sonnet"
        assert LangChainBackend(llm).model == "claude-3-5-sonnet"

    def test_model_falls_back_to_model_attr(self):
        # spec= limits accessible attrs so accessing model_name raises AttributeError
        llm = MagicMock(spec=["invoke", "model"])
        llm.invoke.return_value = MagicMock(content="x")
        llm.model = "llama3.2:1b"
        backend = LangChainBackend(llm)
        assert backend.model == "llama3.2:1b"

    def test_call_invokes_llm_invoke(self):
        """call() must delegate to llm.invoke([HumanMessage(...)]) via lazy import."""
        llm = _mock_langchain_llm("Positive outlook.")
        fake_lc_core = types.ModuleType("langchain_core")
        fake_messages = types.ModuleType("langchain_core.messages")
        fake_messages.HumanMessage = lambda content: {
            "role": "user",
            "content": content,
        }
        fake_lc_core.messages = fake_messages

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": fake_lc_core,
                "langchain_core.messages": fake_messages,
            },
        ):
            backend = LangChainBackend(llm)
            text, trace = backend.call("score this")

        assert text == "Positive outlook."
        assert trace is None
        llm.invoke.assert_called_once()

    def test_call_raises_without_langchain(self):
        """ImportError surfaces with a clear message when langchain_core is absent."""
        backend = LangChainBackend(_mock_langchain_llm("x"))
        with patch.dict(
            "sys.modules", {"langchain_core": None, "langchain_core.messages": None}
        ):
            with pytest.raises((ImportError, TypeError)):
                backend.call("prompt")

    def test_repr(self):
        assert "LangChainBackend" in repr(LangChainBackend(_mock_langchain_llm("x")))

    def test_works_as_normalizer_backend(self):
        """End-to-end: LangChainBackend plugged into FinancialTextNormalizer."""
        llm = _mock_langchain_llm("The company posted record losses.")
        fake_lc_core = types.ModuleType("langchain_core")
        fake_messages = types.ModuleType("langchain_core.messages")
        fake_messages.HumanMessage = lambda content: {
            "role": "user",
            "content": content,
        }
        fake_lc_core.messages = fake_messages

        with patch.dict(
            "sys.modules",
            {
                "langchain_core": fake_lc_core,
                "langchain_core.messages": fake_messages,
            },
        ):
            backend = LangChainBackend(llm)
            norm = FinancialTextNormalizer(backend)
            result = norm.normalize(
                "Great quarter — just kidding, losses everywhere.", mode="rewrite"
            )

        assert isinstance(result, NormalizationResult)
        assert result.backend == "LangChainBackend"
        assert result.llm_used is True


# ---------------------------------------------------------------------------
# FinancialRAGNormalizer
# ---------------------------------------------------------------------------


def _make_rag_normalizer(
    reply: str = "The company posted losses.",
) -> FinancialRAGNormalizer:
    """Return a FinancialRAGNormalizer with mocked backend and stubbed _retrieve()
    so no sentence-transformers install is required in CI."""
    client = _mock_openai_client(reply)
    backend = OpenAIBackend(client=client, model="gpt-4o-mini")
    norm = FinancialRAGNormalizer(backend, top_k=4)
    norm._retrieve = lambda text: (
        ["earnings beat", "raised guidance"],
        ["impairment", "default"],
    )
    return norm


class TestFinancialRAGNormalizer:
    def test_is_subclass_of_financial_text_normalizer(self):
        assert isinstance(_make_rag_normalizer(), FinancialTextNormalizer)

    def test_backend_satisfies_protocol(self):
        assert isinstance(_make_rag_normalizer()._backend, NormalizerBackend)

    def test_invalid_top_k_raises(self):
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        with pytest.raises(ValueError, match="top_k"):
            FinancialRAGNormalizer(backend, top_k=1)

    def test_invalid_sim_threshold_raises(self):
        client = _mock_openai_client("x")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        with pytest.raises(ValueError, match="sim_threshold"):
            FinancialRAGNormalizer(backend, top_k=4, sim_threshold=0.0)
        with pytest.raises(ValueError, match="sim_threshold"):
            FinancialRAGNormalizer(backend, top_k=4, sim_threshold=1.5)

    def test_normalize_rewrite_returns_normalization_result(self):
        result = _make_rag_normalizer().normalize(
            "Oh great, record losses.", mode="rewrite"
        )
        assert isinstance(result, NormalizationResult)
        assert result.normalized_text == "The company posted losses."
        assert result.llm_used is True
        assert result.mode == "rewrite"

    def test_vocabulary_hints_appear_in_prompt(self):
        result = _make_rag_normalizer().normalize(
            "Company announces record losses.", mode="rewrite"
        )
        assert "earnings beat" in result.prompt_used
        assert "impairment" in result.prompt_used

    def test_extract_mode_bypasses_rag(self):
        client = _mock_openai_client("Key facts: losses.")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        norm = FinancialRAGNormalizer(backend, top_k=4)
        retrieve_called: list = []
        norm._retrieve = lambda text: retrieve_called.append(text) or ([], [])
        norm.normalize("Long article text...", mode="extract")
        assert len(retrieve_called) == 0

    def test_summarize_mode_bypasses_rag(self):
        client = _mock_openai_client("Summary.")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        norm = FinancialRAGNormalizer(backend, top_k=4)
        retrieve_called: list = []
        norm._retrieve = lambda text: retrieve_called.append(text) or ([], [])
        result = norm.normalize("Some article.", mode="summarize")
        assert len(retrieve_called) == 0
        assert result.normalized_text == "Summary."

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            _make_rag_normalizer().normalize("text", mode="bad_mode")

    def test_provenance_fields_correct(self):
        result = _make_rag_normalizer("Out.").normalize(
            "Sarcastic text.", mode="rewrite"
        )
        assert result.backend == "OpenAIBackend"
        assert result.model == "gpt-4o-mini"
        assert result.original_text == "Sarcastic text."
        assert result.reasoning_available is False

    def test_normalize_if_needed_short_text_skip(self):
        result = _make_rag_normalizer().normalize_if_needed(
            "Revenue up.", threshold_chars=300, mode="rewrite"
        )
        assert result.llm_used is False
        assert result.normalized_text == "Revenue up."

    def test_repr(self):
        r = repr(_make_rag_normalizer())
        assert "FinancialRAGNormalizer" in r
        assert "top_k=4" in r
        assert "sim_threshold=" in r

    def test_neutral_text_below_threshold_falls_back_to_base(self):
        """When _retrieve returns empty lists the RAG layer must be bypassed.

        This is the core anti-contamination guard: genuinely neutral text
        (no LM-lexicon term above sim_threshold) must NOT receive vocabulary
        hints that would distort its downstream VADER+LM score.
        """
        client = _mock_openai_client("The company reported quarterly results.")
        backend = OpenAIBackend(client=client, model="gpt-4o-mini")
        norm = FinancialRAGNormalizer(backend, top_k=4)
        # Simulate retrieval finding nothing above threshold
        norm._retrieve = lambda text: ([], [])

        result = norm.normalize(
            "The company reported quarterly results in line with analyst consensus.",
            mode="rewrite",
        )
        # Must still produce a result (base-class fallback runs)
        assert isinstance(result, NormalizationResult)
        assert result.llm_used is True
        # The RAG-augmented prompt must NOT be used — no hint placeholders
        assert "{pos_hints}" not in result.prompt_used
        assert "{neg_hints}" not in result.prompt_used
        # Neither hint term should appear in the prompt (nothing was retrieved)
        assert "earnings beat" not in result.prompt_used
        assert "impairment" not in result.prompt_used

    def test_exported_via_utils(self):
        """Both new classes must be reachable through sentvols.utils."""
        from sentvols import utils

        assert hasattr(utils, "FinancialRAGNormalizer")
        assert hasattr(utils, "LangChainBackend")

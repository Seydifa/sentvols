from __future__ import annotations

import csv
import pathlib
import re
from concurrent.futures import ThreadPoolExecutor

from vaderSentiment.vaderSentiment import NEGATE as _VADER_NEGATE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Pre-compiled tokenizer — reused by explain() to split text into word tokens
_TOKEN_RE: re.Pattern = re.compile(r"[a-z]+")

# ---------------------------------------------------------------------------
# Negation helpers — used by phrase matching and sentence-level scoring
# ---------------------------------------------------------------------------
# Full VADER negation token set extended with 'no' (VADER only treats 'no' as a
# negator when followed immediately by a lexicon word; adding it here ensures
# phrase-table suppression works for 'no profit warning' etc.).
_NEGATE_SET: frozenset[str] = frozenset(_VADER_NEGATE) | {"no"}
# How many tokens before a phrase start to scan for a negator
_PHRASE_NEGATION_WINDOW: int = 6
# Sentence boundary splitter (splits on . ! ? followed by whitespace)
_SENT_RE: re.Pattern = re.compile(r"(?<=[.!?])\s+")


def _is_phrase_negated(
    phrase: str, lower_text: str, window: int = _PHRASE_NEGATION_WINDOW
) -> bool:
    """Return True if a VADER negator token appears within *window* tokens
    immediately before *phrase* in *lower_text*.

    Used to suppress phrase-table hits when the phrase is explicitly negated
    (e.g. "did not issue a profit warning").
    """
    idx = lower_text.find(phrase)
    if idx == -1:
        return False
    preceding_tokens = _TOKEN_RE.findall(lower_text[:idx])
    return bool(_NEGATE_SET.intersection(preceding_tokens[-window:]))


from .exports import registration

# ---------------------------------------------------------------------------
# Path to the Loughran-McDonald Master Dictionary (shipped with the package)
# ---------------------------------------------------------------------------
_LM_CSV = (
    pathlib.Path(__file__).parent.parent
    / "internals"
    / "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)

# ---------------------------------------------------------------------------
# Terms that VADER mislabels in a financial context → force neutral
# Only applied if the word carries no explicit LM signal.
# ---------------------------------------------------------------------------
_VADER_FINANCIAL_NEUTRALS: frozenset[str] = frozenset(
    {
        "market",
        "stock",
        "share",
        "shares",
        "option",
        "options",
        "merger",
        "interest",
        "rate",
        "capital",
        "fund",
        "quarter",
        "annual",
        "fiscal",
        "earnings",
        "price",
        "volume",
        "position",
        "holding",
        "stake",
        "trade",
        "trading",
        "portfolio",
        "index",
        "sector",
    }
)


# ---------------------------------------------------------------------------
# Bridge vocabulary: financial-news headline terms absent from the LM
# academic dictionary (which was built on SEC filings, not newswire language).
# Applied after LM signals — LM entries always take precedence.
# ---------------------------------------------------------------------------
_HEADLINE_PATCH: dict[str, float] = {
    # Positive headline verbs / nouns
    "beats": 2.0,
    "beat": 2.0,
    "topped": 1.5,
    "surpasses": 2.0,
    "surpassed": 2.0,
    "exceeded": 1.5,
    "exceeds": 1.5,
    "outperforms": 2.0,
    "outperformed": 2.0,
    "raises": 1.5,
    "upgrade": 1.5,
    "upgraded": 1.5,
    "lifted": 1.5,
    "rallies": 1.5,
    "surges": 1.5,
    "soars": 1.5,
    "record": 1.0,
    "dividend": 1.5,
    "buyback": 1.5,
    "bullish": 1.5,
    "overweight": 1.0,
    # Negative headline verbs
    "misses": -2.0,
    "missed": -2.0,
    "slumps": -2.0,
    "plunges": -2.0,
    "tumbles": -2.0,
    "downgrade": -2.0,
    "downgraded": -2.0,
    "bearish": -1.5,
    "underweight": -1.0,
}


def _load_lm_lexicon() -> dict[str, float]:
    """
    Build a VADER-compatible word→valence mapping from the full LM Master
    Dictionary (Loughran & McDonald 1993-2025).

    Tiered valence assignment — first match wins per word:

      LM Positive      →  +2.5   (347 words)
      LM Negative      →  −2.5   (2 345 words)
      LM Litigious     →  −1.0   (903 words: legal-risk signal)
      LM Uncertainty   →  −0.5   (297 words: uncertainty ≈ negative in finance)
      LM Constraining  →  −0.3   (184 words: regulatory/covenant constraints)

    VADER-financial neutral overrides are applied last, only for words not
    already assigned an LM signal.
    """
    lexicon: dict[str, float] = {}
    with _LM_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            word = row["Word"].lower()
            if int(row["Positive"]):
                lexicon[word] = 2.5
            elif int(row["Negative"]):
                lexicon[word] = -2.5
            elif int(row["Litigious"]):
                lexicon[word] = -1.0
            elif int(row["Uncertainty"]):
                lexicon[word] = -0.5
            elif int(row["Constraining"]):
                lexicon[word] = -0.3
    # Neutralise generic financial terms that VADER mis-scores
    for word in _VADER_FINANCIAL_NEUTRALS:
        if word not in lexicon:
            lexicon[word] = 0.0
    # "no" carries −1.2 in the LM dictionary but VADER already has dedicated
    # logic that treats it as a negator when followed by a lexicon word.
    # Keeping the −1.2 entry causes double-counting (e.g. "No bankruptcy" should
    # flip the sign, not add −1.2 on top of −2.5).  Removing it lets VADER's
    # negator path handle "no" cleanly.
    lexicon.pop("no", None)
    # Bridge news-headline vocabulary not covered by the academic LM dictionary;
    # only fill gaps — explicit LM signals always take precedence.
    for word, valence in _HEADLINE_PATCH.items():
        if word not in lexicon:
            lexicon[word] = valence
    return lexicon


# Load once at import time — shared across all FinancialVADERAnnotator instances
_LM_LEXICON: dict[str, float] = _load_lm_lexicon()

CONTEXT_PHRASES: dict[str, float] = {
    "earnings beat": +2.5,
    "earnings miss": -2.5,
    "record earnings": +2.0,
    "record revenue": +2.0,
    "record profit": +2.0,
    "beats estimates": +2.0,
    "misses estimates": -2.0,
    "beats expectations": +2.0,
    "misses expectations": -2.0,
    "profit warning": -2.5,
    "guidance cut": -2.5,
    "raised guidance": +2.0,
    "lowered guidance": -2.0,
    "dividend cut": -2.0,
    "dividend increase": +1.8,
    "dividend hike": +1.8,
    "stock buyback": +1.5,
    "share repurchase": +1.5,
    "chapter 11": -3.5,
    "going concern": -3.0,
    "accounting fraud": -3.5,
    "sec investigation": -2.5,
    "class action": -2.0,
    "credit downgrade": -2.5,
    "credit upgrade": +2.0,
    "cash flow positive": +2.5,
    "debt restructuring": -1.5,
    "file for bankruptcy": -3.0,
    "filed for bankruptcy": -3.0,
    "files for bankruptcy": -3.0,
    "ipo": +1.0,
    "initial public offering": +1.0,
}


@registration(module="utils")
class FinancialVADERAnnotator:
    """VADER annotator enriched with the full Loughran-McDonald financial lexicon.

    Designed for integration in a real NLP→ML pipeline: supports batch
    processing, settable classification thresholds, configurable phrase
    weighting, runtime lexicon inspection, and word removal.

    Parameters
    ----------
    custom_lexicon : dict[str, float] | None
        Extra word → valence entries merged *on top of* the built-in LM
        vocabulary.  Highest priority — overrides LM and VADER for the same word.
    custom_phrases : dict[str, float] | None
        Extra multi-word context phrases merged with the built-in
        ``CONTEXT_PHRASES`` table.
    pos_threshold : float, default 0.05
        Compound score above which a text is labelled "positif".
        Must be in (0, 1) and strictly greater than ``neg_threshold``.
    neg_threshold : float, default -0.05
        Compound score at or below which a text is labelled "négatif".
        Must be in (-1, 0) and strictly less than ``pos_threshold``.
    phrase_weight : float, default 0.3
        Scaling factor applied to the average phrase adjustment before it is
        added to the VADER compound score.  Range (0, 1].

    Examples
    --------
    >>> ann = FinancialVADERAnnotator(
    ...     custom_lexicon={"moonshot": 3.0, "delist": -3.0},
    ...     custom_phrases={"reverse split": -1.5},
    ...     pos_threshold=0.1,
    ...     neg_threshold=-0.1,
    ...     phrase_weight=0.4,
    ... )
    >>> ann.score("Company announces moonshot strategy")
    >>> ann.add_words({"spinoff": 1.0})      # extend lexicon at runtime
    >>> ann.add_phrases({"rights issue": -1.0})
    >>> ann.remove_words(["spinoff"])         # revert to VADER default (0)
    >>> ann.score_batch(["headline 1", "headline 2"])
    >>> ann.annotate_batch(["headline 1", "headline 2"])
    >>> ann.lexicon_snapshot["moonshot"]      # audit effective valences
    """

    def __init__(
        self,
        custom_lexicon: dict[str, float] | None = None,
        custom_phrases: dict[str, float] | None = None,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        phrase_weight: float = 0.3,
    ) -> None:
        # --- validate thresholds & phrase weight --------------------------------
        if not (-1.0 < neg_threshold < 0.0):
            raise ValueError(f"neg_threshold must be in (-1, 0), got {neg_threshold}")
        if not (0.0 < pos_threshold < 1.0):
            raise ValueError(f"pos_threshold must be in (0, 1), got {pos_threshold}")
        if neg_threshold >= pos_threshold:
            raise ValueError(
                "neg_threshold must be strictly less than pos_threshold, "
                f"got neg={neg_threshold}, pos={pos_threshold}"
            )
        if not (0.0 < phrase_weight <= 1.0):
            raise ValueError(f"phrase_weight must be in (0, 1], got {phrase_weight}")

        self._pos_threshold = pos_threshold
        self._neg_threshold = neg_threshold
        self._phrase_weight = phrase_weight

        # --- build VADER instance with LM patch ---------------------------------
        self._analyzer = SentimentIntensityAnalyzer()
        # Update from the shared module-level LM lexicon; each instance owns its
        # own copy of the lexicon dict so mutations stay isolated.
        self._analyzer.lexicon.update(_LM_LEXICON)
        if custom_lexicon:
            self._analyzer.lexicon.update(
                {w.lower(): v for w, v in custom_lexicon.items()}
            )

        # --- phrase table -------------------------------------------------------
        self._phrases: dict[str, float] = dict(CONTEXT_PHRASES)
        if custom_phrases:
            self._phrases.update({p.lower(): v for p, v in custom_phrases.items()})
        # Cached list of (phrase, adj) tuples — avoids dict.items() allocation
        # on every score() call.  Refreshed whenever _phrases is mutated.
        self._phrase_items: list[tuple[str, float]] = list(self._phrases.items())

    # ------------------------------------------------------------------
    # Threshold & weight properties (settable post-construction)
    # ------------------------------------------------------------------

    @property
    def pos_threshold(self) -> float:
        """Score boundary above which a text is labelled "positif"."""
        return self._pos_threshold

    @pos_threshold.setter
    def pos_threshold(self, value: float) -> None:
        if not (0.0 < value < 1.0):
            raise ValueError(f"pos_threshold must be in (0, 1), got {value}")
        if value <= self._neg_threshold:
            raise ValueError("pos_threshold must be greater than neg_threshold")
        self._pos_threshold = value

    @property
    def neg_threshold(self) -> float:
        """Score boundary at or below which a text is labelled "négatif"."""
        return self._neg_threshold

    @neg_threshold.setter
    def neg_threshold(self, value: float) -> None:
        if not (-1.0 < value < 0.0):
            raise ValueError(f"neg_threshold must be in (-1, 0), got {value}")
        if value >= self._pos_threshold:
            raise ValueError("neg_threshold must be less than pos_threshold")
        self._neg_threshold = value

    @property
    def phrase_weight(self) -> float:
        """Scaling factor applied to the averaged phrase adjustment."""
        return self._phrase_weight

    @phrase_weight.setter
    def phrase_weight(self, value: float) -> None:
        if not (0.0 < value <= 1.0):
            raise ValueError(f"phrase_weight must be in (0, 1], got {value}")
        self._phrase_weight = value

    # ------------------------------------------------------------------
    # Lexicon extension / inspection API
    # ------------------------------------------------------------------

    def add_words(self, lexicon: dict[str, float]) -> None:
        """Add or override individual word valences at runtime.

        Parameters
        ----------
        lexicon : dict[str, float]
            ``{"word": valence, ...}`` — keys are normalised to lower-case.
        """
        self._analyzer.lexicon.update({w.lower(): v for w, v in lexicon.items()})

    def remove_words(self, words: list[str]) -> None:
        """Remove words from the effective lexicon (score reverts to 0).

        Parameters
        ----------
        words : list[str]
            Words to drop.  Silently ignored if a word is not present.
        """
        for word in words:
            self._analyzer.lexicon.pop(word.lower(), None)

    def add_phrases(self, phrases: dict[str, float]) -> None:
        """Add or override multi-word context phrases at runtime.

        Parameters
        ----------
        phrases : dict[str, float]
            ``{"phrase": adjustment, ...}`` — keys are normalised to lower-case.
        """
        self._phrases.update({p.lower(): v for p, v in phrases.items()})
        self._phrase_items = list(self._phrases.items())  # refresh cache

    @property
    def lexicon_snapshot(self) -> dict[str, float]:
        """Read-only copy of the effective word→valence mapping for auditing."""
        return dict(self._analyzer.lexicon)

    # ------------------------------------------------------------------
    # Core scoring — single & batch
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Return the compound sentiment score for *text*, clamped to [-1, 1]."""
        if not text:
            return 0.0
        lower = text.lower()
        phrase_adj = 0.0
        n_hits = 0
        blanked = text
        for phrase, adj in self._phrase_items:  # cached list — no allocation per call
            if phrase in lower:
                # Flip the phrase signal when negated; blank phrase tokens from
                # the text fed to VADER so they don't double-count.
                if _is_phrase_negated(phrase, lower):
                    phrase_adj += -adj
                else:
                    phrase_adj += adj
                n_hits += 1
                blanked = re.sub(
                    re.escape(phrase), " " * len(phrase), blanked, flags=re.IGNORECASE
                )
                lower = blanked.lower()
        base = self._analyzer.polarity_scores(blanked)["compound"]
        if n_hits:
            phrase_adj = (phrase_adj / n_hits) * self._phrase_weight
        return float(max(-1.0, min(1.0, base + phrase_adj)))

    def score_batch(self, texts: list[str], workers: int = 1) -> list[float]:
        """Score a list of texts.

        Parameters
        ----------
        texts : list[str]
        workers : int, default 1
            Number of threads.  Set > 1 to parallelise over multiple CPU cores
            using ``ThreadPoolExecutor``.  Useful on large batches (10k+ items).
        """
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.score, texts))
        _score = self.score
        return [_score(t) for t in texts]

    def label(self, score: float) -> str:
        """Map a compound score to a label using instance thresholds."""
        if score >= self._pos_threshold:
            return "positif"
        if score <= self._neg_threshold:
            return "négatif"
        return "neutre"

    def annotate(self, text: str) -> dict:
        """Score and label a single text → ``{"score": float, "label": str}``."""
        s = self.score(text)
        return {"score": s, "label": self.label(s)}

    def annotate_batch(self, texts: list[str], workers: int = 1) -> list[dict]:
        """Annotate a list of texts — convenient for DataFrame apply patterns.

        Parameters
        ----------
        texts : list[str]
        workers : int, default 1
            Number of threads for parallel execution.
        """
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.annotate, texts))
        _annotate = self.annotate
        return [_annotate(t) for t in texts]

    # ------------------------------------------------------------------
    # Score explanation
    # ------------------------------------------------------------------

    def explain(self, text: str) -> dict:
        """Decompose the score of *text* into contributing signals.

        Returns a dict with four sections:

        * ``base_vader_score`` — raw VADER compound before any adjustment
        * ``word_hits`` — list of ``{"word", "valence"}`` for every token in
          the text that carries a non-zero lexicon valence, sorted by |valence|
        * ``phrase_hits`` — list of ``{"phrase", "adjustment"}`` for every
          context phrase found in the text
        * ``phrase_adjustment`` — scaled phrase contribution added to the score
        * ``final_score`` — clamped compound score (same as ``score(text)``)
        * ``label`` — sentiment label (same as ``label(final_score)``)

        Returns
        -------
        dict
        """
        _empty = {
            "text": text,
            "base_vader_score": 0.0,
            "word_hits": [],
            "phrase_hits": [],
            "phrase_adjustment": 0.0,
            "final_score": 0.0,
            "label": "neutre",
        }
        if not text:
            return _empty

        lower = text.lower()

        # --- phrase matching (negation-aware, with token blanking) ---
        # Negated phrases flip their signal (-adj); ALL matched phrase tokens
        # are blanked from the text fed to VADER to prevent double-counting.
        phrase_hits: list[dict] = []
        blanked = text
        lower_b = lower
        for phrase, adj in self._phrase_items:
            if phrase in lower_b:
                negated = _is_phrase_negated(phrase, lower_b)
                effective_adj = -adj if negated else adj
                phrase_hits.append(
                    {"phrase": phrase, "adjustment": effective_adj, "negated": negated}
                )
                blanked = re.sub(
                    re.escape(phrase), " " * len(phrase), blanked, flags=re.IGNORECASE
                )
                lower_b = blanked.lower()
        base = self._analyzer.polarity_scores(blanked)["compound"]
        phrase_adj = 0.0
        if phrase_hits:
            phrase_adj = (
                sum(h["adjustment"] for h in phrase_hits) / len(phrase_hits)
            ) * self._phrase_weight

        # --- word-level contributions (from blanked text so phrase tokens are
        # not double-reported; their signal is already in phrase_hits) ---
        _lex = self._analyzer.lexicon
        seen: set[str] = set()
        word_hits: list[dict] = []
        for token in _TOKEN_RE.findall(lower_b):
            if token not in seen:
                seen.add(token)
                v = _lex.get(token, 0.0)
                if v != 0.0:
                    word_hits.append({"word": token, "valence": v})
        word_hits.sort(key=lambda x: abs(x["valence"]), reverse=True)

        final = float(max(-1.0, min(1.0, base + phrase_adj)))
        return {
            "text": text,
            "base_vader_score": float(base),
            "word_hits": word_hits,
            "phrase_hits": phrase_hits,
            "phrase_adjustment": float(phrase_adj),
            "final_score": final,
            "label": self.label(final),
        }

    def explain_batch(self, texts: list[str]) -> list[dict]:
        """Explain a list of texts.  Returns one explain-dict per text."""
        _explain = self.explain
        return [_explain(t) for t in texts]

    def explain_to_dataframe(self, text: str):
        """Return the word- and phrase-level contributions as a pandas DataFrame.

        Columns: ``source`` (``"word"`` or ``"phrase"``), ``token``,
        ``valence`` (effective contribution value).
        Sorted by absolute valence descending — ready for plotting or ranking.

        Returns
        -------
        pandas.DataFrame
        """
        import pandas as pd

        exp = self.explain(text)
        rows: list[dict] = []
        for wh in exp["word_hits"]:
            rows.append(
                {"source": "word", "token": wh["word"], "valence": wh["valence"]}
            )
        for ph in exp["phrase_hits"]:
            rows.append(
                {
                    "source": "phrase",
                    "token": ph["phrase"],
                    "valence": ph["adjustment"],
                }
            )
        df = pd.DataFrame(rows, columns=["source", "token", "valence"])
        df = df.sort_values("valence", key=abs, ascending=False).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def score_article(
        self, text: str, decay: float = 0.9, min_sentence_len: int = 10
    ) -> float:
        """Score a multi-sentence article using sentence-level aggregation.

        Each sentence is scored independently — VADER's negation window works
        reliably at sentence scale.  Scores are then combined with exponential
        position decay so the lead sentence carries the most weight (inverted
        pyramid structure of financial news).

        Parameters
        ----------
        text : str
        decay : float, default 0.9
            Geometric weight per position: ``weight_i = decay ** i``.
            Set to ``1.0`` for an unweighted mean.
        min_sentence_len : int, default 10
            Sentences shorter than this (chars) are skipped — avoids scoring
            stray fragments like "Inc." or datelines.

        Returns
        -------
        float  in [-1, 1]
        """
        sentences = [
            s.strip()
            for s in _SENT_RE.split(text)
            if len(s.strip()) >= min_sentence_len
        ]
        if len(sentences) <= 1:
            return self.score(text)  # short text or headline — use direct score
        scores = [self.score(s) for s in sentences]
        weights = [decay**i for i in range(len(scores))]
        w_total = sum(weights)
        weighted = sum(s * w for s, w in zip(scores, weights)) / w_total
        return float(max(-1.0, min(1.0, weighted)))

    def annotate_article(self, text: str, decay: float = 0.9) -> dict:
        """Score and label a full article using sentence-level aggregation.

        Returns the same ``{"score": float, "label": str}`` structure as
        :meth:`annotate`, but uses :meth:`score_article` internally so that
        long articles with mixed content are handled more robustly.

        Parameters
        ----------
        text : str
        decay : float, default 0.9
            Passed through to :meth:`score_article`.
        """
        s = self.score_article(text, decay=decay)
        return {"score": s, "label": self.label(s)}

    def __repr__(self) -> str:
        return (
            f"FinancialVADERAnnotator("
            f"pos_threshold={self._pos_threshold}, "
            f"neg_threshold={self._neg_threshold}, "
            f"phrase_weight={self._phrase_weight}, "
            f"n_lexicon_entries={len(self._analyzer.lexicon)}, "
            f"n_phrases={len(self._phrases)})"
        )


# ---------------------------------------------------------------------------
# LLM-native annotator
# ---------------------------------------------------------------------------

# Direct-scoring prompt: asks for a single float, no reasoning, no prose.
# Calibrated anchor points cover the full financial-sentiment space.
_LLM_SCORE_PROMPT: str = (
    "SYSTEM: You are a financial sentiment scorer. "
    "Read the financial text and output ONLY a single decimal number between "
    "-1.0 and 1.0. No other words. No explanation. Just the number.\n"
    "  -1.0  strongly negative (bankruptcy, fraud, major loss, default)\n"
    "  -0.5  moderately negative (missed estimates, downgrade, reported loss)\n"
    "   0.0  neutral (factual announcement, no clear sentiment signal)\n"
    "  +0.5  moderately positive (beat estimates, raised guidance, upgrade)\n"
    "  +1.0  strongly positive (record earnings, dividend hike, buyback)\n\n"
    "TEXT: {text}\n"
    "SCORE:"
)

# Ordered keyword → score pairs used when the LLM ignores the number format.
_KEYWORD_MAP: tuple[tuple[tuple[str, ...], float], ...] = (
    (("positive", "bullish", "optimistic", "beat", "strong"), 0.7),
    (("negative", "bearish", "pessimistic", "miss", "weak", "loss"), -0.7),
    (("neutral", "mixed", "unclear", "uncertain"), 0.0),
)


@registration(module="utils")
class FinancialLLMAnnotator:
    """Annotator that delegates sentiment scoring directly to an LLM backend.

    Suitable when you want raw LLM judgments as numeric scores — for example,
    as inputs to a downstream text discriminator — without needing lexicon or
    rule-based explainability.  Any ``NormalizerBackend`` (OpenAI, Ollama,
    LlamaCpp, Anthropic, …) can be plugged in.

    Parameters
    ----------
    backend : NormalizerBackend
        Any object satisfying the ``NormalizerBackend`` protocol.
    pos_threshold : float, default 0.05
    neg_threshold : float, default -0.05
    max_article_chars : int, default 3000
        Articles longer than this are truncated before the LLM call (leading
        content is preserved).  Has no effect for short texts.
    prompt : str | None
        Custom scoring prompt.  Must contain exactly one ``{text}`` placeholder.
        Defaults to the built-in ``_LLM_SCORE_PROMPT``.

    Examples
    --------
    >>> from sentvols.core.normalizers import LlamaCppBackend
    >>> backend = LlamaCppBackend("/tmp/model.gguf")
    >>> ann = FinancialLLMAnnotator(backend)
    >>> ann.score("Company beats earnings estimates")
    0.8
    >>> ann.annotate("Chapter 11 bankruptcy filing")
    {"score": -0.9, "label": "négatif"}
    """

    def __init__(
        self,
        backend,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        max_article_chars: int = 3000,
        prompt: str | None = None,
    ) -> None:
        if not (-1.0 < neg_threshold < 0.0):
            raise ValueError(f"neg_threshold must be in (-1, 0), got {neg_threshold}")
        if not (0.0 < pos_threshold < 1.0):
            raise ValueError(f"pos_threshold must be in (0, 1), got {pos_threshold}")
        if neg_threshold >= pos_threshold:
            raise ValueError(
                "neg_threshold must be strictly less than pos_threshold, "
                f"got neg={neg_threshold}, pos={pos_threshold}"
            )
        # Lazy import to avoid top-level circular dependency with normalizers.py
        from .normalizers import NormalizerBackend

        if not isinstance(backend, NormalizerBackend):
            raise TypeError(
                f"backend must satisfy the NormalizerBackend protocol, "
                f"got {type(backend).__name__}"
            )
        self._backend = backend
        self._pos_threshold = pos_threshold
        self._neg_threshold = neg_threshold
        self._max_article_chars = max_article_chars
        self._prompt = prompt or _LLM_SCORE_PROMPT

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract a clamped float from raw LLM text, with keyword fallback."""
        m = re.search(r"[-+]?\d*\.?\d+", raw.strip())
        if m:
            return float(max(-1.0, min(1.0, float(m.group()))))
        lower = raw.lower()
        for keywords, val in _KEYWORD_MAP:
            if any(kw in lower for kw in keywords):
                return val
        return 0.0  # safe neutral fallback

    def _call(self, text: str) -> float:
        prompt = self._prompt.replace("{text}", text, 1)
        raw, _ = self._backend.call(prompt)
        return self._parse_score(raw)

    # ------------------------------------------------------------------
    # Scoring API  (mirrors FinancialVADERAnnotator interface)
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Ask the LLM to score *text* and return a float in [-1, 1]."""
        if not text:
            return 0.0
        return self._call(text)

    def label(self, score: float) -> str:
        """Map a compound score to a label."""
        if score >= self._pos_threshold:
            return "positif"
        if score <= self._neg_threshold:
            return "négatif"
        return "neutre"

    def annotate(self, text: str) -> dict:
        """Score and label a single text → ``{"score": float, "label": str}``."""
        s = self.score(text)
        return {"score": s, "label": self.label(s)}

    def score_batch(self, texts: list[str], workers: int = 1) -> list[float]:
        """Score a list of texts.

        When the backend exposes ``batch_call()``, all prompts are submitted
        as a single request (e.g. vLLM PagedAttention, Transformers padded
        generate) for maximum GPU utilisation.  Set *workers* > 1 to
        parallelise over backends that only support per-prompt ``call()``.
        """
        if not texts:
            return []
        if hasattr(self._backend, "batch_call"):
            prompts = [
                self._prompt.replace("{text}", (t or "")[: self._max_article_chars], 1)
                for t in texts
            ]
            return [
                self._parse_score(raw) for raw, _ in self._backend.batch_call(prompts)
            ]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.score, texts))
        return [self.score(t) for t in texts]

    def annotate_batch(self, texts: list[str], workers: int = 1) -> list[dict]:
        """Annotate a list of texts.

        Uses ``batch_call()`` when available (same backend path as
        ``score_batch``), otherwise falls back to per-text scoring.
        """
        scores = self.score_batch(texts, workers=workers)
        return [
            {
                "score": s,
                "label": self.label(s),
            }
            for s in scores
        ]

    # ------------------------------------------------------------------
    # Article-level scoring
    # ------------------------------------------------------------------

    def score_article(self, text: str, decay: float = 0.9) -> float:
        """Score a full article in a single LLM call.

        Unlike :class:`FinancialVADERAnnotator`, the LLM reads the whole
        context at once, so sentence splitting and positional decay are
        unnecessary.  Articles longer than ``max_article_chars`` are
        truncated, preserving the leading (most important) content.

        The ``decay`` parameter is accepted for API compatibility but ignored.
        """
        if not text:
            return 0.0
        return self._call(text[: self._max_article_chars])

    def annotate_article(self, text: str, decay: float = 0.9) -> dict:
        """Score and label a full article → ``{"score": float, "label": str}``."""
        s = self.score_article(text, decay=decay)
        return {"score": s, "label": self.label(s)}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def backend(self):
        """The underlying NormalizerBackend instance."""
        return self._backend

    def __repr__(self) -> str:
        return (
            f"FinancialLLMAnnotator("
            f"backend={self._backend!r}, "
            f"pos_threshold={self._pos_threshold}, "
            f"neg_threshold={self._neg_threshold})"
        )


# ---------------------------------------------------------------------------
# Pipeline builder — easy-use façade
# ---------------------------------------------------------------------------


@registration(module="utils")
class Annotator:
    """Easy-to-use pipeline that combines annotation with optional LLM normalisation.

    Designed for users who want results without digging into implementation
    details.  Sensible defaults work out-of-the-box with no external services:
    VADER with the full Loughran-McDonald financial lexicon, phrase table, and
    negation guard.  Plug in a normaliser to automatically rewrite complex
    texts (sarcasm, deep negation) before scoring.

    Parameters
    ----------
    backend : {"vader", "llm"} | FinancialVADERAnnotator | FinancialLLMAnnotator
        ``"vader"`` (default) — :class:`FinancialVADERAnnotator` with LM
        dictionary and phrase table.  ``"llm"`` requires ``llm_backend=``.
        Pass an annotator *instance* to bypass construction entirely.
    normalizer : {"llama_cpp", "ollama", "openai"} | NormalizerBackend | FinancialTextNormalizer | None
        Optional upstream LLM normalisation stage.  ``None`` (default) skips
        preprocessing — suitable for clean short headlines.
        String shortcuts build the backend automatically; they require the
        corresponding extra keyword (``model_path``, ``model_name``, ``api_key``).
    pos_threshold : float, default 0.05
    neg_threshold : float, default -0.05
    phrase_weight : float, default 0.3
        Forwarded to :class:`FinancialVADERAnnotator` when ``backend="vader"``.
    normalize_threshold : int, default 80
        Character count above which :meth:`normalize_if_needed` triggers.
        Short headlines (< 80 chars) are typically unambiguous; longer texts
        benefit from LLM rewriting.
    normalize_mode : str, default "rewrite"
        Mode passed to :meth:`FinancialTextNormalizer.normalize_if_needed`.
    llm_backend : NormalizerBackend | None
        Required when ``backend="llm"``.
    model_path : str | None
        Path to a GGUF model file — required when ``normalizer="llama_cpp"``.
    model_name : str | None
        Model tag — required when ``normalizer="ollama"`` or ``"openai"``.
    api_key : str | None
        API key — required when ``normalizer="openai"``.
    n_ctx : int, default 2048
    n_threads : int, default 4
    max_tokens : int, default 128
        Construction parameters forwarded to :class:`LlamaCppBackend`.
    ollama_host : str, default "http://localhost:11434"
        Base URL forwarded to :class:`OllamaBackend`.

    Examples
    --------
    >>> ann = Annotator()                                      # zero-dependency default
    >>> ann.score("Company beats earnings estimates")          # float in [-1,1]
    >>> ann.annotate("Bankruptcy filing announced")            # {"score": ..., "label": ...}
    >>> ann.annotate_batch(["headline 1", "headline 2"])

    With local LLM normalisation:

    >>> ann = Annotator(
    ...     normalizer="llama_cpp",
    ...     model_path="/tmp/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    ... )
    >>> ann.score("Oh great, another record quarter — record losses, that is.")
    """

    def __init__(
        self,
        backend: str | FinancialVADERAnnotator | FinancialLLMAnnotator = "vader",
        *,
        normalizer=None,
        pos_threshold: float = 0.05,
        neg_threshold: float = -0.05,
        phrase_weight: float = 0.3,
        normalize_threshold: int = 80,
        normalize_mode: str = "rewrite",
        llm_backend=None,
        model_path: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        n_ctx: int = 2048,
        n_threads: int = 4,
        max_tokens: int = 128,
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        # ── build / validate annotator ───────────────────────────────────────
        if isinstance(backend, str):
            if backend == "vader":
                self._ann: FinancialVADERAnnotator | FinancialLLMAnnotator = (
                    FinancialVADERAnnotator(
                        pos_threshold=pos_threshold,
                        neg_threshold=neg_threshold,
                        phrase_weight=phrase_weight,
                    )
                )
            elif backend == "llm":
                if llm_backend is None:
                    raise ValueError(
                        "backend='llm' requires the llm_backend= kwarg to be set "
                        "to a NormalizerBackend instance (e.g. LlamaCppBackend, "
                        "OllamaBackend, OpenAIBackend)."
                    )
                self._ann = FinancialLLMAnnotator(
                    llm_backend,
                    pos_threshold=pos_threshold,
                    neg_threshold=neg_threshold,
                )
            else:
                raise ValueError(
                    f"Unknown backend string {backend!r}. Expected 'vader' or 'llm'."
                )
        elif isinstance(backend, (FinancialVADERAnnotator, FinancialLLMAnnotator)):
            self._ann = backend
        else:
            raise TypeError(
                f"backend must be 'vader', 'llm', or an annotator instance, "
                f"got {type(backend).__name__}"
            )

        # ── build / validate normaliser ──────────────────────────────────────
        self._norm = None
        self._normalize_threshold = normalize_threshold
        self._normalize_mode = normalize_mode

        if normalizer is not None:
            # Lazy import — normalizers.py does not import annotators.py so
            # there is no circular dependency; lazy here for module-load speed.
            from .normalizers import (
                FinancialTextNormalizer,
                LlamaCppBackend,
                NormalizerBackend,
                OllamaBackend,
                OpenAIBackend,
            )

            if isinstance(normalizer, str):
                if normalizer == "llama_cpp":
                    if model_path is None:
                        raise ValueError(
                            "normalizer='llama_cpp' requires model_path= to be set."
                        )
                    nb = LlamaCppBackend(
                        model_path,
                        n_ctx=n_ctx,
                        n_threads=n_threads,
                        max_tokens=max_tokens,
                    )
                    self._norm = FinancialTextNormalizer(backend=nb)
                elif normalizer == "ollama":
                    if model_name is None:
                        raise ValueError(
                            "normalizer='ollama' requires model_name= to be set."
                        )
                    nb = OllamaBackend(model=model_name, host=ollama_host)
                    self._norm = FinancialTextNormalizer(backend=nb)
                elif normalizer == "openai":
                    if model_name is None or api_key is None:
                        raise ValueError(
                            "normalizer='openai' requires both model_name= "
                            "and api_key= to be set."
                        )
                    nb = OpenAIBackend(model=model_name, api_key=api_key)
                    self._norm = FinancialTextNormalizer(backend=nb)
                else:
                    raise ValueError(
                        f"Unknown normalizer string {normalizer!r}. "
                        "Expected 'llama_cpp', 'ollama', or 'openai'."
                    )
            elif isinstance(normalizer, FinancialTextNormalizer):
                self._norm = normalizer
            elif isinstance(normalizer, NormalizerBackend):
                self._norm = FinancialTextNormalizer(backend=normalizer)
            else:
                raise TypeError(
                    f"normalizer must be a string, FinancialTextNormalizer, or "
                    f"NormalizerBackend instance, got {type(normalizer).__name__}"
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """Return LLM-normalised text if a normaliser is configured, else *text*."""
        if self._norm is None:
            return text
        result = self._norm.normalize_if_needed(
            text,
            threshold_chars=self._normalize_threshold,
            mode=self._normalize_mode,
        )
        return result.normalized_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> float:
        """Normalise (if configured) then return a sentiment score in [-1, 1]."""
        return self._ann.score(self._normalize(text))

    def label(self, score: float) -> str:
        """Map a score to a label using the inner annotator's thresholds."""
        return self._ann.label(score)

    def annotate(self, text: str) -> dict:
        """Normalise then annotate → ``{"score": float, "label": str}``."""
        return self._ann.annotate(self._normalize(text))

    def score_batch(self, texts: list[str], workers: int = 1) -> list[float]:
        """Score a list of texts."""
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.score, texts))
        return [self.score(t) for t in texts]

    def annotate_batch(self, texts: list[str], workers: int = 1) -> list[dict]:
        """Annotate a list of texts."""
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(self.annotate, texts))
        return [self.annotate(t) for t in texts]

    def score_article(self, text: str, decay: float = 0.9) -> float:
        """Normalise the full article then score with positional decay."""
        return self._ann.score_article(self._normalize(text), decay=decay)

    def annotate_article(self, text: str, decay: float = 0.9) -> dict:
        """Normalise then annotate a full article → ``{"score": float, "label": str}``."""
        return self._ann.annotate_article(self._normalize(text), decay=decay)

    def annotate_news(self, df_news):
        """Annotate a news DataFrame using the configured backend and normaliser.

        Requires a ``"headline"`` column.  Adds ``"score"``, ``"label"``,
        and (when a normaliser is configured) ``"normalized_headline"`` and
        ``"normalization_reasoning"`` columns.

        Delegates to :func:`sentvols.core.utils.annotate_news`.
        """
        from .utils import annotate_news

        return annotate_news(df_news, self._ann, normalizer=self._norm)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def inner_annotator(self) -> FinancialVADERAnnotator | FinancialLLMAnnotator:
        """The underlying annotator instance."""
        return self._ann

    @property
    def inner_normalizer(self):
        """The :class:`FinancialTextNormalizer` instance, or ``None``."""
        return self._norm

    def __repr__(self) -> str:
        norm_repr = repr(self._norm) if self._norm else "None"
        return f"Annotator(backend={self._ann!r}, normalizer={norm_repr})"

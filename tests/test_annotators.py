"""Tests for sentvols.core.annotators (public: sentvols.utils.FinancialVADERAnnotator)."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from sentvols.utils import FinancialVADERAnnotator


@pytest.fixture(scope="module")
def annotator() -> FinancialVADERAnnotator:
    return FinancialVADERAnnotator()


class TestFinancialVADERAnnotatorScore:
    def test_score_returns_float(self, annotator):
        result = annotator.score("Company reports record earnings")
        assert isinstance(result, float)

    def test_score_bounded(self, annotator):
        for text in [
            "Record profits and massive dividend increase, stock buyback announced",
            "Chapter 11 bankruptcy, accounting fraud, SEC investigation, class action lawsuit",
            "The company released its quarterly report",
        ]:
            s = annotator.score(text)
            assert -1.0 <= s <= 1.0, f"Score {s} out of [-1, 1] for '{text}'"

    def test_empty_string_returns_zero(self, annotator):
        assert annotator.score("") == 0.0

    def test_known_positive_headlines(self, annotator):
        positives = [
            "Company reports record earnings, beats estimates by 15%",
            "Strong revenue growth driven by new product launches",
            "Management raises guidance, dividend hike announced",
        ]
        for text in positives:
            assert annotator.score(text) > 0.0, f"Expected positive score for: '{text}'"

    def test_known_negative_headlines(self, annotator):
        negatives = [
            "Firm files for Chapter 11 bankruptcy amid fraud investigation",
            "Company issues profit warning, guidance cut for the quarter",
            "SEC investigation reveals accounting fraud, shares plunge 40%",
        ]
        for text in negatives:
            assert annotator.score(text) < 0.0, f"Expected negative score for: '{text}'"

    def test_lm_positive_words_lift_score(self, annotator):
        """LM-patched positive words should score higher than neutral text."""
        base = annotator.score("The company released a statement.")
        boosted = annotator.score(
            "The company reported record profit and revenue growth."
        )
        assert boosted > base

    def test_lm_negative_words_lower_score(self, annotator):
        """LM-patched negative words should score lower than neutral text."""
        base = annotator.score("The company released a statement.")
        lowered = annotator.score(
            "The company disclosed rising liabilities and mounting debt losses."
        )
        assert lowered < base


class TestFinancialVADERAnnotatorLabel:
    def test_positive_label(self, annotator):
        assert annotator.label(0.5) == "positif"

    def test_negative_label(self, annotator):
        assert annotator.label(-0.5) == "négatif"

    def test_neutral_label_zero(self, annotator):
        assert annotator.label(0.0) == "neutre"

    def test_neutral_label_borderline_positive(self, annotator):
        assert annotator.label(0.04) == "neutre"

    def test_neutral_label_borderline_negative(self, annotator):
        assert annotator.label(-0.04) == "neutre"

    def test_threshold_boundaries(self, annotator):
        assert annotator.label(0.05) == "positif"
        assert annotator.label(-0.05) == "négatif"


class TestFinancialVADERAnnotatorAnnotate:
    def test_returns_dict_with_expected_keys(self, annotator):
        result = annotator.annotate("Strong earnings beat")
        assert set(result.keys()) == {"score", "label"}

    def test_score_and_label_consistent(self, annotator):
        result = annotator.annotate("Record revenue and profit growth")
        assert result["score"] == annotator.score("Record revenue and profit growth")
        assert result["label"] == annotator.label(result["score"])

    def test_positive_annotate_end_to_end(self, annotator):
        result = annotator.annotate("Company beats earnings estimates, raises dividend")
        assert result["label"] == "positif"
        assert result["score"] > 0

    def test_negative_annotate_end_to_end(self, annotator):
        result = annotator.annotate("Bankruptcy filing, fraud investigation ongoing")
        assert result["label"] == "négatif"
        assert result["score"] < 0


# ---------------------------------------------------------------------------
# Custom lexicon / phrase extension API
# ---------------------------------------------------------------------------


class TestCustomLexicon:
    def test_constructor_custom_lexicon_overrides(self):
        """A word supplied via custom_lexicon must override the built-in value."""
        base = FinancialVADERAnnotator()
        custom = FinancialVADERAnnotator(custom_lexicon={"moonshot": 3.5})
        assert custom.score("The moonshot plan") > base.score("The moonshot plan")

    def test_constructor_custom_lexicon_case_insensitive(self):
        """Keys are normalised to lower-case."""
        ann = FinancialVADERAnnotator(custom_lexicon={"DELIST": -3.5})
        assert ann.score("Company faces delist warning") < -0.05

    def test_constructor_custom_phrases(self):
        """A phrase supplied via custom_phrases must affect the score."""
        base = FinancialVADERAnnotator()
        custom = FinancialVADERAnnotator(custom_phrases={"reverse split": -2.5})
        text = "Board approves a reverse split next quarter"
        assert custom.score(text) < base.score(text)

    def test_constructor_does_not_mutate_shared_lexicon(self):
        """Instantiation with custom_lexicon must not alter _LM_LEXICON."""
        from sentvols.core.annotators import _LM_LEXICON

        word = "canaryword_xyz"
        before = word in _LM_LEXICON
        FinancialVADERAnnotator(custom_lexicon={word: 3.0})
        assert (word in _LM_LEXICON) == before

    def test_add_words_increases_score(self):
        """add_words() applied on a neutral word lifts the score."""
        ann = FinancialVADERAnnotator()
        text = "The spinoff was announced today"
        score_before = ann.score(text)
        ann.add_words({"spinoff": 2.5})
        assert ann.score(text) > score_before

    def test_add_words_case_insensitive(self):
        ann = FinancialVADERAnnotator()
        ann.add_words({"TURNAROUND": 3.0})
        assert ann.score("Great turnaround story") > 0.05

    def test_add_phrases_affects_score(self):
        """add_phrases() must register the new phrase and alter the score."""
        ann = FinancialVADERAnnotator()
        text = "Management announces secondary offering next week"
        score_before = ann.score(text)
        ann.add_phrases({"secondary offering": -2.0})
        assert ann.score(text) < score_before

    def test_add_phrases_case_insensitive(self):
        ann = FinancialVADERAnnotator()
        ann.add_phrases({"SHARE SPLIT": 2.0})
        text = "Company announces a share split"
        ann2 = FinancialVADERAnnotator()
        assert ann.score(text) > ann2.score(text)

    def test_instances_are_independent(self):
        """Modifying one instance must not affect another."""
        ann1 = FinancialVADERAnnotator()
        ann2 = FinancialVADERAnnotator()
        ann1.add_words({"privatekeyword": 3.5})
        assert ann2._analyzer.lexicon.get("privatekeyword") is None


# ---------------------------------------------------------------------------
# Settable thresholds & phrase_weight
# ---------------------------------------------------------------------------


class TestThresholdsAndPhraseWeight:
    def test_constructor_custom_thresholds(self):
        ann = FinancialVADERAnnotator(pos_threshold=0.2, neg_threshold=-0.2)
        assert ann.label(0.1) == "neutre"
        assert ann.label(0.25) == "positif"
        assert ann.label(-0.25) == "négatif"

    def test_property_setter_pos_threshold(self):
        ann = FinancialVADERAnnotator()
        ann.pos_threshold = 0.15
        assert ann.pos_threshold == 0.15
        assert ann.label(0.1) == "neutre"
        assert ann.label(0.2) == "positif"

    def test_property_setter_neg_threshold(self):
        ann = FinancialVADERAnnotator()
        ann.neg_threshold = -0.15
        assert ann.label(-0.1) == "neutre"
        assert ann.label(-0.2) == "négatif"

    def test_invalid_pos_threshold_raises(self):
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(pos_threshold=0.0)
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(pos_threshold=1.0)

    def test_invalid_neg_threshold_raises(self):
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(neg_threshold=0.0)
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(neg_threshold=-1.0)

    def test_setter_rejects_neg_above_zero(self):
        """neg_threshold setter must reject values outside (-1, 0)."""
        ann = FinancialVADERAnnotator()
        with pytest.raises(ValueError):
            ann.neg_threshold = 0.1  # positive value is outside valid range

    def test_setter_rejects_pos_above_one(self):
        """pos_threshold setter must reject values outside (0, 1)."""
        ann = FinancialVADERAnnotator()
        with pytest.raises(ValueError):
            ann.pos_threshold = 1.5

    def test_invalid_phrase_weight_raises(self):
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(phrase_weight=0.0)
        with pytest.raises(ValueError):
            FinancialVADERAnnotator(phrase_weight=1.5)

    def test_phrase_weight_affects_score(self):
        light = FinancialVADERAnnotator(phrase_weight=0.1)
        heavy = FinancialVADERAnnotator(phrase_weight=1.0)
        text = "earnings beat"
        assert abs(heavy.score(text)) >= abs(light.score(text))


# ---------------------------------------------------------------------------
# Batch methods
# ---------------------------------------------------------------------------


class TestBatchMethods:
    def test_score_batch_matches_single(self):
        ann = FinancialVADERAnnotator()
        texts = [
            "Record earnings beat, dividend hike announced",
            "Chapter 11 bankruptcy, SEC fraud investigation",
            "The company released quarterly results",
        ]
        assert ann.score_batch(texts) == [ann.score(t) for t in texts]

    def test_annotate_batch_matches_single(self):
        ann = FinancialVADERAnnotator()
        texts = ["Strong profit growth", "Going concern warning"]
        assert ann.annotate_batch(texts) == [ann.annotate(t) for t in texts]

    def test_score_batch_empty_list(self):
        ann = FinancialVADERAnnotator()
        assert ann.score_batch([]) == []

    def test_annotate_batch_preserves_order(self):
        ann = FinancialVADERAnnotator()
        texts = [f"headline {i}" for i in range(20)]
        batch = ann.annotate_batch(texts)
        assert len(batch) == 20
        for i, result in enumerate(batch):
            assert result == ann.annotate(texts[i])


# ---------------------------------------------------------------------------
# remove_words & lexicon_snapshot
# ---------------------------------------------------------------------------


class TestRemoveWordsAndLexiconSnapshot:
    def test_remove_words_lowers_score(self):
        ann = FinancialVADERAnnotator()
        text = "The spinoff was announced"
        ann.add_words({"spinoff": 3.0})
        score_before = ann.score(text)
        ann.remove_words(["spinoff"])
        assert ann.score(text) < score_before

    def test_remove_words_silent_on_missing(self):
        ann = FinancialVADERAnnotator()
        ann.remove_words(["nonexistentword_xyz"])  # must not raise

    def test_lexicon_snapshot_contains_custom_word(self):
        ann = FinancialVADERAnnotator(custom_lexicon={"moonshot": 3.5})
        snap = ann.lexicon_snapshot
        assert snap["moonshot"] == 3.5

    def test_lexicon_snapshot_is_copy(self):
        ann = FinancialVADERAnnotator()
        snap = ann.lexicon_snapshot
        snap["__test__"] = 99.0
        assert ann._analyzer.lexicon.get("__test__") is None


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_contains_thresholds_and_counts(self):
        ann = FinancialVADERAnnotator(pos_threshold=0.1, neg_threshold=-0.1)
        r = repr(ann)
        assert "pos_threshold=0.1" in r
        assert "neg_threshold=-0.1" in r
        assert "n_lexicon_entries=" in r
        assert "n_phrases=" in r


# ---------------------------------------------------------------------------
# explain() / explain_batch() / explain_to_dataframe()
# ---------------------------------------------------------------------------


class TestExplain:
    def test_explain_returns_expected_keys(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("Company beats earnings estimates")
        assert set(exp.keys()) == {
            "text",
            "base_vader_score",
            "word_hits",
            "phrase_hits",
            "phrase_adjustment",
            "final_score",
            "label",
        }

    def test_explain_empty_string(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("")
        assert exp["final_score"] == 0.0
        assert exp["word_hits"] == []
        assert exp["phrase_hits"] == []

    def test_explain_word_hits_have_correct_keys(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("The company reported a loss and bankruptcy")
        for wh in exp["word_hits"]:
            assert set(wh.keys()) == {"word", "valence"}

    def test_explain_phrase_hits_detected(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("The firm issued a profit warning")
        phrases = [ph["phrase"] for ph in exp["phrase_hits"]]
        assert "profit warning" in phrases

    def test_explain_final_score_matches_score(self):
        ann = FinancialVADERAnnotator()
        text = "Record earnings beat with dividend hike"
        assert abs(ann.explain(text)["final_score"] - ann.score(text)) < 1e-9

    def test_explain_word_hits_sorted_by_abs_valence(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("Company faces bankruptcy losses and fraud investigation")
        valences = [abs(wh["valence"]) for wh in exp["word_hits"]]
        assert valences == sorted(valences, reverse=True)

    def test_explain_positive_text_has_positive_word_hits(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("Record profit and strong revenue growth")
        positive_hits = [wh for wh in exp["word_hits"] if wh["valence"] > 0]
        assert len(positive_hits) > 0

    def test_explain_label_consistent_with_final_score(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("Bankruptcy amid accounting fraud SEC investigation")
        assert exp["label"] == ann.label(exp["final_score"])

    def test_explain_to_dataframe_columns(self):
        import pandas as pd

        ann = FinancialVADERAnnotator()
        df = ann.explain_to_dataframe("Chapter 11 bankruptcy amid fraud investigation")
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"source", "token", "valence"}

    def test_explain_to_dataframe_source_values(self):
        ann = FinancialVADERAnnotator()
        df = ann.explain_to_dataframe(
            "earnings beat with profit growth record dividends"
        )
        assert set(df["source"].unique()).issubset({"word", "phrase"})

    def test_explain_to_dataframe_sorted_by_abs_valence(self):
        import pandas as pd

        ann = FinancialVADERAnnotator()
        df = ann.explain_to_dataframe(
            "Strong earnings beat, chapter 11 bankruptcy fraud"
        )
        valences = df["valence"].abs().tolist()
        assert valences == sorted(valences, reverse=True)

    def test_explain_to_dataframe_empty_text_returns_empty_df(self):
        import pandas as pd

        ann = FinancialVADERAnnotator()
        df = ann.explain_to_dataframe("")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_explain_batch_length(self):
        ann = FinancialVADERAnnotator()
        texts = ["headline one", "headline two", "headline three"]
        results = ann.explain_batch(texts)
        assert len(results) == 3
        for r in results:
            assert "final_score" in r

    def test_explain_batch_matches_single(self):
        ann = FinancialVADERAnnotator()
        texts = [
            "Record earnings beat, dividend hike",
            "Chapter 11 bankruptcy fraud",
            "Quarterly results released",
        ]
        batch = ann.explain_batch(texts)
        for exp, text in zip(batch, texts):
            assert abs(exp["final_score"] - ann.score(text)) < 1e-9


# ---------------------------------------------------------------------------
# Batch workers parameter
# ---------------------------------------------------------------------------


class TestBatchWorkers:
    def test_score_batch_workers_matches_single(self):
        ann = FinancialVADERAnnotator()
        texts = ["Record earnings", "Bankruptcy filing", "Market update"] * 5
        assert ann.score_batch(texts, workers=2) == ann.score_batch(texts, workers=1)

    def test_annotate_batch_workers_matches_single(self):
        ann = FinancialVADERAnnotator()
        texts = ["Strong growth", "SEC investigation", "Quarterly results"] * 5
        assert ann.annotate_batch(texts, workers=2) == ann.annotate_batch(
            texts, workers=1
        )

    def test_score_batch_workers_preserves_order(self):
        ann = FinancialVADERAnnotator()
        texts = [f"Company report {i}" for i in range(30)]
        single = ann.score_batch(texts, workers=1)
        multi = ann.score_batch(texts, workers=4)
        assert single == multi

    def test_annotate_batch_workers_empty_list(self):
        ann = FinancialVADERAnnotator()
        assert ann.annotate_batch([], workers=2) == []


# ---------------------------------------------------------------------------
# Negation-aware phrase matching
# ---------------------------------------------------------------------------


class TestNegationAwarePhrases:
    def test_negated_phrase_scores_differently(self):
        ann = FinancialVADERAnnotator()
        plain = ann.score("The firm issued a profit warning")
        negated = ann.score("The firm did not issue a profit warning")
        assert negated > plain  # negation suppresses the negative phrase hit

    def test_negated_earnings_beat_scores_lower(self):
        ann = FinancialVADERAnnotator()
        plain = ann.score("Company reported an earnings beat")
        negated = ann.score("Company did not report an earnings beat")
        assert negated < plain

    def test_no_before_phrase_suppresses_it(self):
        ann = FinancialVADERAnnotator()
        plain = ann.score("There was a profit warning from the firm")
        negated = ann.score("There was no profit warning from the firm")
        assert negated > plain

    def test_non_negated_phrase_still_fires(self):
        ann = FinancialVADERAnnotator()
        text = "The company filed for chapter 11 protection"
        s = ann.score(text)
        assert s < -0.05  # phrase fires, score is negative

    def test_is_phrase_negated_helper_direct(self):
        from sentvols.core.annotators import _is_phrase_negated

        assert (
            _is_phrase_negated("profit warning", "did not issue a profit warning")
            is True
        )
        assert _is_phrase_negated("profit warning", "issued a profit warning") is False
        assert (
            _is_phrase_negated("earnings beat", "no earnings beat this quarter") is True
        )
        assert _is_phrase_negated("earnings beat", "reported an earnings beat") is False

    def test_explain_negated_phrase_is_flagged(self):
        ann = FinancialVADERAnnotator()
        exp = ann.explain("The firm did not issue a profit warning")
        phrase_names = [ph["phrase"] for ph in exp["phrase_hits"]]
        # Negated phrase IS detected and listed — but with negated=True and
        # a flipped (positive) adjustment so the final score is non-negative.
        assert "profit warning" in phrase_names
        match = next(
            ph for ph in exp["phrase_hits"] if ph["phrase"] == "profit warning"
        )
        assert match["negated"] is True
        assert match["adjustment"] > 0  # signal flipped: was -2.5, now +2.5
        assert exp["final_score"] > -0.05  # sentence is no longer scored negative


# ---------------------------------------------------------------------------
# score_article() / annotate_article()
# ---------------------------------------------------------------------------


class TestScoreArticle:
    def test_returns_float_in_range(self):
        ann = FinancialVADERAnnotator()
        article = "Company beats earnings. Revenue rose 12%. Management raised guidance. Strong results."
        s = ann.score_article(article)
        assert isinstance(s, float)
        assert -1.0 <= s <= 1.0

    def test_positive_article_positive_score(self):
        ann = FinancialVADERAnnotator()
        article = (
            "The company reported record earnings this quarter. "
            "Revenue surpassed analyst estimates. Management raised full-year guidance. "
            "Dividends were increased. The board approved a share buyback program."
        )
        assert ann.score_article(article) > 0.05

    def test_negative_article_negative_score(self):
        ann = FinancialVADERAnnotator()
        article = (
            "The company filed for Chapter 11 bankruptcy protection. "
            "The SEC opened a fraud investigation into accounting practices. "
            "Three executives resigned amid the scandal. "
            "Credit agencies issued a downgrade to junk status."
        )
        assert ann.score_article(article) < -0.05

    def test_decay_one_equals_simple_mean(self):
        import re

        ann = FinancialVADERAnnotator()
        article = (
            "The company beat earnings estimates. "
            "However, the firm acknowledged ongoing risks."
        )
        sents = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", article)
            if len(s.strip()) >= 10
        ]
        expected_mean = sum(ann.score(s) for s in sents) / len(sents)
        assert abs(ann.score_article(article, decay=1.0) - expected_mean) < 1e-9

    def test_short_text_delegates_to_score(self):
        ann = FinancialVADERAnnotator()
        headline = "Company beats earnings estimates"
        assert ann.score_article(headline) == ann.score(headline)

    def test_annotate_article_keys(self):
        ann = FinancialVADERAnnotator()
        result = ann.annotate_article(
            "Record earnings. Strong guidance. Dividend hike announced."
        )
        assert set(result.keys()) == {"score", "label"}
        assert isinstance(result["score"], float)
        assert result["label"] in {"positif", "négatif", "neutre"}


# ---------------------------------------------------------------------------
# FinancialLLMAnnotator
# ---------------------------------------------------------------------------


class TestFinancialLLMAnnotator:
    """Tests for FinancialLLMAnnotator — all LLM calls are mocked."""

    @staticmethod
    def _make_backend(reply: str = "0.7"):
        """Return a concrete stub class satisfying NormalizerBackend.

        Python 3.12+ runtime_checkable Protocol checks class-level attributes;
        MagicMock's __getattr__ is no longer sufficient.  We create a fresh
        class per call so each instance gets its own call tracker.
        """
        call_mock = MagicMock(return_value=(reply, None))

        class _StubBackend:
            model = "mock-model"
            reasoning_available = False
            call = call_mock  # class attr — still callable & trackable

        b = _StubBackend()
        b.call = call_mock  # shadow at instance level (both refer to same mock)
        return b

    def test_score_returns_float_in_range(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.7"))
        s = ann.score("Company beats earnings")
        assert isinstance(s, float)
        assert -1.0 <= s <= 1.0

    def test_score_positive_value(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.8"))
        assert ann.score("text") == pytest.approx(0.8)

    def test_score_negative_value(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("-0.6"))
        assert ann.score("text") == pytest.approx(-0.6)

    def test_score_parses_noisy_output(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("Score: 0.45."))
        assert ann.score("text") == pytest.approx(0.45)

    def test_score_keyword_fallback_positive(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("positive sentiment"))
        assert ann.score("text") == pytest.approx(0.7)

    def test_score_keyword_fallback_negative(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("negative outlook"))
        assert ann.score("text") == pytest.approx(-0.7)

    def test_score_keyword_fallback_neutral(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("neutral mixed signals"))
        assert ann.score("text") == pytest.approx(0.0)

    def test_score_clamped_high(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("2.5"))
        assert ann.score("text") == pytest.approx(1.0)

    def test_score_clamped_low(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("-3.0"))
        assert ann.score("text") == pytest.approx(-1.0)

    def test_empty_text_returns_zero(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.9"))
        assert ann.score("") == 0.0
        ann._backend.call.assert_not_called()

    def test_annotate_returns_expected_keys(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.7"))
        result = ann.annotate("Strong earnings")
        assert set(result.keys()) == {"score", "label"}

    def test_annotate_positive_label(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.7"))
        assert ann.annotate("Strong earnings")["label"] == "positif"

    def test_annotate_negative_label(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("-0.7"))
        assert ann.annotate("Bankruptcy filing")["label"] == "négatif"

    def test_annotate_neutral_label(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.0"))
        assert ann.annotate("Quarterly report released")["label"] == "neutre"

    def test_score_batch_length(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.3"))
        results = ann.score_batch(["t1", "t2", "t3"])
        assert len(results) == 3

    def test_annotate_batch_length(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.3"))
        results = ann.annotate_batch(["t1", "t2"])
        assert len(results) == 2
        for r in results:
            assert "score" in r and "label" in r

    def test_score_batch_uses_batch_call_when_available(self):
        """score_batch() must delegate to backend.batch_call() in a single call."""
        from sentvols.core.annotators import FinancialLLMAnnotator

        texts = ["Earnings beat", "Dividend cut", "Neutral report"]
        replies = [("0.8", None), ("-0.6", None), ("0.0", None)]

        batch_mock = MagicMock(return_value=replies)

        class _BatchBackend:
            model = "batch-model"
            reasoning_available = False
            call = MagicMock(return_value=("0.0", None))  # should NOT be called
            batch_call = batch_mock

        ann = FinancialLLMAnnotator(_BatchBackend())
        scores = ann.score_batch(texts)

        batch_mock.assert_called_once()  # exactly one batched call
        _BatchBackend.call.assert_not_called()  # per-text call must be bypassed
        assert len(scores) == 3
        assert scores[0] == pytest.approx(0.8)
        assert scores[1] == pytest.approx(-0.6)
        assert scores[2] == pytest.approx(0.0)

    def test_annotate_batch_uses_batch_call_when_available(self):
        """annotate_batch() returns correct labels when batch_call is used."""
        from sentvols.core.annotators import FinancialLLMAnnotator

        replies = [("0.7", None), ("-0.7", None)]
        batch_mock = MagicMock(return_value=replies)

        class _BatchBackend:
            model = "batch-model"
            reasoning_available = False
            call = MagicMock(return_value=("0.0", None))
            batch_call = batch_mock

        ann = FinancialLLMAnnotator(_BatchBackend())
        results = ann.annotate_batch(["Strong buy", "Bankruptcy"])

        assert results[0]["label"] == "positif"
        assert results[1]["label"] == "négatif"
        batch_mock.assert_called_once()

    def test_score_batch_empty_with_batch_call(self):
        """score_batch([]) returns [] immediately without calling batch_call."""
        from sentvols.core.annotators import FinancialLLMAnnotator

        batch_mock = MagicMock(return_value=[])

        class _BatchBackend:
            model = "batch-model"
            reasoning_available = False
            call = MagicMock(return_value=("0.0", None))
            batch_call = batch_mock

        ann = FinancialLLMAnnotator(_BatchBackend())
        assert ann.score_batch([]) == []
        batch_mock.assert_not_called()

    def test_score_batch_fallback_loop_when_no_batch_call(self):
        """Backends without batch_call still work via per-item call() loop."""
        from sentvols.core.annotators import FinancialLLMAnnotator

        texts = ["text1", "text2"]
        b = self._make_backend("0.5")
        ann = FinancialLLMAnnotator(b)
        scores = ann.score_batch(texts)
        assert len(scores) == 2
        assert b.call.call_count == len(texts)

    def test_score_article_single_llm_call(self):
        """LLM annotator sends the whole article in one call, not per sentence."""
        from sentvols.core.annotators import FinancialLLMAnnotator

        b = self._make_backend("0.6")
        ann = FinancialLLMAnnotator(b)
        ann.score_article("Sentence one. Sentence two. Sentence three.")
        assert b.call.call_count == 1

    def test_score_article_truncates_long_text(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        b = self._make_backend("0.0")
        ann = FinancialLLMAnnotator(b, max_article_chars=10)
        ann.score_article("A" * 100)
        called_prompt = b.call.call_args[0][0]
        # The truncated text (10 chars of 'A') must appear in the prompt
        assert "A" * 10 in called_prompt
        assert "A" * 11 not in called_prompt

    def test_annotate_article_returns_expected_keys(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.5"))
        result = ann.annotate_article("Record earnings. Revenue up. Guidance raised.")
        assert set(result.keys()) == {"score", "label"}

    def test_invalid_backend_raises_type_error(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        with pytest.raises(TypeError, match="NormalizerBackend"):
            FinancialLLMAnnotator("not-a-backend")

    def test_invalid_threshold_raises(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        with pytest.raises(ValueError):
            FinancialLLMAnnotator(self._make_backend(), pos_threshold=-0.5)

    def test_final_score_matches_score(self):
        from sentvols.core.annotators import FinancialLLMAnnotator

        ann = FinancialLLMAnnotator(self._make_backend("0.4"))
        text = "Company reports solid results"
        result = ann.annotate(text)
        assert result["score"] == ann.score(text)


# ---------------------------------------------------------------------------
# Annotator (pipeline builder)
# ---------------------------------------------------------------------------


class TestAnnotator:
    """Tests for the Annotator pipeline façade."""

    def test_default_creates_vader_annotator(self):
        from sentvols.core.annotators import Annotator, FinancialVADERAnnotator

        ann = Annotator()
        assert isinstance(ann.inner_annotator, FinancialVADERAnnotator)

    def test_string_vader_creates_annotator(self):
        from sentvols.core.annotators import Annotator, FinancialVADERAnnotator

        ann = Annotator(backend="vader")
        assert isinstance(ann.inner_annotator, FinancialVADERAnnotator)

    def test_vader_instance_accepted(self):
        from sentvols.core.annotators import Annotator, FinancialVADERAnnotator

        vader = FinancialVADERAnnotator()
        ann = Annotator(backend=vader)
        assert ann.inner_annotator is vader

    def test_llm_annotator_instance_accepted(self):
        from sentvols.core.annotators import Annotator, FinancialLLMAnnotator

        call_mock = MagicMock(return_value=("0.5", None))

        class _B:
            model = "m"
            reasoning_available = False
            call = call_mock

        llm_ann = FinancialLLMAnnotator(_B())
        ann = Annotator(backend=llm_ann)
        assert ann.inner_annotator is llm_ann

    def test_invalid_backend_string_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(ValueError, match="'vader' or 'llm'"):
            Annotator(backend="unknown")

    def test_invalid_backend_type_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(TypeError):
            Annotator(backend=42)

    def test_llm_string_without_llm_backend_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(ValueError, match="llm_backend"):
            Annotator(backend="llm")

    def test_score_returns_float(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        s = ann.score("Company beats earnings estimates")
        assert isinstance(s, float)
        assert -1.0 <= s <= 1.0

    def test_annotate_returns_dict_with_keys(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        result = ann.annotate("Company beats earnings estimates")
        assert set(result.keys()) == {"score", "label"}
        assert isinstance(result["score"], float)
        assert result["label"] in {"positif", "négatif", "neutre"}

    def test_score_batch_length(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        results = ann.score_batch(["t1", "t2", "t3"])
        assert len(results) == 3

    def test_annotate_batch_length(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        results = ann.annotate_batch(["t1", "t2"])
        assert len(results) == 2

    def test_no_normalizer_by_default(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        assert ann.inner_normalizer is None

    def test_normalizer_financial_text_normalizer_accepted(self):
        from sentvols.core.annotators import Annotator
        from sentvols.core.normalizers import FinancialTextNormalizer

        call_mock = MagicMock(return_value=("rewritten text", None))

        class _B:
            model = "m"
            reasoning_available = False
            call = call_mock

        norm = FinancialTextNormalizer(backend=_B())
        ann = Annotator(normalizer=norm)
        assert ann.inner_normalizer is norm

    def test_normalizer_backend_instance_wrapped_in_ftn(self):
        """Passing a raw NormalizerBackend auto-wraps it in FinancialTextNormalizer."""
        from sentvols.core.annotators import Annotator
        from sentvols.core.normalizers import FinancialTextNormalizer

        call_mock = MagicMock(return_value=("rewritten", None))

        class _B:
            model = "m"
            reasoning_available = False
            call = call_mock

        ann = Annotator(normalizer=_B())
        assert isinstance(ann.inner_normalizer, FinancialTextNormalizer)

    def test_normalizer_is_applied_when_long_text(self):
        """Score differs when normalizer rewrites the text."""
        from sentvols.core.annotators import Annotator
        from sentvols.core.normalizers import FinancialTextNormalizer

        call_mock = MagicMock(return_value=("The company posted record losses.", None))

        class _B:
            model = "m"
            reasoning_available = False
            call = call_mock

        ann_with = Annotator(
            normalizer=FinancialTextNormalizer(backend=_B()),
            normalize_threshold=1,  # always normalise
        )
        ann_plain = Annotator()
        ironic = "Oh great, another record quarter."
        assert ann_with.score(ironic) != ann_plain.score(ironic)

    def test_annotate_article_delegates(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        result = ann.annotate_article(
            "Earnings beat. Revenue up 14%. Guidance raised. Dividend hike."
        )
        assert set(result.keys()) == {"score", "label"}
        assert isinstance(result["score"], float)

    def test_score_article_consistent_with_annotate_article(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        text = "Record earnings. Revenue beat. Guidance raised. Dividend hike."
        assert ann.score_article(text) == ann.annotate_article(text)["score"]

    def test_invalid_normalizer_string_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(ValueError, match="llama_cpp"):
            Annotator(normalizer="unknown")

    def test_normalizer_llama_cpp_without_path_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(ValueError, match="model_path"):
            Annotator(normalizer="llama_cpp")

    def test_normalizer_openai_without_api_key_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(ValueError, match="api_key"):
            Annotator(normalizer="openai", model_name="gpt-4o-mini")

    def test_invalid_normalizer_type_raises(self):
        from sentvols.core.annotators import Annotator

        with pytest.raises(TypeError):
            Annotator(normalizer=42)

    def test_repr_contains_backend_and_normalizer(self):
        from sentvols.core.annotators import Annotator

        ann = Annotator()
        r = repr(ann)
        assert "Annotator" in r
        assert "FinancialVADERAnnotator" in r
        assert "None" in r

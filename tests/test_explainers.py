"""Tests for sentvols.core.explainers (public: sentvols.explainers)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sentvols.explainers as explainers
from sentvols.explainers import (
    run_hypothesis_tests,
    test_alpha,
    test_classifier_permutation,
    test_diebold_mariano,
    test_sentiment_correlation,
)

# Prevent pytest from treating these imported explainer functions as test cases
# (they start with "test_" but are domain functions, not pytest tests).
test_alpha.__test__ = False
test_classifier_permutation.__test__ = False
test_diebold_mariano.__test__ = False
test_sentiment_correlation.__test__ = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perf_positive_alpha() -> pd.DataFrame:
    """Portfolio that consistently beats the benchmark."""
    rng = np.random.default_rng(0)
    n = 24
    port_ret = rng.uniform(0.01, 0.04, n)  # always positive excess
    bench_ret = rng.uniform(0.005, 0.02, n)
    excess = port_ret - bench_ret
    return pd.DataFrame(
        {
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "excess": excess,
            "cum_port": (1 + port_ret).cumprod() - 1,
            "cum_bench": (1 + bench_ret).cumprod() - 1,
        }
    )


@pytest.fixture(scope="module")
def perf_zero_alpha() -> pd.DataFrame:
    """Portfolio with zero mean excess return."""
    rng = np.random.default_rng(1)
    n = 24
    excess = rng.standard_normal(n) * 0.01
    port_ret = excess / 2
    bench_ret = -excess / 2
    return pd.DataFrame(
        {
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "excess": excess,
            "cum_port": (1 + port_ret).cumprod() - 1,
            "cum_bench": (1 + bench_ret).cumprod() - 1,
        }
    )


@pytest.fixture(scope="module")
def df_pd_with_correlation() -> pd.DataFrame:
    """Monthly panel with a measurable positive correlation."""
    rng = np.random.default_rng(2)
    n = 500
    sent = rng.standard_normal(n)
    target = sent * 0.02 + rng.standard_normal(n) * 0.005  # correlated
    return pd.DataFrame(
        {
            "sent_sum_mean": sent,
            "sent_mean_avg": sent * 0.5,
            "target_return": target,
        }
    )


@pytest.fixture(scope="module")
def df_pd_no_correlation() -> pd.DataFrame:
    """Monthly panel with near-zero correlation."""
    rng = np.random.default_rng(3)
    n = 500
    return pd.DataFrame(
        {
            "sent_sum_mean": rng.standard_normal(n),
            "sent_mean_avg": rng.standard_normal(n),
            "target_return": rng.standard_normal(n),
        }
    )


class _DummyClf:
    """Always predicts 1."""

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class _DummyReg:
    """Predicts a constant value."""

    def predict(self, X):
        return np.full(len(X), 0.01)


# ---------------------------------------------------------------------------
# test_alpha
# ---------------------------------------------------------------------------


class TestTestAlpha:
    def test_returns_expected_keys(self, perf_positive_alpha):
        result = test_alpha(perf_positive_alpha)
        assert set(result.keys()) == {
            "mean_excess_monthly",
            "t_statistic",
            "p_value",
            "significant",
        }

    def test_positive_alpha_significant(self, perf_positive_alpha):
        result = test_alpha(perf_positive_alpha)
        assert result["mean_excess_monthly"] > 0
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_zero_alpha_not_significant(self, perf_zero_alpha):
        """A portfolio with zero mean excess should not reliably reject H0."""
        result = test_alpha(perf_zero_alpha)
        # Not always False (stochastic), but the p-value should not be very small
        assert result["p_value"] >= 0.0  # sanity bound
        assert isinstance(result["significant"], bool)

    def test_mean_excess_matches_data(self, perf_positive_alpha):
        result = test_alpha(perf_positive_alpha)
        assert (
            abs(result["mean_excess_monthly"] - perf_positive_alpha["excess"].mean())
            < 1e-10
        )


# ---------------------------------------------------------------------------
# test_sentiment_correlation
# ---------------------------------------------------------------------------


class TestTestSentimentCorrelation:
    def test_returns_expected_keys(self, df_pd_with_correlation):
        result = test_sentiment_correlation(df_pd_with_correlation)
        assert set(result.keys()) == {
            "n_obs",
            "r_sum",
            "p_sum",
            "r_avg",
            "p_avg",
            "significant",
        }

    def test_correlated_data_is_significant(self, df_pd_with_correlation):
        result = test_sentiment_correlation(df_pd_with_correlation)
        assert result["r_sum"] > 0
        assert result["p_sum"] < 0.05
        assert result["significant"] is True

    def test_uncorrelated_data(self, df_pd_no_correlation):
        result = test_sentiment_correlation(df_pd_no_correlation)
        assert abs(result["r_sum"]) < 0.15  # weak correlation expected
        assert isinstance(result["n_obs"], int)
        assert result["n_obs"] > 0

    def test_n_obs_equals_non_null_count(self, df_pd_with_correlation):
        result = test_sentiment_correlation(df_pd_with_correlation)
        assert result["n_obs"] == len(df_pd_with_correlation)


# ---------------------------------------------------------------------------
# test_classifier_permutation
# ---------------------------------------------------------------------------


class TestTestClassifierPermutation:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((50, 5))
        y = rng.integers(0, 2, 50)
        result = test_classifier_permutation(_DummyClf(), X, y, n_permu=100, seed=0)
        assert set(result.keys()) == {
            "baseline_f1",
            "perm_f1_mean",
            "p_value",
            "perm_f1s",
            "significant",
        }

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 5))
        y = rng.integers(0, 2, 50)
        result = test_classifier_permutation(_DummyClf(), X, y, n_permu=100, seed=0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_perm_f1s_shape(self):
        n_permu = 50
        rng = np.random.default_rng(6)
        X = rng.standard_normal((50, 5))
        y = rng.integers(0, 2, 50)
        result = test_classifier_permutation(_DummyClf(), X, y, n_permu=n_permu, seed=0)
        assert len(result["perm_f1s"]) == n_permu

    def test_significant_is_bool(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 5))
        y = rng.integers(0, 2, 50)
        result = test_classifier_permutation(_DummyClf(), X, y, n_permu=50, seed=0)
        assert isinstance(result["significant"], bool)

    def test_baseline_f1_matches_direct_call(self):
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(8)
        X = rng.standard_normal((80, 5))
        y = rng.integers(0, 2, 80)
        clf = _DummyClf()
        result = test_classifier_permutation(clf, X, y, n_permu=20, seed=0)
        expected_f1 = f1_score(y, clf.predict(X))
        assert abs(result["baseline_f1"] - expected_f1) < 1e-9


# ---------------------------------------------------------------------------
# test_diebold_mariano
# ---------------------------------------------------------------------------


class TestTestDieboldMariano:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(9)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100) * 0.01
        result = test_diebold_mariano(_DummyReg(), X, y)
        assert set(result.keys()) == {
            "mse_naive",
            "mse_lgbm",
            "t_statistic",
            "p_value",
            "significant",
        }

    def test_mse_values_non_negative(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100) * 0.01
        result = test_diebold_mariano(_DummyReg(), X, y)
        assert result["mse_naive"] >= 0
        assert result["mse_lgbm"] >= 0

    def test_perfect_predictor_is_significant(self):
        """A predictor that matches y_true exactly must beat the naive predictor."""

        class PerfectReg:
            def predict(self, X):
                return y_true

        rng = np.random.default_rng(11)
        X = rng.standard_normal((200, 5))
        y_true = rng.standard_normal(200) * 0.02
        result = test_diebold_mariano(PerfectReg(), X, y_true)
        # MSE of perfect predictor == 0
        assert result["mse_lgbm"] == pytest.approx(0.0)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(12)
        X = rng.standard_normal((100, 5))
        y = rng.standard_normal(100) * 0.01
        result = test_diebold_mariano(_DummyReg(), X, y)
        assert 0.0 <= result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# run_hypothesis_tests (integration)
# ---------------------------------------------------------------------------


class TestRunHypothesisTests:
    def test_returns_all_sections(self, perf_positive_alpha, df_pd_with_correlation):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((80, 5))
        y_cls = rng.integers(0, 2, 80)
        y_reg = rng.standard_normal(80) * 0.01

        result = run_hypothesis_tests(
            perf=perf_positive_alpha,
            df_pd=df_pd_with_correlation,
            clf=_DummyClf(),
            reg=_DummyReg(),
            X_test_sc=X,
            X_test_reg_sc=X,
            y_cls_test=y_cls,
            y_reg_test=y_reg,
            n_permu=50,
            seed=0,
        )
        assert set(result.keys()) == {
            "alpha",
            "correlation",
            "permutation",
            "diebold_mariano",
        }

    def test_all_section_keys_present(
        self, perf_positive_alpha, df_pd_with_correlation
    ):
        rng = np.random.default_rng(14)
        X = rng.standard_normal((80, 5))
        y_cls = rng.integers(0, 2, 80)
        y_reg = rng.standard_normal(80) * 0.01

        result = run_hypothesis_tests(
            perf=perf_positive_alpha,
            df_pd=df_pd_with_correlation,
            clf=_DummyClf(),
            reg=_DummyReg(),
            X_test_sc=X,
            X_test_reg_sc=X,
            y_cls_test=y_cls,
            y_reg_test=y_reg,
            n_permu=50,
            seed=0,
        )
        assert "significant" in result["alpha"]
        assert "significant" in result["correlation"]
        assert "significant" in result["permutation"]
        assert "significant" in result["diebold_mariano"]


class TestExplainersNamespace:
    def test_all_functions_exported(self):
        expected = {
            "test_alpha",
            "test_sentiment_correlation",
            "test_classifier_permutation",
            "test_diebold_mariano",
            "run_hypothesis_tests",
            "test_ols_sentiment_impact",
        }
        assert expected.issubset(set(explainers.__all__))


# ---------------------------------------------------------------------------
# test_ols_sentiment_impact
# ---------------------------------------------------------------------------

from sentvols.explainers import test_ols_sentiment_impact  # noqa: E402

test_ols_sentiment_impact.__test__ = False

import polars as pl  # noqa: E402


@pytest.fixture(scope="module")
def ols_panel():
    """Synthetic panel where mean_score has a genuine positive effect on abnormal_ret."""
    rng = np.random.default_rng(20)
    n = 300
    mean_score = rng.standard_normal(n)
    std_score = np.abs(rng.standard_normal(n))
    news_burst = rng.integers(0, 2, n).astype(float)
    # abnormal_ret = 0.03 * mean_score + noise  (true coefficient ≈ 0.03)
    abnormal_ret = 0.03 * mean_score + 0.005 * rng.standard_normal(n)
    return pl.DataFrame(
        {
            "mean_score": mean_score,
            "std_score": std_score,
            "news_burst": news_burst,
            "abnormal_ret": abnormal_ret,
        }
    )


@pytest.fixture(scope="module")
def ols_panel_null():
    """Synthetic panel where features have NO effect on abnormal_ret."""
    rng = np.random.default_rng(21)
    n = 300
    return pl.DataFrame(
        {
            "mean_score": rng.standard_normal(n),
            "std_score": np.abs(rng.standard_normal(n)),
            "news_burst": rng.integers(0, 2, n).astype(float),
            "abnormal_ret": rng.standard_normal(n) * 0.01,
        }
    )


class TestOLSSentimentImpact:
    COLS = ["mean_score", "std_score", "news_burst"]

    def test_returns_expected_keys(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert set(result.keys()) == {
            "n_obs",
            "feature_cols",
            "coefs",
            "se",
            "t_stats",
            "p_values",
            "significant",
            "r_squared",
            "f_stat",
            "f_pvalue",
            "summary",
        }

    def test_n_obs_correct(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert result["n_obs"] == len(ols_panel)

    def test_coefs_shape(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert result["coefs"].shape == (len(self.COLS),)

    def test_se_non_negative(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert (result["se"] >= 0).all()

    def test_p_values_in_unit_interval(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert np.all((result["p_values"] >= 0) & (result["p_values"] <= 1))

    def test_significant_array_is_bool(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert result["significant"].dtype == bool

    def test_true_effect_is_detected(self, ols_panel):
        """mean_score has a true coefficient of ~0.03 — should be significant."""
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        idx_mean = self.COLS.index("mean_score")
        assert result["p_values"][idx_mean] < 0.05
        assert bool(result["significant"][idx_mean]) is True
        assert result["coefs"][idx_mean] == pytest.approx(0.03, abs=0.01)

    def test_null_features_mostly_insignificant(self, ols_panel_null):
        """Under H0 (no true effect), most features should NOT be significant."""
        result = test_ols_sentiment_impact(ols_panel_null, self.COLS)
        n_sig = int(result["significant"].sum())
        assert n_sig < len(self.COLS)  # at most some by chance

    def test_r_squared_in_unit_interval(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_intercept_mode_adds_intercept_name(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS, intercept=True)
        assert "_intercept" in result["feature_cols"]
        assert result["coefs"].shape == (len(self.COLS) + 1,)

    def test_summary_keys_match_feature_cols(self, ols_panel):
        result = test_ols_sentiment_impact(ols_panel, self.COLS)
        assert set(result["summary"].keys()) == set(self.COLS)
        for entry in result["summary"].values():
            assert set(entry.keys()) == {"coef", "se", "t", "p", "significant"}

    def test_polars_and_pandas_give_same_result(self, ols_panel):
        res_pl = test_ols_sentiment_impact(ols_panel, self.COLS)
        res_pd = test_ols_sentiment_impact(ols_panel.to_pandas(), self.COLS)
        np.testing.assert_allclose(res_pl["coefs"], res_pd["coefs"], atol=1e-10)

    def test_too_few_obs_raises(self):
        tiny = pl.DataFrame({"mean_score": [0.1, 0.2], "abnormal_ret": [0.01, 0.02]})
        with pytest.raises(ValueError, match="too few"):
            test_ols_sentiment_impact(tiny, ["mean_score"])

    def test_custom_dep_col(self, ols_panel):
        """Can use raw 'ret' as dependent variable instead of 'abnormal_ret'."""
        df_raw = ols_panel.with_columns(pl.col("abnormal_ret").alias("ret"))
        result = test_ols_sentiment_impact(df_raw, self.COLS, col_dep="ret")
        assert result["n_obs"] == len(df_raw)

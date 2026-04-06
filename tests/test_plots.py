"""Tests for sentvols.core.plots (public: sentvols.plots).

All tests are smoke tests — they verify the functions run without error
and return a matplotlib Figure without rendering on screen.
"""

from __future__ import annotations

import datetime

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import sentvols.plots as plots


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def df_news_pd() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    scores = rng.uniform(-1, 1, n)
    labels = np.where(
        scores >= 0.05, "positif", np.where(scores <= -0.05, "négatif", "neutre")
    )
    return pd.DataFrame(
        {
            "sentiment_score": scores,
            "sentiment_label": labels,
            "log_return": rng.standard_normal(n) * 0.01,
            "n_articles": rng.integers(1, 20, n),
            "pct_positif": rng.uniform(0, 1, n),
            "pct_negatif": rng.uniform(0, 1, n),
            "sentiment_sum": rng.standard_normal(n),
            "sentiment_mean": rng.uniform(-1, 1, n),
        }
    )


@pytest.fixture(scope="module")
def df_corr_pd() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 100
    return pd.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "corr_sum": rng.uniform(-0.5, 0.5, n),
            "corr_mean": rng.uniform(-0.5, 0.5, n),
            "avg_return": rng.standard_normal(n) * 0.001,
            "n_points": rng.integers(50, 500, n),
            "total_articles": rng.integers(100, 1000, n),
        }
    )


@pytest.fixture(scope="module")
def portfolio_perf() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    periods = [201901 + i for i in range(12)]
    port_ret = rng.standard_normal(12) * 0.02
    bench_ret = rng.standard_normal(12) * 0.015
    excess = port_ret - bench_ret
    return pd.DataFrame(
        {
            "period": periods,
            "port_ret": port_ret,
            "bench_ret": bench_ret,
            "excess": excess,
            "cum_port": (1 + port_ret).cumprod() - 1,
            "cum_bench": (1 + bench_ret).cumprod() - 1,
        }
    )


@pytest.fixture(scope="module")
def dummy_portfolio(portfolio_perf) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 120
    return pd.DataFrame(
        {
            "score": rng.standard_normal(n),
            "fwd_log_ret": rng.standard_normal(n) * 0.02,
        }
    )


@pytest.fixture(scope="module")
def dummy_feature_cols() -> list[str]:
    from sentvols.utils import FEATURE_COLS

    return list(FEATURE_COLS)


class _DummyModel:
    """Minimal model stub with feature_importances_."""

    def __init__(self, n_feats: int = 14):
        self._importances = np.abs(np.random.default_rng(4).standard_normal(n_feats))

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._importances


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlotSentimentDistribution:
    def test_returns_figure(self, df_news_pd):
        fig = plots.plot_sentiment_distribution(df_news_pd)
        assert isinstance(fig, plt.Figure)

    def test_no_save_by_default(self, df_news_pd, tmp_path):
        """Passing save_path saves a file; omitting it does not."""
        out = tmp_path / "sent.png"
        plots.plot_sentiment_distribution(df_news_pd, save_path=str(out))
        assert out.exists()


class TestPlotDescriptiveDashboard:
    def test_returns_figure(self, df_news_pd):
        fig = plots.plot_descriptive_dashboard(df_news_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, df_news_pd, tmp_path):
        out = tmp_path / "dashboard.png"
        plots.plot_descriptive_dashboard(df_news_pd, save_path=str(out))
        assert out.exists()


class TestPlotCorrelationAnalysis:
    def test_returns_figure(self, df_corr_pd):
        fig = plots.plot_correlation_analysis(df_corr_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, df_corr_pd, tmp_path):
        out = tmp_path / "corr.png"
        plots.plot_correlation_analysis(df_corr_pd, save_path=str(out))
        assert out.exists()


class TestPlotIntrastockScatter:
    def test_returns_figure(self, df_corr_pd):
        fig = plots.plot_intrastock_scatter(df_corr_pd)
        assert isinstance(fig, plt.Figure)


class TestPlotPortfolioPerformance:
    def test_returns_figure(self, portfolio_perf):
        fig = plots.plot_portfolio_performance(
            portfolio_perf,
            ann_port=0.12,
            ann_bench=0.08,
            sharpe=0.85,
            ic=0.05,
        )
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, portfolio_perf, tmp_path):
        out = tmp_path / "port.png"
        plots.plot_portfolio_performance(
            portfolio_perf,
            ann_port=0.12,
            ann_bench=0.08,
            sharpe=0.85,
            ic=0.05,
            save_path=str(out),
        )
        assert out.exists()


class TestPlotFeatureImportance:
    def test_returns_figure(self, dummy_feature_cols):
        clf = _DummyModel(len(dummy_feature_cols))
        reg = _DummyModel(len(dummy_feature_cols))
        fig = plots.plot_feature_importance(clf, reg, dummy_feature_cols)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, dummy_feature_cols, tmp_path):
        clf = _DummyModel(len(dummy_feature_cols))
        reg = _DummyModel(len(dummy_feature_cols))
        out = tmp_path / "fi.png"
        plots.plot_feature_importance(clf, reg, dummy_feature_cols, save_path=str(out))
        assert out.exists()


class TestPlotHypothesisPermutation:
    def test_returns_figure(self):
        rng = np.random.default_rng(5)
        perm_f1s = rng.uniform(0.45, 0.65, 500)
        fig = plots.plot_hypothesis_permutation(perm_f1s, baseline_f1=0.63, p_perm=0.04)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, tmp_path):
        rng = np.random.default_rng(6)
        perm_f1s = rng.uniform(0.45, 0.65, 500)
        out = tmp_path / "perm.png"
        plots.plot_hypothesis_permutation(
            perm_f1s, baseline_f1=0.63, p_perm=0.04, save_path=str(out)
        )
        assert out.exists()


class TestPlotScoreExplanation:
    @pytest.fixture(scope="class")
    def explanation(self):
        from sentvols.utils import FinancialVADERAnnotator

        ann = FinancialVADERAnnotator()
        return ann.explain(
            "Company beats earnings estimates, record profit, dividend hike announced"
        )

    @pytest.fixture(scope="class")
    def negative_explanation(self):
        from sentvols.utils import FinancialVADERAnnotator

        ann = FinancialVADERAnnotator()
        return ann.explain("Bankruptcy amid accounting fraud SEC investigation")

    def test_returns_figure(self, explanation):
        fig = plots.plot_score_explanation(explanation)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, explanation, tmp_path):
        path = str(tmp_path / "explain.png")
        plots.plot_score_explanation(explanation, save_path=path)
        import os

        assert os.path.exists(path)

    def test_empty_text_does_not_raise(self):
        from sentvols.utils import FinancialVADERAnnotator

        ann = FinancialVADERAnnotator()
        exp = ann.explain("")
        fig = plots.plot_score_explanation(exp)
        assert isinstance(fig, plt.Figure)

    def test_negative_text_returns_figure(self, negative_explanation):
        fig = plots.plot_score_explanation(negative_explanation)
        assert isinstance(fig, plt.Figure)

    def test_top_n_limits_bars(self, explanation):
        fig = plots.plot_score_explanation(explanation, top_n=3)
        assert isinstance(fig, plt.Figure)


class TestPlotsNamespace:
    def test_all_plot_functions_exported(self):
        expected = {
            "plot_sentiment_distribution",
            "plot_descriptive_dashboard",
            "plot_correlation_analysis",
            "plot_intrastock_scatter",
            "plot_portfolio_performance",
            "plot_feature_importance",
            "plot_hypothesis_permutation",
            "plot_score_explanation",
        }
        assert expected.issubset(set(plots.__all__))

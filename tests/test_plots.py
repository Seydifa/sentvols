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


class TestPlotSentimentFeaturesOverview:
    @pytest.fixture(scope="class")
    def feat_df_pd(self):
        rng = np.random.default_rng(10)
        n = 300
        return pd.DataFrame(
            {
                "n_articles": rng.integers(1, 30, n),
                "n_positive_articles": rng.integers(0, 15, n),
                "n_negative_articles": rng.integers(0, 10, n),
                "n_neutral_articles": rng.integers(0, 10, n),
                "news_burst": rng.integers(0, 2, n),
                "mean_score": rng.uniform(-1, 1, n),
                "median_score": rng.uniform(-1, 1, n),
                "std_score": rng.uniform(0, 0.5, n),
                "score_range": rng.uniform(0, 2, n),
                "score_skew": rng.standard_normal(n),
                "pct_positive": rng.uniform(0, 1, n),
                "pct_negative": rng.uniform(0, 1, n),
                "sentiment_balance": rng.uniform(-1, 1, n),
            }
        )

    def test_returns_figure(self, feat_df_pd):
        fig = plots.plot_sentiment_features_overview(feat_df_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, feat_df_pd, tmp_path):
        out = tmp_path / "feat_overview.png"
        plots.plot_sentiment_features_overview(feat_df_pd, save_path=str(out))
        assert out.exists()

    def test_empty_columns_does_not_raise(self):
        fig = plots.plot_sentiment_features_overview(
            pd.DataFrame({"unrelated": [1, 2, 3]})
        )
        assert isinstance(fig, plt.Figure)


class TestPlotMarketBetas:
    @pytest.fixture(scope="class")
    def beta_df_pd(self):
        rng = np.random.default_rng(11)
        n = 80
        return pd.DataFrame(
            {
                "ticker": [f"T{i:03d}" for i in range(n)],
                "beta": rng.uniform(0.3, 1.8, n),
                "alpha": rng.uniform(-0.002, 0.002, n),
                "n_obs": rng.integers(24, 120, n),
            }
        )

    def test_returns_figure(self, beta_df_pd):
        fig = plots.plot_market_betas(beta_df_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, beta_df_pd, tmp_path):
        out = tmp_path / "betas.png"
        plots.plot_market_betas(beta_df_pd, save_path=str(out))
        assert out.exists()


class TestPlotAbnormalReturns:
    @pytest.fixture(scope="class")
    def ar_df_pd(self):
        rng = np.random.default_rng(12)
        n = 500
        mkt = rng.standard_normal(n) * 0.01
        return pd.DataFrame(
            {
                "ret": mkt + rng.standard_normal(n) * 0.005,
                "abnormal_ret": rng.standard_normal(n) * 0.003,
            }
        )

    def test_returns_figure(self, ar_df_pd):
        fig = plots.plot_abnormal_returns(ar_df_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, ar_df_pd, tmp_path):
        out = tmp_path / "ar.png"
        plots.plot_abnormal_returns(ar_df_pd, save_path=str(out))
        assert out.exists()

    def test_custom_col_names(self, ar_df_pd):
        df = ar_df_pd.rename(columns={"ret": "r", "abnormal_ret": "ar"})
        fig = plots.plot_abnormal_returns(df, col_ret="r", col_ar="ar")
        assert isinstance(fig, plt.Figure)


class TestPlotOlsSentimentResults:
    @pytest.fixture(scope="class")
    def ols_result(self):
        rng = np.random.default_rng(13)
        from sentvols.features import SENTIMENT_FEATURE_COLS

        k = len(SENTIMENT_FEATURE_COLS)
        coefs = rng.standard_normal(k) * 0.01
        se = rng.uniform(0.001, 0.02, k)
        t_stats = coefs / se
        p_values = np.clip(rng.uniform(0, 0.2, k), 1e-6, 1.0)
        return {
            "n_obs": 500,
            "feature_cols": list(SENTIMENT_FEATURE_COLS),
            "coefs": coefs,
            "se": se,
            "t_stats": t_stats,
            "p_values": p_values,
            "significant": p_values < 0.05,
            "r_squared": 0.04,
            "f_stat": 2.31,
            "f_pvalue": 0.008,
            "alpha_level": 0.05,
            "summary": {},
        }

    def test_returns_figure(self, ols_result):
        fig = plots.plot_ols_sentiment_results(ols_result)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, ols_result, tmp_path):
        out = tmp_path / "ols.png"
        plots.plot_ols_sentiment_results(ols_result, save_path=str(out))
        assert out.exists()

    def test_no_significant_features(self):
        """Should still render when no features are significant."""
        result = {
            "n_obs": 100,
            "feature_cols": ["feat_a", "feat_b"],
            "coefs": np.array([0.001, -0.002]),
            "se": np.array([0.01, 0.01]),
            "t_stats": np.array([0.1, -0.2]),
            "p_values": np.array([0.9, 0.8]),
            "significant": np.array([False, False]),
            "r_squared": 0.01,
            "f_stat": 0.5,
            "f_pvalue": 0.6,
        }
        fig = plots.plot_ols_sentiment_results(result)
        assert isinstance(fig, plt.Figure)


class TestPlotPortfolioWeights:
    @pytest.fixture(scope="class")
    def portfolio_pd(self):
        rng = np.random.default_rng(14)
        periods = [201901 + i for i in range(12)]
        rows = []
        for p in periods:
            n_tickers = rng.integers(5, 20)
            raw = rng.dirichlet(np.ones(n_tickers))
            for i, w in enumerate(raw):
                rows.append({"period": p, "ticker": f"T{i:02d}", "weight": float(w)})
        return pd.DataFrame(rows)

    def test_returns_figure(self, portfolio_pd):
        fig = plots.plot_portfolio_weights(portfolio_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, portfolio_pd, tmp_path):
        out = tmp_path / "weights.png"
        plots.plot_portfolio_weights(portfolio_pd, save_path=str(out))
        assert out.exists()


class TestPlotPortfolioManagerHistory:
    @pytest.fixture(scope="class")
    def trade_history_pd(self):
        rng = np.random.default_rng(15)
        periods = [201901 + i for i in range(12)]
        rows = []
        cash = 100_000.0
        for p in periods:
            for _ in range(rng.integers(2, 8)):
                action = "buy" if rng.random() > 0.4 else "sell"
                cost = float(rng.uniform(500, 5000))
                cash = cash - cost if action == "buy" else cash + cost
                rows.append(
                    {
                        "period": p,
                        "ticker": f"T{rng.integers(0, 20):02d}",
                        "action": action,
                        "shares": float(rng.uniform(1, 50)),
                        "price": float(rng.uniform(10, 200)),
                        "cost": cost,
                        "cash_after": max(cash, 0.0),
                    }
                )
        return pd.DataFrame(rows)

    def test_returns_figure(self, trade_history_pd):
        fig = plots.plot_portfolio_manager_history(trade_history_pd)
        assert isinstance(fig, plt.Figure)

    def test_saves_file(self, trade_history_pd, tmp_path):
        out = tmp_path / "mgr_history.png"
        plots.plot_portfolio_manager_history(trade_history_pd, save_path=str(out))
        assert out.exists()

    def test_empty_history_does_not_raise(self):
        fig = plots.plot_portfolio_manager_history(pd.DataFrame())
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
            "plot_sentiment_features_overview",
            "plot_market_betas",
            "plot_abnormal_returns",
            "plot_ols_sentiment_results",
            "plot_portfolio_weights",
            "plot_portfolio_manager_history",
        }
        assert expected.issubset(set(plots.__all__))

"""Tests for sentvols.core.features (public: sentvols.features)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import sentvols.features as features
from sentvols.features import (
    SENTIMENT_FEATURE_COLS,
    add_abnormal_returns,
    build_full_feature_set,
    build_sentiment_features,
    compute_market_betas,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def annotated_df():
    """Synthetic annotated news: 4 tickers × 3 periods × 10 articles each."""
    rng = np.random.default_rng(30)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    periods = [202301, 202302, 202303]

    rows = []
    for t in tickers:
        for p in periods:
            n = 10
            scores = rng.uniform(-1, 1, n)
            labels = [
                "positif" if s >= 0.05 else ("négatif" if s <= -0.05 else "neutre")
                for s in scores
            ]
            for s, l in zip(scores, labels):
                rows.append(
                    {
                        "ticker": t,
                        "period": p,
                        "sentiment_score": float(s),
                        "sentiment_label": l,
                    }
                )

    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def annotated_df_burst(annotated_df):
    """Same as annotated_df but one ticker/period has 30 articles (burst)."""
    rng = np.random.default_rng(31)
    extra = pl.DataFrame(
        {
            "ticker": ["AAPL"] * 20,
            "period": [202301] * 20,
            "sentiment_score": rng.uniform(-1, 1, 20).tolist(),
            "sentiment_label": ["positif"] * 20,
        }
    )
    return pl.concat([annotated_df, extra])


@pytest.fixture(scope="module")
def prices_df():
    """Period-level returns for 4 tickers × 3 periods."""
    rng = np.random.default_rng(32)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    periods = [202301, 202302, 202303]
    rows = [
        {"ticker": t, "period": p, "ret": float(rng.standard_normal(1)[0] * 0.02)}
        for t in tickers
        for p in periods
    ]
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def mkt_df():
    """Market returns for 3 periods."""
    rng = np.random.default_rng(33)
    periods = [202301, 202302, 202303]
    return pl.DataFrame(
        {
            "period": periods,
            "mkt_ret": rng.standard_normal(3).tolist(),
        }
    )


@pytest.fixture(scope="module")
def panel_with_mkt(prices_df, mkt_df):
    """Joined ticker returns + market return, used for beta estimation."""
    return prices_df.join(mkt_df, on="period", how="left")


# ---------------------------------------------------------------------------
# SENTIMENT_FEATURE_COLS constant
# ---------------------------------------------------------------------------


class TestSentimentFeatureCols:
    def test_is_tuple(self):
        assert isinstance(SENTIMENT_FEATURE_COLS, tuple)

    def test_contains_core_fields(self):
        required = {
            "n_articles",
            "n_positive_articles",
            "n_negative_articles",
            "mean_score",
            "median_score",
            "std_score",
            "news_burst",
            "pct_positive",
            "pct_negative",
            "sentiment_balance",
        }
        assert required.issubset(set(SENTIMENT_FEATURE_COLS))

    def test_no_duplicates(self):
        assert len(SENTIMENT_FEATURE_COLS) == len(set(SENTIMENT_FEATURE_COLS))


# ---------------------------------------------------------------------------
# build_sentiment_features
# ---------------------------------------------------------------------------


class TestBuildSentimentFeatures:
    def test_returns_polars_dataframe(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        assert isinstance(result, pl.DataFrame)

    def test_row_count(self, annotated_df):
        """One row per (ticker, period) = 4 × 3 = 12."""
        result = build_sentiment_features(annotated_df)
        assert len(result) == 12

    def test_expected_columns_present(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        for col in SENTIMENT_FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_n_articles_correct(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        # Every (ticker, period) has exactly 10 articles in the base fixture
        assert (result["n_articles"] == 10).all()

    def test_article_counts_sum_correctly(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        total = (
            result["n_positive_articles"]
            + result["n_negative_articles"]
            + result["n_neutral_articles"]
        )
        np.testing.assert_array_equal(total.to_numpy(), result["n_articles"].to_numpy())

    def test_pct_positive_in_unit_interval(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        assert (result["pct_positive"] >= 0).all() and (
            result["pct_positive"] <= 1
        ).all()

    def test_pct_negative_in_unit_interval(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        assert (result["pct_negative"] >= 0).all() and (
            result["pct_negative"] <= 1
        ).all()

    def test_sentiment_balance_equals_diff(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        expected = result["pct_positive"] - result["pct_negative"]
        np.testing.assert_allclose(
            result["sentiment_balance"].to_numpy(),
            expected.to_numpy(),
            atol=1e-10,
        )

    def test_std_score_non_negative(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        assert (result["std_score"] >= 0).all()

    def test_score_range_non_negative(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        assert (result["score_range"] >= 0).all()

    def test_news_burst_is_binary(self, annotated_df):
        result = build_sentiment_features(annotated_df)
        vals = set(result["news_burst"].to_list())
        assert vals.issubset({0, 1})

    def test_news_burst_fires_for_high_volume(self, annotated_df_burst):
        """The AAPL/202301 row (30 articles) must be flagged as burst."""
        result = build_sentiment_features(annotated_df_burst)
        row = result.filter((pl.col("ticker") == "AAPL") & (pl.col("period") == 202301))
        assert len(row) == 1
        assert row["news_burst"][0] == 1

    def test_custom_column_names(self, annotated_df):
        """Custom col_ticker / col_period names are respected."""
        renamed = annotated_df.rename({"ticker": "stock", "period": "ym"})
        result = build_sentiment_features(renamed, col_ticker="stock", col_period="ym")
        assert "stock" in result.columns
        assert "ym" in result.columns


# ---------------------------------------------------------------------------
# compute_market_betas
# ---------------------------------------------------------------------------


class TestComputeMarketBetas:
    def test_returns_polars_dataframe(self, panel_with_mkt):
        result = compute_market_betas(panel_with_mkt)
        assert isinstance(result, pl.DataFrame)

    def test_one_row_per_ticker(self, panel_with_mkt):
        result = compute_market_betas(panel_with_mkt)
        assert len(result) == panel_with_mkt["ticker"].n_unique()

    def test_expected_columns(self, panel_with_mkt):
        result = compute_market_betas(panel_with_mkt)
        assert set(result.columns) == {"ticker", "alpha", "beta", "n_obs"}

    def test_insufficient_obs_beta_defaults_to_one(self):
        """Tickers with < min_obs should receive beta = 1.0."""
        df = pl.DataFrame(
            {
                "ticker": ["X"] * 5 + ["Y"] * 30,
                "period": list(range(5)) + list(range(30)),
                "ret": np.random.default_rng(40).standard_normal(35).tolist(),
                "mkt_ret": np.random.default_rng(41).standard_normal(35).tolist(),
            }
        )
        result = compute_market_betas(df, min_obs=10)
        x_row = result.filter(pl.col("ticker") == "X")
        assert x_row["beta"][0] == pytest.approx(1.0)

    def test_n_obs_correct(self, panel_with_mkt):
        result = compute_market_betas(panel_with_mkt)
        # 3 periods per ticker
        assert (result["n_obs"] == 3).all()


# ---------------------------------------------------------------------------
# add_abnormal_returns
# ---------------------------------------------------------------------------


class TestAddAbnormalReturns:
    def test_column_added(self, panel_with_mkt):
        beta_df = compute_market_betas(panel_with_mkt)
        result = add_abnormal_returns(panel_with_mkt, beta_df)
        assert "abnormal_ret" in result.columns

    def test_row_count_preserved(self, panel_with_mkt):
        beta_df = compute_market_betas(panel_with_mkt)
        result = add_abnormal_returns(panel_with_mkt, beta_df)
        assert len(result) == len(panel_with_mkt)

    def test_abnormal_ret_differs_from_raw(self, panel_with_mkt):
        """Abnormal return should differ from raw return (market component removed)."""
        beta_df = compute_market_betas(panel_with_mkt)
        result = add_abnormal_returns(panel_with_mkt, beta_df)
        # At least some rows should differ
        raw = result["ret"].to_numpy()
        ab = result["abnormal_ret"].to_numpy()
        assert not np.allclose(raw, ab)

    def test_abnormal_ret_formula(self):
        """Manually verify AR = r - (alpha + beta * mkt_ret)."""
        df = pl.DataFrame(
            {
                "ticker": ["A", "A"],
                "period": [1, 2],
                "ret": [0.05, -0.02],
                "mkt_ret": [0.03, 0.01],
            }
        )
        beta_df = pl.DataFrame({"ticker": ["A"], "alpha": [0.001], "beta": [1.2]})
        result = add_abnormal_returns(df, beta_df)
        expected_ar = np.array([0.05, -0.02]) - (0.001 + 1.2 * np.array([0.03, 0.01]))
        np.testing.assert_allclose(
            result["abnormal_ret"].to_numpy(), expected_ar, atol=1e-10
        )

    def test_alpha_beta_columns_dropped(self, panel_with_mkt):
        """alpha and beta columns must NOT remain in the output."""
        beta_df = compute_market_betas(panel_with_mkt)
        result = add_abnormal_returns(panel_with_mkt, beta_df)
        assert "alpha" not in result.columns
        assert "beta" not in result.columns


# ---------------------------------------------------------------------------
# build_full_feature_set (integration)
# ---------------------------------------------------------------------------


class TestBuildFullFeatureSet:
    def test_returns_polars_dataframe(self, annotated_df, prices_df, mkt_df):
        result = build_full_feature_set(
            annotated_df,
            prices_df,
            mkt_df,
            col_price_ret="ret",
            col_mkt="mkt_ret",
            min_beta_obs=2,
        )
        assert isinstance(result, pl.DataFrame)

    def test_abnormal_ret_column_present(self, annotated_df, prices_df, mkt_df):
        result = build_full_feature_set(
            annotated_df,
            prices_df,
            mkt_df,
            col_price_ret="ret",
            col_mkt="mkt_ret",
            min_beta_obs=2,
        )
        assert "abnormal_ret" in result.columns

    def test_sentiment_feature_cols_present(self, annotated_df, prices_df, mkt_df):
        result = build_full_feature_set(
            annotated_df,
            prices_df,
            mkt_df,
            col_price_ret="ret",
            col_mkt="mkt_ret",
            min_beta_obs=2,
        )
        for col in SENTIMENT_FEATURE_COLS:
            assert col in result.columns, f"Missing: {col}"


# ---------------------------------------------------------------------------
# Namespace export
# ---------------------------------------------------------------------------


class TestFeaturesNamespace:
    def test_public_symbols_exported(self):
        expected = {
            "SENTIMENT_FEATURE_COLS",
            "build_sentiment_features",
            "compute_market_betas",
            "add_abnormal_returns",
            "build_full_feature_set",
        }
        assert expected.issubset(set(features.__all__))

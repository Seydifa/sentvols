"""Tests for sentvols.core.utils (public: sentvols.utils)."""

from __future__ import annotations

import datetime
import tempfile
import textwrap

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sentvols.utils import (
    FEATURE_COLS,
    aggregate_daily_sentiment,
    annotate_news,
    build_monthly_features,
    compute_log_returns,
    load_and_clean_news,
    merge_sentiment_prices,
    prepare_splits,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_news_csv(tmp_path, rows=10) -> str:
    """Write a minimal analyst_ratings_processed-style CSV and return its path."""
    path = tmp_path / "news.csv"
    lines = [",title,date,stock"]
    for i in range(rows):
        ticker = f"AAPL" if i % 2 == 0 else "GOOG"
        score = "+15%" if i % 3 == 0 else "earnings miss"
        headline = f"headline {i}: {score}"
        date = f"2019-0{(i % 9) + 1}-01 10:00:00-04:00"
        lines.append(f"{i},{headline},{date},{ticker}")
    path.write_text("\n".join(lines))
    return str(path)


class _MockAnnotator:
    """Deterministic annotator: score = 0.5 for every text."""

    def score(self, text: str) -> float:
        return 0.5


def _make_daily_sentiment(n=30) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    base = datetime.date(2019, 1, 2)
    dates = [base + datetime.timedelta(days=i) for i in range(n)]
    tickers = ["AAPL"] * (n // 2) + ["GOOG"] * (n - n // 2)
    return pl.DataFrame(
        {
            "date_only": dates,
            "ticker": tickers,
            "sentiment_sum": rng.standard_normal(n).tolist(),
            "sentiment_mean": rng.standard_normal(n).tolist(),
            "sentiment_std": np.abs(rng.standard_normal(n)).tolist(),
            "n_articles": rng.integers(1, 10, n).tolist(),
            "pct_positif": rng.uniform(0, 1, n).tolist(),
            "pct_negatif": rng.uniform(0, 1, n).tolist(),
        }
    )


def _make_prices(n=30) -> pl.DataFrame:
    rng = np.random.default_rng(1)
    base = datetime.date(2019, 1, 2)
    dates = [base + datetime.timedelta(days=i) for i in range(n)]
    tickers = ["AAPL"] * (n // 2) + ["GOOG"] * (n - n // 2)
    prices = (100 + np.cumsum(rng.standard_normal(n) * 0.5)).tolist()
    return pl.DataFrame(
        {
            "date_only": dates,
            "ticker": tickers,
            "close_price": prices,
        }
    )


# ---------------------------------------------------------------------------
# load_and_clean_news
# ---------------------------------------------------------------------------


class TestLoadAndCleanNews:
    def test_returns_polars_dataframe(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        assert isinstance(df, pl.DataFrame)

    def test_has_expected_columns(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        assert set(df.columns) == {"headline", "date", "date_only", "ticker"}

    def test_tickers_are_uppercase(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        for t in df["ticker"].to_list():
            assert t == t.upper()

    def test_no_nulls_in_key_columns(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        for col in ["headline", "date", "date_only", "ticker"]:
            assert df[col].null_count() == 0

    def test_date_only_is_date_type(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        assert df["date_only"].dtype == pl.Date


# ---------------------------------------------------------------------------
# annotate_news
# ---------------------------------------------------------------------------


class TestAnnotateNews:
    def test_adds_sentiment_columns(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        assert "sentiment_score" in df.columns
        assert "sentiment_label" in df.columns

    def test_score_is_mock_value(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        assert (df["sentiment_score"] == 0.5).all()

    def test_label_positive_for_score_above_threshold(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        # 0.5 >= 0.05 → positif
        assert (df["sentiment_label"] == "positif").all()


# ---------------------------------------------------------------------------
# aggregate_daily_sentiment
# ---------------------------------------------------------------------------


class TestAggregateDailySentiment:
    def test_output_columns(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        agg = aggregate_daily_sentiment(df)
        expected = {
            "date_only",
            "ticker",
            "sentiment_sum",
            "sentiment_mean",
            "sentiment_std",
            "n_articles",
            "pct_positif",
            "pct_negatif",
        }
        assert expected.issubset(set(agg.columns))

    def test_n_articles_positive(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        agg = aggregate_daily_sentiment(df)
        assert (agg["n_articles"] > 0).all()

    def test_pct_positif_between_0_and_1(self, tmp_path):
        path = _make_news_csv(tmp_path)
        df = load_and_clean_news(path)
        df = annotate_news(df, _MockAnnotator())
        agg = aggregate_daily_sentiment(df)
        assert agg["pct_positif"].min() >= 0.0
        assert agg["pct_positif"].max() <= 1.0


# ---------------------------------------------------------------------------
# compute_log_returns
# ---------------------------------------------------------------------------


class TestComputeLogReturns:
    def test_adds_log_return_column(self):
        df = _make_prices()
        result = compute_log_returns(df)
        assert "log_return" in result.columns

    def test_no_nulls_after_compute(self):
        df = _make_prices()
        result = compute_log_returns(df)
        assert result["log_return"].null_count() == 0

    def test_first_return_is_zero(self):
        """First observation per ticker has no previous price → filled with 0."""
        df = pl.DataFrame(
            {
                "date_only": [datetime.date(2019, 1, i) for i in range(1, 6)],
                "ticker": ["AAPL"] * 5,
                "close_price": [100.0, 102.0, 101.0, 103.0, 105.0],
            }
        )
        result = compute_log_returns(df)
        first_return = result.filter(pl.col("date_only") == datetime.date(2019, 1, 1))[
            "log_return"
        ][0]
        assert first_return == 0.0

    def test_log_return_values_correct(self):
        df = pl.DataFrame(
            {
                "date_only": [datetime.date(2019, 1, i) for i in range(1, 4)],
                "ticker": ["X"] * 3,
                "close_price": [100.0, 110.0, 99.0],
            }
        )
        result = compute_log_returns(df).sort("date_only")
        expected_second = float(np.log(110.0 / 100.0))
        assert abs(result["log_return"][1] - expected_second) < 1e-9


# ---------------------------------------------------------------------------
# merge_sentiment_prices
# ---------------------------------------------------------------------------


class TestMergeSentimentPrices:
    def test_inner_join_reduces_rows(self):
        sent = _make_daily_sentiment(30)
        prices = _make_prices(20)
        merged = merge_sentiment_prices(sent, prices)
        assert len(merged) <= min(len(sent), len(prices))

    def test_merged_has_log_return(self):
        sent = _make_daily_sentiment(30)
        prices = _make_prices(30)
        merged = merge_sentiment_prices(sent, prices)
        assert "log_return" in merged.columns

    def test_merged_has_sentiment_columns(self):
        sent = _make_daily_sentiment(30)
        prices = _make_prices(30)
        merged = merge_sentiment_prices(sent, prices)
        for col in ["sentiment_sum", "sentiment_mean", "n_articles"]:
            assert col in merged.columns


# ---------------------------------------------------------------------------
# build_monthly_features
# ---------------------------------------------------------------------------


def _make_merged_df(n_months=24) -> pl.DataFrame:
    """Build a synthetic daily merged DataFrame covering n_months of AAPL & GOOG."""
    rng = np.random.default_rng(3)
    rows = []
    base = datetime.date(2015, 1, 2)
    for m in range(n_months):
        year = 2015 + (m // 12)
        month = (m % 12) + 1
        for ticker in ["AAPL", "GOOG"]:
            for d in range(15):  # 15 trading days per synthetic month
                day = base + datetime.timedelta(days=m * 30 + d)
                rows.append(
                    {
                        "date_only": day,
                        "ticker": ticker,
                        "sentiment_sum": float(rng.standard_normal()),
                        "sentiment_mean": float(rng.standard_normal()),
                        "sentiment_std": float(abs(rng.standard_normal())),
                        "n_articles": int(rng.integers(1, 5)),
                        "pct_positif": float(rng.uniform()),
                        "pct_negatif": float(rng.uniform()),
                        "log_return": float(rng.standard_normal() * 0.01),
                    }
                )
    return pl.DataFrame(rows)


class TestBuildMonthlyFeatures:
    def test_returns_polars_dataframe(self):
        df = build_monthly_features(_make_merged_df(12))
        assert isinstance(df, pl.DataFrame)

    def test_has_target_columns(self):
        df = build_monthly_features(_make_merged_df(12))
        for col in ["target_return", "target_class", "ym"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_has_feature_cols(self):
        df = build_monthly_features(_make_merged_df(12))
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature column: {col}"

    def test_target_class_is_binary(self):
        df = build_monthly_features(_make_merged_df(12))
        assert set(df["target_class"].unique().to_list()).issubset({0, 1})

    def test_no_nulls_in_key_features(self):
        df = build_monthly_features(_make_merged_df(12))
        for col in ["target_return", "ret_lag1", "ret_lag2", "sent_lag1", "vol_lag1"]:
            assert df[col].null_count() == 0, f"Nulls in {col}"


# ---------------------------------------------------------------------------
# prepare_splits
# ---------------------------------------------------------------------------


class TestPrepareSplits:
    @pytest.fixture(scope="class")
    def splits(self):
        df = build_monthly_features(_make_merged_df(36))
        # 36 months from 2015-01 covers up to 2017-12; use 2016-06 / 2016-12
        # so that test data falls in 2017 (ym > 201612)
        return prepare_splits(df, train_end=201606, val_end=201612)

    def test_returns_dict(self, splits):
        assert isinstance(splits, dict)

    def test_expected_keys_present(self, splits):
        required = {
            "X_train_sc",
            "X_val_sc",
            "X_test_sc",
            "X_train_reg_sc",
            "X_val_reg_sc",
            "X_test_reg_sc",
            "y_cls_train",
            "y_cls_val",
            "y_cls_test",
            "y_reg_train",
            "y_reg_val",
            "y_reg_test",
            "X_train_reg_pos",
            "y_reg_train_pos",
            "df_model_test",
            "scaler_clf",
            "scaler_reg",
        }
        assert required.issubset(splits.keys())

    def test_feature_dimension_matches(self, splits):
        n_feats = len(FEATURE_COLS)
        for key in ["X_train_sc", "X_val_sc", "X_test_sc"]:
            assert splits[key].shape[1] == n_feats, f"{key} has wrong n_features"

    def test_label_binary(self, splits):
        for key in ["y_cls_train", "y_cls_val", "y_cls_test"]:
            assert set(splits[key]).issubset({0, 1}), f"{key} should be binary"

    def test_df_model_test_columns(self, splits):
        df = splits["df_model_test"]
        assert set(df.columns) == {"period", "ticker", "fwd_log_ret"}

    def test_positive_subset_is_subset_of_train(self, splits):
        n_pos = len(splits["X_train_reg_pos"])
        n_train = len(splits["X_train_reg_sc"])
        assert n_pos <= n_train

    def test_feature_cols_exported(self):
        assert isinstance(FEATURE_COLS, tuple)
        assert len(FEATURE_COLS) == 14

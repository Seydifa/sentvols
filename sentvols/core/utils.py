from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import pandas as pd
import polars as pl
import yfinance as yf
from sklearn.preprocessing import RobustScaler, StandardScaler

from .exports import registration


FEATURE_COLS: tuple[str, ...] = (
    "sent_sum_mean",
    "sent_sum_std",
    "sent_mean_avg",
    "sent_std_avg",
    "n_articles_total",
    "pct_pos_avg",
    "pct_neg_avg",
    "sentiment_balance",
    "sent_lag1",
    "ret_lag1",
    "ret_lag2",
    "monthly_vol",
    "vol_lag1",
    "n_trading_days",
)
registration(module="utils", name="FEATURE_COLS")(FEATURE_COLS)


@registration(module="utils")
def load_and_clean_news(csv_path: str) -> pl.DataFrame:
    return (
        pl.scan_csv(csv_path, infer_schema_length=10_000)
        .rename({"title": "headline"})
        .with_columns(
            [
                pl.col("date")
                .str.slice(0, 19)
                .str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False)
                .alias("date"),
                pl.col("stock").str.to_uppercase().str.strip_chars().alias("ticker"),
            ]
        )
        .filter(
            pl.col("headline").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("ticker").is_not_null()
        )
        .with_columns(pl.col("date").dt.date().alias("date_only"))
        .select(["headline", "date", "date_only", "ticker"])
        .collect()
    )


@registration(module="utils")
def annotate_news(
    df_news: pl.DataFrame,
    annotator,
    normalizer=None,
    batch_size: int = 256,
    workers: int = 1,
) -> pl.DataFrame:
    """Score and label news headlines.

    Parameters
    ----------
    df_news : pl.DataFrame
        Must contain a ``headline`` column.
    annotator :
        A ``FinancialVADERAnnotator`` or ``FinancialLLMAnnotator`` instance
        (any object with a ``score_batch(texts) -> list[float]`` method, or
        a ``score(text) -> float`` method as fallback).
    normalizer : FinancialTextNormalizer or None, default None
        Optional LLM pre-processing stage.  When provided, headlines are
        passed through ``normalize_if_needed_batch()`` before scoring —
        short texts bypass the LLM, long texts are batched in chunks of
        *batch_size* so the backend (vLLM, Transformers) can use PagedAttention
        / padded-generate instead of one GPU call per row.

        Two extra columns are added to the output:

        * ``normalized_headline`` — the text actually scored (``null`` when
          the normalizer passed the text through unchanged).
        * ``normalization_reasoning`` — chain-of-thought trace from the
          backend (``null`` when not available or when LLM was not used).

        When ``None``, the function behaves exactly as before — no extra
        columns, no breaking change.
    batch_size : int, default 256
        Maximum number of prompts sent to the LLM backend per call.  Tune
        down to reduce GPU memory pressure; tune up to reduce round-trip
        overhead with API-hosted backends.  Ignored when ``normalizer`` is
        ``None``.
    workers : int, default 1
        Thread-pool size for backends that lack a ``batch_call()`` method.
        Has no effect when the backend already exposes ``batch_call()``.
    """
    headlines = df_news["headline"].to_list()

    # ── label helper (shared) ──────────────────────────────────────────────
    def _make_label_col(scores: list[float]) -> pl.Series:
        return pl.Series(
            "sentiment_label",
            [
                "positif" if s >= 0.05 else ("négatif" if s <= -0.05 else "neutre")
                for s in scores
            ],
            dtype=pl.Utf8,
        )

    # ── batch-score helper: prefers score_batch(), falls back to score() ──
    def _batch_score(texts: list[str]) -> list[float]:
        if hasattr(annotator, "score_batch"):
            return annotator.score_batch(texts, workers=workers)
        return [annotator.score(t) for t in texts]

    # ── no-normalizer path ────────────────────────────────────────────────
    if normalizer is None:
        scores = _batch_score(headlines)
        return df_news.with_columns(
            pl.Series("sentiment_score", scores, dtype=pl.Float64),
            _make_label_col(scores),
        )

    # ── normalizer path ───────────────────────────────────────────────────
    if hasattr(normalizer, "normalize_if_needed_batch"):
        results = normalizer.normalize_if_needed_batch(
            headlines, batch_size=batch_size, workers=workers
        )
    else:
        # Fallback for custom normalizer objects without the batch method
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor as _TPE

            with _TPE(max_workers=workers) as pool:
                results = list(pool.map(normalizer.normalize_if_needed, headlines))
        else:
            results = [normalizer.normalize_if_needed(h) for h in headlines]

    texts_to_score = [r.normalized_text for r in results]
    scores = _batch_score(texts_to_score)
    normalized_texts = [r.normalized_text if r.llm_used else None for r in results]
    reasoning_traces = [r.reasoning_trace for r in results]

    return df_news.with_columns(
        pl.Series("sentiment_score", scores, dtype=pl.Float64),
        pl.Series("normalized_headline", normalized_texts, dtype=pl.Utf8),
        pl.Series("normalization_reasoning", reasoning_traces, dtype=pl.Utf8),
        _make_label_col(scores),
    )


@registration(module="utils")
def aggregate_daily_sentiment(df_news: pl.DataFrame) -> pl.DataFrame:
    return (
        df_news.group_by(["date_only", "ticker"])
        .agg(
            [
                pl.col("sentiment_score").sum().alias("sentiment_sum"),
                pl.col("sentiment_score").mean().alias("sentiment_mean"),
                pl.col("sentiment_score").std().alias("sentiment_std"),
                pl.col("sentiment_score").count().alias("n_articles"),
                (pl.col("sentiment_label") == "positif").mean().alias("pct_positif"),
                (pl.col("sentiment_label") == "négatif").mean().alias("pct_negatif"),
            ]
        )
        .sort(["ticker", "date_only"])
    )


@registration(module="utils")
def download_stock_prices(
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    batch_size: int = 500,
    cache_path: str | None = None,
) -> pl.DataFrame:
    if cache_path and os.path.exists(cache_path):
        return pl.read_parquet(cache_path)

    all_tickers = list(tickers)
    batches: list[list[str]] = [
        all_tickers[i : i + batch_size] for i in range(0, len(all_tickers), batch_size)
    ]
    dfs_list: list[pl.DataFrame] = []

    for batch in batches:
        try:
            raw = yf.download(
                batch,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close_col = raw["Close"]
            else:
                close_col = raw[["Close"]].rename(columns={"Close": batch[0]})

            df_batch = (
                pl.from_pandas(close_col.reset_index())
                .unpivot(index="Date", variable_name="ticker", value_name="close_price")
                .with_columns(
                    [
                        pl.col("ticker").str.to_uppercase().str.strip_chars(),
                        pl.col("Date").cast(pl.Date).alias("date_only"),
                    ]
                )
                .select(["date_only", "ticker", "close_price"])
                .drop_nulls(subset=["close_price"])
            )
            dfs_list.append(df_batch)
        except Exception:
            pass

    df_prices = pl.concat(dfs_list)
    if cache_path:
        df_prices.write_parquet(cache_path)
    return df_prices


@registration(module="utils")
def compute_log_returns(df_prices: pl.DataFrame) -> pl.DataFrame:
    return (
        df_prices.sort(["ticker", "date_only"])
        .with_columns(
            pl.col("close_price").shift(1).over("ticker").alias("_close_prev")
        )
        .with_columns(
            (pl.col("close_price") / pl.col("_close_prev")).log().alias("log_return")
        )
        .with_columns(pl.col("log_return").fill_null(0.0))
        .drop("_close_prev")
    )


@registration(module="utils")
def merge_sentiment_prices(
    daily_sentiment: pl.DataFrame,
    df_prices: pl.DataFrame,
) -> pl.DataFrame:
    df_with_returns = compute_log_returns(df_prices)
    return daily_sentiment.join(
        df_with_returns, on=["date_only", "ticker"], how="inner"
    )


@registration(module="utils")
def build_monthly_features(df_final: pl.DataFrame) -> pl.DataFrame:
    df_monthly = (
        df_final.with_columns(
            [
                pl.col("date_only").dt.year().cast(pl.Int32).alias("year"),
                pl.col("date_only").dt.month().cast(pl.Int32).alias("month"),
            ]
        )
        .group_by(["ticker", "year", "month"])
        .agg(
            [
                pl.col("sentiment_sum").mean().alias("sent_sum_mean"),
                pl.col("sentiment_sum").std().alias("sent_sum_std"),
                pl.col("sentiment_mean").mean().alias("sent_mean_avg"),
                pl.col("sentiment_std").mean().alias("sent_std_avg"),
                pl.col("n_articles").sum().alias("n_articles_total"),
                pl.col("pct_positif").mean().alias("pct_pos_avg"),
                pl.col("pct_negatif").mean().alias("pct_neg_avg"),
                pl.col("log_return").sum().alias("monthly_return"),
                pl.col("log_return").std().alias("monthly_vol"),
                pl.col("date_only").count().alias("n_trading_days"),
            ]
        )
        .sort(["ticker", "year", "month"])
        .with_columns(
            (pl.col("year") * 100 + pl.col("month")).cast(pl.Int32).alias("ym")
        )
    )
    return (
        df_monthly.sort(["ticker", "ym"])
        .with_columns(
            [
                pl.col("monthly_return")
                .shift(-1)
                .over("ticker")
                .alias("target_return"),
                pl.col("monthly_return").shift(1).over("ticker").alias("ret_lag1"),
                pl.col("monthly_return").shift(2).over("ticker").alias("ret_lag2"),
                pl.col("sent_sum_mean").shift(1).over("ticker").alias("sent_lag1"),
                pl.col("monthly_vol").shift(1).over("ticker").alias("vol_lag1"),
                (pl.col("pct_pos_avg") - pl.col("pct_neg_avg")).alias(
                    "sentiment_balance"
                ),
            ]
        )
        .with_columns(
            (pl.col("target_return") > 0).cast(pl.Int32).alias("target_class")
        )
        .drop_nulls(
            subset=["target_return", "ret_lag1", "ret_lag2", "sent_lag1", "vol_lag1"]
        )
    )


@registration(module="utils")
def prepare_splits(
    df_features: pl.DataFrame,
    feature_cols: list[str] | None = None,
    train_end: int = 201612,
    val_end: int = 201812,
) -> dict:
    if feature_cols is None:
        feature_cols = list(FEATURE_COLS)

    df_pd = df_features.to_pandas()
    df_pd[feature_cols] = df_pd[feature_cols].fillna(0.0)

    mask_train = df_pd["ym"] <= train_end
    mask_val = (df_pd["ym"] > train_end) & (df_pd["ym"] <= val_end)
    mask_test = df_pd["ym"] > val_end

    X_train = df_pd.loc[mask_train, feature_cols].values
    X_val = df_pd.loc[mask_val, feature_cols].values
    X_test = df_pd.loc[mask_test, feature_cols].values

    y_cls_train = df_pd.loc[mask_train, "target_class"].values
    y_cls_val = df_pd.loc[mask_val, "target_class"].values
    y_cls_test = df_pd.loc[mask_test, "target_class"].values

    y_reg_train = df_pd.loc[mask_train, "target_return"].values
    y_reg_val = df_pd.loc[mask_val, "target_return"].values
    y_reg_test = df_pd.loc[mask_test, "target_return"].values

    scaler_clf = StandardScaler()
    X_train_sc = scaler_clf.fit_transform(X_train)
    X_val_sc = scaler_clf.transform(X_val)
    X_test_sc = scaler_clf.transform(X_test)

    scaler_reg = RobustScaler()
    X_train_reg_sc = scaler_reg.fit_transform(X_train)
    X_val_reg_sc = scaler_reg.transform(X_val)
    X_test_reg_sc = scaler_reg.transform(X_test)

    mask_pos_train = y_cls_train == 1
    mask_pos_val = y_cls_val == 1

    df_model_test = (
        df_pd.loc[mask_test, ["ticker", "ym", "target_return"]]
        .rename(columns={"ym": "period", "target_return": "fwd_log_ret"})
        .reset_index(drop=True)
    )

    return {
        "feature_cols": feature_cols,
        "df_pd": df_pd,
        "X_train_sc": X_train_sc,
        "X_val_sc": X_val_sc,
        "X_test_sc": X_test_sc,
        "X_train_reg_sc": X_train_reg_sc,
        "X_val_reg_sc": X_val_reg_sc,
        "X_test_reg_sc": X_test_reg_sc,
        "X_train_reg_pos": X_train_reg_sc[mask_pos_train],
        "y_reg_train_pos": y_reg_train[mask_pos_train],
        "X_val_reg_pos": X_val_reg_sc[mask_pos_val],
        "y_reg_val_pos": y_reg_val[mask_pos_val],
        "y_cls_train": y_cls_train,
        "y_cls_val": y_cls_val,
        "y_cls_test": y_cls_test,
        "y_reg_train": y_reg_train,
        "y_reg_val": y_reg_val,
        "y_reg_test": y_reg_test,
        "df_model_test": df_model_test,
        "scaler_clf": scaler_clf,
        "scaler_reg": scaler_reg,
    }

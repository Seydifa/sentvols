"""Feature engineering for the sentvols pipeline.

This module transforms scored & labelled news DataFrames into period-level
feature sets ready for ML training and econometric testing.

Two complementary outputs are produced:

* **Sentiment features** — cross-sectional, per (ticker, period) aggregates
  derived from annotator scores.
* **Market-adjusted returns** — excess returns after removing the
  systematic market component (market-model residual / abnormal return).
  These are the *correct* dependent variable for testing whether sentiment
  impacts stock performance without confounding by market-wide moves.

Public API (accessible via ``sentvols.features``)
--------------------------------------------------
- ``SENTIMENT_FEATURE_COLS``   — ordered tuple of the sentiment feature names
- ``build_sentiment_features`` — aggregate annotated news per period
- ``compute_market_betas``     — OLS β̂_i for each ticker vs market return
- ``add_abnormal_returns``     — subtract market component from raw returns
- ``build_full_feature_set``   — convenience: combines sentiment + abnormal returns
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

from .exports import registration

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

#: Ordered tuple of the sentiment feature columns produced by
#: :func:`build_sentiment_features`.  Use this as ``feature_cols`` when
#: calling :func:`sentvols.utils.prepare_splits`.
SENTIMENT_FEATURE_COLS: tuple[str, ...] = (
    # volume / event features
    "n_articles",
    "n_positive_articles",
    "n_negative_articles",
    "n_neutral_articles",
    "news_burst",  # 1 when n_articles > 75th pct across all periods
    # central tendency
    "mean_score",
    "median_score",
    # dispersion / shape
    "std_score",
    "score_range",  # max − min score (disagreement proxy)
    "score_skew",  # skewness of score distribution
    # directional balance
    "pct_positive",
    "pct_negative",
    "sentiment_balance",  # pct_positive − pct_negative
)
registration(module="features", name="SENTIMENT_FEATURE_COLS")(SENTIMENT_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Sentiment feature engineering
# ---------------------------------------------------------------------------


@registration(module="features")
def build_sentiment_features(
    df_annotated: pl.DataFrame,
    col_ticker: str = "ticker",
    col_period: str = "period",
    col_score: str = "sentiment_score",
    col_label: str = "sentiment_label",
    pos_label: str = "positif",
    neg_label: str = "négatif",
    neu_label: str = "neutre",
    burst_quantile: float = 0.75,
) -> pl.DataFrame:
    """Aggregate per-article sentiment scores into period-level features.

    Parameters
    ----------
    df_annotated : pl.DataFrame
        Output of :func:`sentvols.utils.annotate_news` (or equivalent).
        Must contain ``col_ticker``, ``col_period``, ``col_score``,
        ``col_label``.
    col_ticker, col_period : str
        Grouping columns.
    col_score : str
        Numeric sentiment score column (float in [−1, 1]).
    col_label : str
        Categorical label column.
    pos_label, neg_label, neu_label : str
        Label values for positive, negative, and neutral articles.
    burst_quantile : float
        Quantile threshold for the ``news_burst`` indicator.  A period is
        flagged when its article count exceeds this quantile **across all
        (ticker, period) cells** in the DataFrame.  Default ``0.75`` (top
        quarter of coverage density).

    Returns
    -------
    pl.DataFrame
        One row per (ticker, period) with columns listed in
        :data:`SENTIMENT_FEATURE_COLS` plus the grouping columns.

    Notes
    -----
    ``score_skew`` falls back to 0 when a period has fewer than 3 articles
    (insufficient for a meaningful skewness estimate).
    ``std_score``, ``score_range``, and ``score_skew`` are set to 0 for
    single-article periods.
    """
    agg = (
        df_annotated.group_by([col_ticker, col_period])
        .agg(
            pl.len().alias("n_articles"),
            (pl.col(col_label) == pos_label)
            .sum()
            .cast(pl.Int32)
            .alias("n_positive_articles"),
            (pl.col(col_label) == neg_label)
            .sum()
            .cast(pl.Int32)
            .alias("n_negative_articles"),
            (pl.col(col_label) == neu_label)
            .sum()
            .cast(pl.Int32)
            .alias("n_neutral_articles"),
            pl.col(col_score).mean().alias("mean_score"),
            pl.col(col_score).median().alias("median_score"),
            pl.col(col_score).std(ddof=1).fill_null(0.0).alias("std_score"),
            (pl.col(col_score).max() - pl.col(col_score).min())
            .fill_null(0.0)
            .alias("score_range"),
            (pl.col(col_label) == pos_label).mean().alias("pct_positive"),
            (pl.col(col_label) == neg_label).mean().alias("pct_negative"),
        )
        .sort([col_ticker, col_period])
    )

    # score_skew: compute outside main agg (needs numpy per-group)
    # Use map_groups via polars group_by — efficient for small group sizes
    def _skew(group_df: pl.DataFrame) -> pl.DataFrame:
        scores = group_df[col_score].drop_nulls().to_numpy()
        skew_val = float(_safe_skew(scores))
        return group_df.head(1).with_columns(pl.lit(skew_val).alias("score_skew"))

    skew_df = (
        df_annotated.group_by([col_ticker, col_period], maintain_order=True)
        .map_groups(_skew)
        .select([col_ticker, col_period, "score_skew"])
    )

    agg = agg.join(skew_df, on=[col_ticker, col_period], how="left")

    # news_burst: flag periods in the top burst_quantile of article count
    threshold = float(
        agg["n_articles"].quantile(burst_quantile, interpolation="linear") or 1.0
    )
    agg = agg.with_columns(
        (pl.col("n_articles") > threshold).cast(pl.Int32).alias("news_burst"),
        (pl.col("pct_positive") - pl.col("pct_negative")).alias("sentiment_balance"),
    )

    # Return columns in canonical order
    keep = [col_ticker, col_period] + list(SENTIMENT_FEATURE_COLS)
    available = [c for c in keep if c in agg.columns]
    return agg.select(available)


def _safe_skew(arr: np.ndarray) -> float:
    """Population skewness; returns 0 for n < 3 or zero variance."""
    if len(arr) < 3:
        return 0.0
    mu = arr.mean()
    sigma = arr.std()
    if sigma == 0:
        return 0.0
    return float(((arr - mu) ** 3).mean() / sigma**3)


# ---------------------------------------------------------------------------
# Market-adjustment (abnormal return computation)
# ---------------------------------------------------------------------------


@registration(module="features")
def compute_market_betas(
    df: pl.DataFrame,
    col_ticker: str = "ticker",
    col_period: str = "period",
    col_ret: str = "ret",
    col_mkt: str = "mkt_ret",
    min_obs: int = 24,
) -> pl.DataFrame:
    """Estimate each ticker's market beta via OLS (no-intercept optional).

    The market model is:

    .. math::

        r_{i,t} = \\alpha_i + \\beta_i \\cdot r_{m,t} + \\varepsilon_{i,t}

    Parameters
    ----------
    df : pl.DataFrame
        Panel data with at least ``col_ticker``, ``col_period``, ``col_ret``,
        ``col_mkt``.
    col_ticker, col_period : str
        Cross-sectional and time identifiers.
    col_ret : str
        Individual stock return column.
    col_mkt : str
        Market (index) return column — same calendar period as ``col_ret``.
    min_obs : int
        Minimum number of observations required to fit a ticker.  Tickers
        with fewer observations receive ``beta = 1.0`` (market-like proxy).

    Returns
    -------
    pl.DataFrame
        Columns: ``col_ticker``, ``alpha``, ``beta``, ``n_obs``.
        One row per ticker.
    """
    rows = []
    pdf = df.select([col_ticker, col_period, col_ret, col_mkt]).to_pandas()

    for ticker, grp in pdf.groupby(col_ticker):
        grp_clean = grp.dropna(subset=[col_ret, col_mkt])
        n = len(grp_clean)
        if n < min_obs:
            rows.append({"ticker": ticker, "alpha": 0.0, "beta": 1.0, "n_obs": n})
            continue

        y = grp_clean[col_ret].values
        x = grp_clean[col_mkt].values
        # OLS with intercept: [1, x]
        X_mat = np.column_stack([np.ones(n), x])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)
            alpha_val, beta_val = float(coeffs[0]), float(coeffs[1])
        except np.linalg.LinAlgError:
            alpha_val, beta_val = 0.0, 1.0
        rows.append(
            {"ticker": ticker, "alpha": alpha_val, "beta": beta_val, "n_obs": n}
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "alpha": pl.Float64,
                "beta": pl.Float64,
                "n_obs": pl.Int64,
            }
        )

    beta_df = pl.DataFrame(rows).with_columns(pl.col("ticker").cast(pl.Utf8))
    # rename ticker to match col_ticker if different
    if col_ticker != "ticker":
        beta_df = beta_df.rename({"ticker": col_ticker})
    return beta_df


@registration(module="features")
def add_abnormal_returns(
    df: pl.DataFrame,
    beta_df: pl.DataFrame,
    col_ticker: str = "ticker",
    col_ret: str = "ret",
    col_mkt: str = "mkt_ret",
) -> pl.DataFrame:
    """Subtract the market-model expected return to get abnormal returns.

    .. math::

        AR_{i,t} = r_{i,t} - (\\hat{\\alpha}_i + \\hat{\\beta}_i \\cdot r_{m,t})

    Parameters
    ----------
    df : pl.DataFrame
        Panel with ``col_ticker``, ``col_ret``, ``col_mkt``.
    beta_df : pl.DataFrame
        Output of :func:`compute_market_betas`.
    col_ticker, col_ret, col_mkt : str
        Column name overrides.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with an added ``abnormal_ret`` column.
    """
    joined = df.join(
        beta_df.select([col_ticker, "alpha", "beta"]), on=col_ticker, how="left"
    )
    joined = joined.with_columns(
        pl.col("alpha").fill_null(0.0),
        pl.col("beta").fill_null(1.0),
    )
    return joined.with_columns(
        (pl.col(col_ret) - pl.col("alpha") - pl.col("beta") * pl.col(col_mkt)).alias(
            "abnormal_ret"
        )
    ).drop(["alpha", "beta"])


@registration(module="features")
def build_full_feature_set(
    df_annotated: pl.DataFrame,
    df_prices: pl.DataFrame,
    mkt_returns: pl.DataFrame,
    col_ticker: str = "ticker",
    col_period: str = "period",
    col_score: str = "sentiment_score",
    col_label: str = "sentiment_label",
    col_price_ret: str = "ret",
    col_mkt: str = "mkt_ret",
    min_beta_obs: int = 24,
    burst_quantile: float = 0.75,
) -> pl.DataFrame:
    """Convenience wrapper: build sentiment features + abnormal returns in one call.

    Combines :func:`build_sentiment_features`, :func:`compute_market_betas`,
    and :func:`add_abnormal_returns` into a single pipeline call.

    Parameters
    ----------
    df_annotated : pl.DataFrame
        Scored news with ticker, period, score, label columns.
    df_prices : pl.DataFrame
        Period-level stock returns.  Must have ``col_ticker``, ``col_period``,
        ``col_price_ret``.
    mkt_returns : pl.DataFrame
        Market-index returns.  Must have ``col_period`` and ``col_mkt``.
    col_ticker, col_period : str
        Grouping column names (must match across all input DataFrames).
    col_score, col_label : str
        Sentiment columns in ``df_annotated``.
    col_price_ret : str
        Return column in ``df_prices``.
    col_mkt : str
        Market return column in ``mkt_returns``.
    min_beta_obs, burst_quantile :
        Forwarded to :func:`compute_market_betas` and
        :func:`build_sentiment_features`.

    Returns
    -------
    pl.DataFrame
        One row per (ticker, period) with all sentiment features plus
        ``ret`` (raw return), ``mkt_ret``, and ``abnormal_ret``.
    """
    feat = build_sentiment_features(
        df_annotated,
        col_ticker=col_ticker,
        col_period=col_period,
        col_score=col_score,
        col_label=col_label,
        burst_quantile=burst_quantile,
    )

    # join returns
    returns_with_mkt = df_prices.rename({col_price_ret: "ret"}).join(
        mkt_returns, on=col_period, how="left"
    )

    # estimate betas (using the full price panel)
    beta_df = compute_market_betas(
        returns_with_mkt,
        col_ticker=col_ticker,
        col_period=col_period,
        min_obs=min_beta_obs,
    )

    # compute abnormal returns
    returns_ab = add_abnormal_returns(returns_with_mkt, beta_df, col_ticker=col_ticker)

    # join everything
    return feat.join(
        returns_ab.select([col_ticker, col_period, "ret", col_mkt, "abnormal_ret"]),
        on=[col_ticker, col_period],
        how="left",
    )

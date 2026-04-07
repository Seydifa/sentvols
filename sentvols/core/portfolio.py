from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import joblib
import numpy as np
import polars as pl

from .exports import registration

if TYPE_CHECKING:
    from .models import SentvolsClassifier, SentvolsRegressor


# ---------------------------------------------------------------------------
# Annualisation-frequency aliases
# Sub-daily values assume US equity market hours (390 min/day, 252 trading days).
# ---------------------------------------------------------------------------

_FREQ_ALIASES: dict[str, int] = {
    "monthly": 12,
    "weekly": 52,
    "daily": 252,
    "hourly": 1512,  # 252 × 6 trading hours
    "30min": 3276,  # 252 × 13 half-hour bars
    "15min": 6552,  # 252 × 26 quarter-hour bars
    "5min": 19656,  # 252 × 78 five-minute bars
    "1min": 98280,  # 252 × 390 one-minute bars
}

# ---------------------------------------------------------------------------
# Built-in weighting strategies
# ---------------------------------------------------------------------------

_WEIGHTING_STRATEGIES = frozenset({"equal", "score", "softmax", "rank"})


def _weights_equal(scores: np.ndarray) -> np.ndarray:
    """Uniform weight: 1/n per selected stock."""
    return np.full(len(scores), 1.0 / len(scores))


def _weights_score(scores: np.ndarray) -> np.ndarray:
    """Weight proportional to min-max normalised score."""
    s = scores - scores.min()
    total = s.sum()
    return s / total if total > 0 else _weights_equal(scores)


def _weights_softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax of scores — exponential tilting toward high-scorers."""
    e = np.exp(scores - scores.max())  # numerically stable
    return e / e.sum()


def _weights_rank(scores: np.ndarray) -> np.ndarray:
    """Weight ∝ 1/rank (rank 1 = highest score), normalised to sum to 1."""
    ranks = np.argsort(np.argsort(-scores)) + 1  # 1-based dense rank
    w = 1.0 / ranks.astype(float)
    return w / w.sum()


_WEIGHT_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "equal": _weights_equal,
    "score": _weights_score,
    "softmax": _weights_softmax,
    "rank": _weights_rank,
}

# Schema for PortfolioManager trade history (used to build empty DataFrame)
_TRADE_SCHEMA: dict[str, type] = {
    "period": pl.Utf8,
    "ticker": pl.Utf8,
    "action": pl.Utf8,
    "shares": pl.Float64,
    "price": pl.Float64,
    "cost": pl.Float64,
    "cash_after": pl.Float64,
}


@registration(module="portfolio")
class PortfolioBuilder:
    """Build and evaluate a long-only factor portfolio.

    Central component of the sentvols proof-of-concept pipeline.  It
    consumes scored predictions from a :class:`SentvolsClassifier` and a
    :class:`SentvolsRegressor` (or raw scores supplied directly), selects
    the top-n instruments per period, assigns portfolio weights, and
    provides performance analytics.

    All DataFrames are processed with **polars** for performance on large
    financial datasets.  Pandas DataFrames are accepted transparently and
    converted at the boundary.

    Parameters
    ----------
    n : int
        Number of top-scoring stocks to select per period.
    col_period : str
        Column that identifies the cross-sectional period.  Any dtype that
        polars can sort and group works: integers (``202501``), date strings,
        or native ``pl.Datetime`` values — including hourly timestamps.
    col_ret : str
        Column containing the realised forward return, used only by
        ``performance()`` and ``metrics()``.
    scoring_fn : callable or None
        ``(pred_prob: ndarray, pred_ret: ndarray) -> ndarray`` that returns a
        per-row score.  Defaults to ``pred_prob * pred_ret``.
    weighting : str | callable
        Portfolio weighting strategy applied *within* each period to the
        selected top-n stocks.  Built-in options:

        * ``"equal"`` *(default)* — uniform 1/n weight.
        * ``"score"`` — weight proportional to min-max normalised score.
        * ``"softmax"`` — softmax of scores (exponential tilting).
        * ``"rank"`` — weight ∝ 1/rank, normalised to sum to 1.

        Pass a callable ``fn(scores: ndarray) -> ndarray`` to use a fully
        custom strategy.  The callable receives the 1-D score array for
        the period and must return a weight array that sums to 1.
    freq : int | str
        Annualisation factor — number of periods per year.  Pass an integer
        directly or one of the named aliases:
        ``"monthly"`` (12), ``"weekly"`` (52), ``"daily"`` (252),
        ``"hourly"`` (1512), ``"30min"`` (3276), ``"15min"`` (6552),
        ``"5min"`` (19656), ``"1min"`` (98280).
    """

    def __init__(
        self,
        n: int = 50,
        col_period: str = "period",
        col_ret: str = "fwd_log_ret",
        scoring_fn: Callable | None = None,
        weighting: str | Callable = "equal",
        freq: int | str = 12,
    ) -> None:
        self.n = n
        self.col_period = col_period
        self.col_ret = col_ret
        self.scoring_fn = scoring_fn
        if isinstance(weighting, str) and weighting not in _WEIGHTING_STRATEGIES:
            raise ValueError(
                f"Unknown weighting '{weighting}'. "
                f"Valid options: {sorted(_WEIGHTING_STRATEGIES)} or a callable."
            )
        self.weighting = weighting
        if isinstance(freq, str):
            if freq not in _FREQ_ALIASES:
                raise ValueError(
                    f"Unknown freq alias '{freq}'. Valid aliases: {list(_FREQ_ALIASES)}"
                )
            self.freq: int = _FREQ_ALIASES[freq]
        else:
            self.freq = int(freq)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_polars(df) -> pl.DataFrame:
        """Accept polars or pandas DataFrames; always return a pl.DataFrame."""
        if isinstance(df, pl.DataFrame):
            return df
        try:
            import pandas as _pd  # optional at runtime

            if isinstance(df, _pd.DataFrame):
                return pl.from_pandas(df)
        except ImportError:
            pass
        raise TypeError(
            f"Expected a polars or pandas DataFrame, got {type(df).__name__}."
        )

    def _check_columns(self, df: pl.DataFrame, required: list[str]) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"PortfolioBuilder: missing columns {missing} in DataFrame. "
                f"Available: {df.columns}"
            )

    def _score(self, pred_prob: np.ndarray, pred_ret: np.ndarray) -> np.ndarray:
        if self.scoring_fn is not None:
            return np.asarray(self.scoring_fn(pred_prob, pred_ret))
        return pred_prob * pred_ret

    def _get_weight_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        if callable(self.weighting):
            return self.weighting
        return _WEIGHT_FNS[self.weighting]  # type: ignore[index]

    def _apply_weights(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add a ``weight`` column by applying the weighting strategy per period.

        Uses :py:meth:`polars.GroupBy.map_groups` so the weight function
        receives a contiguous numpy array per period — efficient even for
        custom callables.
        """
        weight_fn = self._get_weight_fn()

        def _add_weight(group_df: pl.DataFrame) -> pl.DataFrame:
            scores = group_df["score"].to_numpy()
            return group_df.with_columns(pl.Series("weight", weight_fn(scores)))

        return (
            df.group_by(self.col_period, maintain_order=True)
            .map_groups(_add_weight)
            .sort(self.col_period)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        df_test=None,
        clf: SentvolsClassifier | None = None,
        reg: SentvolsRegressor | None = None,
        X_test_clf_sc=None,
        X_test_reg_sc=None,
        *,
        scores: np.ndarray | None = None,
    ) -> pl.DataFrame:
        """Score every row, retain the top-n stocks per period, and assign weights.

        Can be called in two ways:

        **ML path** — pass ``clf``, ``reg``, ``X_test_clf_sc``,
        ``X_test_reg_sc`` (the standard pipeline usage).

        **Scores path** — pass a precomputed ``scores`` array directly and
        omit the model arguments.  This makes the builder fully usable
        without a test dataset or trained models (e.g. rule-based or
        external scoring).

        ``col_ret`` is **not** required here — the result can be used
        immediately by :class:`PortfolioManager` for live trading without
        forward returns.

        Parameters
        ----------
        df_test : DataFrame
            Universe DataFrame. Must contain ``col_period``.  For the ML
            path, also the feature columns used by ``clf``/``reg``.
        clf, reg : SentvolsClassifier, SentvolsRegressor, optional
            Trained models.  Required for the ML path.
        X_test_clf_sc, X_test_reg_sc : array-like, optional
            Scaled feature matrices.  Required for the ML path.
        scores : ndarray, optional
            Pre-computed per-row scores.  Use instead of clf/reg.

        Returns
        -------
        pl.DataFrame
            Input columns plus ``pred_prob``, ``pred_ret`` (ML path only),
            ``score``, and ``weight``, filtered to the top-n per period.
        """
        if scores is None:
            # --- ML path ---
            if clf is None or reg is None:
                raise ValueError(
                    "Either pass trained 'clf' and 'reg' models, "
                    "or supply a precomputed 'scores' array."
                )
            df = self._to_polars(df_test)
            self._check_columns(df, [self.col_period])
            pred_prob = clf.predict_proba(X_test_clf_sc)[:, 1]
            pred_ret = reg.predict(X_test_reg_sc)
            scores_arr = self._score(pred_prob, pred_ret)
            df = df.with_columns(
                pl.Series("pred_prob", pred_prob),
                pl.Series("pred_ret", pred_ret),
                pl.Series("score", scores_arr),
            )
        else:
            # --- Scores-only path ---
            df = self._to_polars(df_test)
            self._check_columns(df, [self.col_period])
            df = df.with_columns(pl.Series("score", np.asarray(scores)))

        portfolio = (
            df.with_columns(
                pl.col("score")
                .rank(method="ordinal", descending=True)
                .over(self.col_period)
                .alias("_rank")
            )
            .filter(pl.col("_rank") <= self.n)
            .drop("_rank")
            .sort(self.col_period)
        )
        return self._apply_weights(portfolio)

    def performance(self, portfolio, df_all_test) -> pl.DataFrame:
        """Compute period-level portfolio vs benchmark returns.

        If the ``portfolio`` DataFrame contains a ``weight`` column (set by
        :meth:`build`), the portfolio return is computed as a weighted mean.
        Otherwise a simple equal-weight mean is used as a fallback.

        Returns
        -------
        pl.DataFrame
            Columns: ``period``, ``port_ret``, ``bench_ret``, ``excess``,
            ``cum_port``, ``cum_bench``.
        """
        portfolio = self._to_polars(portfolio)
        df_all_test = self._to_polars(df_all_test)
        self._check_columns(portfolio, [self.col_period, self.col_ret])
        self._check_columns(df_all_test, [self.col_period, self.col_ret])

        if "weight" in portfolio.columns:
            port_agg = (
                portfolio.group_by(self.col_period)
                .agg((pl.col(self.col_ret) * pl.col("weight")).sum().alias("port_ret"))
                .sort(self.col_period)
            )
        else:
            port_agg = (
                portfolio.group_by(self.col_period)
                .agg(pl.col(self.col_ret).mean().alias("port_ret"))
                .sort(self.col_period)
            )

        bench_agg = (
            df_all_test.group_by(self.col_period)
            .agg(pl.col(self.col_ret).mean().alias("bench_ret"))
            .sort(self.col_period)
        )
        perf = (
            port_agg.join(bench_agg, on=self.col_period)
            .rename({self.col_period: "period"})
            .sort("period")
            .with_columns(
                (pl.col("port_ret") - pl.col("bench_ret")).alias("excess"),
                ((1 + pl.col("port_ret")).cum_prod() - 1).alias("cum_port"),
                ((1 + pl.col("bench_ret")).cum_prod() - 1).alias("cum_bench"),
            )
        )
        return perf

    def metrics(self, perf, portfolio) -> dict:
        """Compute summary performance metrics.

        Returns
        -------
        dict
            ``ann_port``, ``ann_bench``, ``sharpe``, ``ic``.
        """
        perf = self._to_polars(perf)
        portfolio = self._to_polars(portfolio)
        self._check_columns(perf, ["port_ret", "bench_ret"])
        self._check_columns(portfolio, ["score", self.col_ret])

        port_ret = perf["port_ret"].to_numpy()
        bench_ret = perf["bench_ret"].to_numpy()
        scores = portfolio["score"].to_numpy()
        rets = portfolio[self.col_ret].to_numpy()

        ann_port = float(port_ret.mean() * self.freq)
        ann_bench = float(bench_ret.mean() * self.freq)
        vol = float(np.std(port_ret, ddof=1))
        sharpe = float(port_ret.mean() / vol * np.sqrt(self.freq) if vol > 0 else 0.0)
        ic = float(np.corrcoef(scores, rets)[0, 1])
        return {
            "ann_port": ann_port,
            "ann_bench": ann_bench,
            "sharpe": sharpe,
            "ic": ic,
        }


@registration(module="portfolio")
class PortfolioManager:
    """Stateful portfolio manager: tracks positions and executes buy/sell at each period.

    Maintains a cash balance and a set of holdings (ticker → shares).  At
    each rebalance call it computes the minimum set of trades needed to
    match the target weights produced by :class:`PortfolioBuilder` and
    records every trade in an internal history.

    **Persistence** — no external database is required.  The full object
    state (cash, positions, history) is serialised with :meth:`save` /
    :meth:`load` via joblib.  For multi-user or multi-process deployments,
    replace the internal ``_history`` list with a DuckDB or PostgreSQL
    table using the same schema as :attr:`trade_history`.

    Parameters
    ----------
    initial_cash : float
        Starting cash balance (in account currency).
    col_ticker : str
        Column name identifying each asset in the portfolio DataFrame.
    col_price : str
        Column name containing prices when a DataFrame is passed to
        :meth:`rebalance`.
    col_period : str
        Column name identifying the timestamp / period in the portfolio
        DataFrame.
    transaction_cost : float
        Proportional round-trip cost per trade (e.g. ``0.001`` = 0.1 %).
        Applied to the gross notional value of every buy and sell.
    """

    def __init__(
        self,
        initial_cash: float,
        col_ticker: str = "ticker",
        col_price: str = "price",
        col_period: str = "period",
        transaction_cost: float = 0.0,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.col_ticker = col_ticker
        self.col_price = col_price
        self.col_period = col_period
        self.transaction_cost = float(transaction_cost)

        self._cash: float = self.initial_cash
        self._positions: dict[str, float] = {}  # ticker -> shares held
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        """Current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, float]:
        """Copy of current holdings: {ticker: shares}."""
        return dict(self._positions)

    @property
    def trade_history(self) -> pl.DataFrame:
        """All trades executed so far as a polars DataFrame.

        Columns: ``period``, ``ticker``, ``action``, ``shares``, ``price``,
        ``cost``, ``cash_after``.
        """
        if not self._history:
            return pl.DataFrame(schema=_TRADE_SCHEMA)
        return pl.DataFrame(self._history)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prices_dict(self, prices, period=None) -> dict[str, float]:
        """Resolve *prices* to a ``{ticker: price}`` mapping.

        Accepts:
        - ``dict[str, float]`` — used as-is.
        - polars or pandas DataFrame with ``col_ticker`` / ``col_price``
          columns.  If a ``col_period`` column is present and ``period`` is
          given, the DataFrame is first filtered to that period.
        """
        if isinstance(prices, dict):
            return {str(k): float(v) for k, v in prices.items()}

        df = PortfolioBuilder._to_polars(prices)

        if period is not None and self.col_period in df.columns:
            df = df.filter(pl.col(self.col_period) == period)

        if self.col_ticker not in df.columns or self.col_price not in df.columns:
            raise ValueError(
                f"PortfolioManager: price DataFrame must contain "
                f"'{self.col_ticker}' and '{self.col_price}' columns. "
                f"Available: {df.columns}"
            )
        return dict(zip(df[self.col_ticker].to_list(), df[self.col_price].to_list()))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def portfolio_value(self, prices) -> float:
        """Total portfolio value: cash + market value of all open positions.

        Parameters
        ----------
        prices : dict[str, float] | DataFrame
            Current prices per ticker.
        """
        price_map = self._prices_dict(prices)
        equity = sum(
            qty * price_map.get(ticker, 0.0)
            for ticker, qty in self._positions.items()
            if qty != 0
        )
        return self._cash + equity

    def snapshot(self) -> dict:
        """Return a lightweight summary of the current state.

        Returns
        -------
        dict
            ``cash``, ``positions`` (ticker → shares), ``n_positions``
            (number of non-zero holdings).
        """
        return {
            "cash": self._cash,
            "positions": dict(self._positions),
            "n_positions": sum(1 for v in self._positions.values() if v != 0),
        }

    def rebalance(self, portfolio, prices) -> pl.DataFrame:
        """Rebalance holdings to match the target weights in ``portfolio``.

        For each period in the DataFrame, the method:

        1. Computes the total portfolio value (cash + open positions).
        2. Derives target notional for every ticker based on its ``weight``.
        3. Calculates the delta vs current holdings.
        4. Executes the minimum necessary buys and sells, applying
           ``transaction_cost`` to each trade.

        Parameters
        ----------
        portfolio : pl.DataFrame
            Output of :meth:`PortfolioBuilder.build`.  Must contain
            ``col_period``, ``col_ticker``, and ``"weight"`` columns.
        prices : dict[str, float] | DataFrame
            Current prices per ticker.  Pass a DataFrame with a
            ``col_period`` column to use period-specific prices.

        Returns
        -------
        pl.DataFrame
            Trades executed during this rebalance call (same schema as
            :attr:`trade_history`).
        """
        portfolio = PortfolioBuilder._to_polars(portfolio)
        for col in [self.col_period, self.col_ticker, "weight"]:
            if col not in portfolio.columns:
                raise ValueError(
                    f"PortfolioManager.rebalance: missing column '{col}'. "
                    f"Available: {portfolio.columns}"
                )

        trades_before = len(self._history)
        periods = portfolio[self.col_period].unique().sort()

        for period in periods:
            grp = portfolio.filter(pl.col(self.col_period) == period)
            price_map = self._prices_dict(prices, period=period)

            # Snapshot total value at the start of this period
            total_value = self.portfolio_value(price_map)

            target_weights: dict[str, float] = dict(
                zip(grp[self.col_ticker].to_list(), grp["weight"].to_list())
            )

            # All tickers that appear in current positions *or* target
            all_tickers = set(self._positions.keys()) | set(target_weights.keys())

            for ticker in sorted(all_tickers):
                price = price_map.get(str(ticker), 0.0)
                if price <= 0:
                    continue

                current_shares = self._positions.get(ticker, 0.0)
                target_notional = total_value * target_weights.get(ticker, 0.0)
                current_notional = current_shares * price
                delta = target_notional - current_notional

                if abs(delta) < 1e-6:
                    continue

                shares_delta = delta / price
                cost = abs(delta) * self.transaction_cost

                if delta > 0:
                    action = "buy"
                    self._cash -= delta + cost
                else:
                    action = "sell"
                    self._cash += abs(delta) - cost

                new_shares = self._positions.get(ticker, 0.0) + shares_delta
                if abs(new_shares) < 1e-10:
                    self._positions.pop(ticker, None)
                else:
                    self._positions[ticker] = new_shares

                self._history.append(
                    {
                        "period": str(period),
                        "ticker": str(ticker),
                        "action": action,
                        "shares": float(abs(shares_delta)),
                        "price": float(price),
                        "cost": float(cost),
                        "cash_after": float(self._cash),
                    }
                )

        new_trades = self._history[trades_before:]
        if not new_trades:
            return pl.DataFrame(schema=_TRADE_SCHEMA)
        return pl.DataFrame(new_trades)

    def close_all(self, prices) -> pl.DataFrame:
        """Liquidate all open positions at the given prices.

        Useful at the end of a backtest or when shutting down a live
        strategy.  Proceeds (minus transaction costs) are added to cash.

        Parameters
        ----------
        prices : dict[str, float] | DataFrame
            Current prices per ticker.

        Returns
        -------
        pl.DataFrame
            Sell trades executed (same schema as :attr:`trade_history`).
        """
        liquidation_weights = pl.DataFrame(
            {
                self.col_period: ["_close_all"] * len(self._positions),
                self.col_ticker: list(self._positions.keys()),
                "weight": [0.0] * len(self._positions),
            }
        )
        if self._positions:
            return self.rebalance(liquidation_weights, prices)
        return pl.DataFrame(schema=_TRADE_SCHEMA)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the full manager state to *path* via joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PortfolioManager":
        """Deserialise a :class:`PortfolioManager` previously saved with :meth:`save`."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected PortfolioManager, got {type(obj).__name__}")
        return obj

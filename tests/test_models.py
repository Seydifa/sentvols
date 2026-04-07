"""Tests for sentvols.core.models (public: sentvols.models)."""

from __future__ import annotations

import pathlib
import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sentvols.models import SentvolsClassifier, SentvolsRegressor
from sentvols.portfolio import PortfolioBuilder, PortfolioManager


# ---------------------------------------------------------------------------
# Fixtures — small deterministic datasets
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clf_data():
    """Binary classification dataset: 200 train / 50 val / 50 test."""
    rng = np.random.default_rng(0)
    n_train, n_val, n_test = 200, 50, 50
    n_feats = 14

    X_tr = rng.standard_normal((n_train, n_feats))
    y_tr = (X_tr[:, 0] + rng.standard_normal(n_train) * 0.5 > 0).astype(int)

    X_va = rng.standard_normal((n_val, n_feats))
    y_va = (X_va[:, 0] + rng.standard_normal(n_val) * 0.5 > 0).astype(int)

    X_te = rng.standard_normal((n_test, n_feats))
    y_te = (X_te[:, 0] + rng.standard_normal(n_test) * 0.5 > 0).astype(int)

    return X_tr, y_tr, X_va, y_va, X_te, y_te


@pytest.fixture(scope="module")
def reg_data():
    """Regression dataset: 200 train / 50 val / 50 test."""
    rng = np.random.default_rng(1)
    n_train, n_val, n_test = 200, 50, 50
    n_feats = 14

    X_tr = rng.standard_normal((n_train, n_feats))
    y_tr = X_tr[:, 0] * 0.02 + rng.standard_normal(n_train) * 0.01

    X_va = rng.standard_normal((n_val, n_feats))
    y_va = X_va[:, 0] * 0.02 + rng.standard_normal(n_val) * 0.01

    X_te = rng.standard_normal((n_test, n_feats))
    y_te = X_te[:, 0] * 0.02 + rng.standard_normal(n_test) * 0.01

    return X_tr, y_tr, X_va, y_va, X_te, y_te


_FAST_CLF_PARAMS = {
    "n_estimators": 10,
    "max_depth": 3,
}

_FAST_REG_PARAMS = {
    "n_estimators": 10,
    "max_depth": 3,
}

_BASE_CLF = RandomForestClassifier(random_state=42, n_jobs=1)
_BASE_REG = RandomForestRegressor(random_state=42, n_jobs=1)


# ---------------------------------------------------------------------------
# SentvolsClassifier
# ---------------------------------------------------------------------------


class TestSentvolsClassifier:
    def test_predict_before_fit_raises(self):
        clf = SentvolsClassifier(estimator=_BASE_CLF)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 14)))

    def test_predict_proba_before_fit_raises(self):
        clf = SentvolsClassifier(estimator=_BASE_CLF)
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(np.zeros((5, 14)))

    def test_feature_importances_before_fit_raises(self):
        clf = SentvolsClassifier(estimator=_BASE_CLF)
        with pytest.raises(RuntimeError):
            _ = clf.feature_importances_

    def test_fit_without_params_raises(self):
        clf = SentvolsClassifier(estimator=_BASE_CLF)
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.integers(0, 2, 20)
        with pytest.raises(RuntimeError, match="optimize"):
            clf.fit(X, y)

    def test_fit_predict_with_explicit_params(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate_keys(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        metrics = clf.evaluate(X_te, y_te)
        assert set(metrics.keys()) == {"f1", "precision", "recall"}
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_feature_importances_shape(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        imp = clf.feature_importances_
        assert imp.shape == (X_tr.shape[1],)
        assert (imp >= 0).all()


# ---------------------------------------------------------------------------
# SentvolsRegressor
# ---------------------------------------------------------------------------


class TestSentvolsRegressor:
    def test_predict_before_fit_raises(self):
        reg = SentvolsRegressor(estimator=_BASE_REG)
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.zeros((5, 14)))

    def test_feature_importances_before_fit_raises(self):
        reg = SentvolsRegressor(estimator=_BASE_REG)
        with pytest.raises(RuntimeError):
            _ = reg.feature_importances_

    def test_fit_without_params_raises(self):
        reg = SentvolsRegressor(estimator=_BASE_REG)
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.standard_normal(20)
        with pytest.raises(RuntimeError, match="optimize"):
            reg.fit(X, y)

    def test_fit_predict_with_explicit_params(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
        reg.fit(X_tr, y_tr, params=_FAST_REG_PARAMS)
        preds = reg.predict(X_te)
        assert preds.shape == (len(y_te),)
        assert preds.dtype == np.float64

    def test_evaluate_keys(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
        reg.fit(X_tr, y_tr, params=_FAST_REG_PARAMS)
        metrics = reg.evaluate(X_te, y_te)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_feature_importances_shape(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
        reg.fit(X_tr, y_tr, params=_FAST_REG_PARAMS)
        imp = reg.feature_importances_
        assert imp.shape == (X_tr.shape[1],)


# ---------------------------------------------------------------------------
# PortfolioBuilder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_clf_reg(clf_data, reg_data):
    X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
    Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te = reg_data

    clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
    clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)

    reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
    reg.fit(Xr_tr, yr_tr, params=_FAST_REG_PARAMS)

    return clf, reg, X_te, Xr_te


@pytest.fixture(scope="module")
def dummy_test_df():
    """50-row polars DataFrame with monthly integer periods."""
    rng = np.random.default_rng(2)
    periods = [201901] * 25 + [201902] * 25
    tickers = [f"T{i:03d}" for i in range(25)] * 2
    fwd = rng.standard_normal(50) * 0.02
    return pl.DataFrame({"period": periods, "ticker": tickers, "fwd_log_ret": fwd})


@pytest.fixture(scope="module")
def dummy_hourly_df():
    """50-row polars DataFrame with pl.Datetime hourly periods."""
    rng = np.random.default_rng(99)
    base = datetime.datetime(2025, 1, 2, 9, 30)
    # 2 hourly periods × 25 tickers
    hours = [base + datetime.timedelta(hours=h) for h in range(2) for _ in range(25)]
    fwd = rng.standard_normal(50) * 0.005
    return pl.DataFrame({"ts": hours, "fwd_log_ret": fwd})


class TestPortfolioBuilder:
    def test_build_returns_polars_dataframe(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        assert isinstance(portfolio, pl.DataFrame)

    def test_build_adds_expected_columns(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        assert {"pred_prob", "pred_ret", "score"}.issubset(set(portfolio.columns))

    def test_build_top_n_per_period(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        n = 5
        builder = PortfolioBuilder(n=n)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        counts = portfolio.group_by("period").agg(pl.len().alias("cnt"))["cnt"]
        assert (counts <= n).all()

    def test_performance_returns_expected_columns(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf = builder.performance(portfolio, dummy_test_df)
        assert isinstance(perf, pl.DataFrame)
        assert {"port_ret", "bench_ret", "excess", "cum_port", "cum_bench"}.issubset(
            set(perf.columns)
        )

    def test_metrics_returns_expected_keys(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf = builder.performance(portfolio, dummy_test_df)
        m = builder.metrics(perf, portfolio)
        assert set(m.keys()) == {"ann_port", "ann_bench", "sharpe", "ic"}
        assert isinstance(m["sharpe"], float)
        assert isinstance(m["ic"], float)

    def test_pandas_input_accepted(self, fitted_clf_reg):
        """Pandas DataFrames must be transparently converted to polars."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        rng = np.random.default_rng(3)
        df_pd = pd.DataFrame(
            {
                "date": np.repeat([202301, 202302], 25).tolist(),
                "ret_fwd": (rng.standard_normal(50) * 0.02).tolist(),
            }
        )
        builder = PortfolioBuilder(n=5, col_period="date", col_ret="ret_fwd")
        portfolio = builder.build(df_pd, clf, reg, X_te, Xr_te)
        assert isinstance(portfolio, pl.DataFrame)
        perf = builder.performance(portfolio, df_pd)
        m = builder.metrics(perf, portfolio)
        assert set(m.keys()) == {"ann_port", "ann_bench", "sharpe", "ic"}

    def test_custom_scoring_fn(self, fitted_clf_reg, dummy_test_df):
        """Custom scoring function is applied instead of default."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=5, scoring_fn=lambda p, r: p)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        np.testing.assert_allclose(
            portfolio["score"].to_numpy(), portfolio["pred_prob"].to_numpy()
        )

    def test_freq_integer(self, fitted_clf_reg, dummy_test_df):
        """freq parameter scales annualised returns correctly."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        b12 = PortfolioBuilder(n=10, freq=12)
        b52 = PortfolioBuilder(n=10, freq=52)
        port12 = b12.build(dummy_test_df, clf, reg, X_te, Xr_te)
        port52 = b52.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf12 = b12.performance(port12, dummy_test_df)
        perf52 = b52.performance(port52, dummy_test_df)
        m12 = b12.metrics(perf12, port12)
        m52 = b52.metrics(perf52, port52)
        assert abs(m52["ann_port"] / m12["ann_port"] - 52 / 12) < 1e-6

    def test_freq_string_aliases(self):
        """Named freq aliases resolve to the correct integer."""
        assert PortfolioBuilder(freq="monthly").freq == 12
        assert PortfolioBuilder(freq="weekly").freq == 52
        assert PortfolioBuilder(freq="daily").freq == 252
        assert PortfolioBuilder(freq="hourly").freq == 1512
        assert PortfolioBuilder(freq="1min").freq == 98280

    def test_freq_unknown_alias_raises(self):
        with pytest.raises(ValueError, match="Unknown freq alias"):
            PortfolioBuilder(freq="yearly")

    def test_missing_column_raises(self, fitted_clf_reg, dummy_test_df):
        """Missing required columns raise ValueError with a clear message."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=5, col_period="nonexistent")
        with pytest.raises(ValueError, match="missing columns"):
            builder.build(dummy_test_df, clf, reg, X_te, Xr_te)

    def test_build_without_ret_column(self, fitted_clf_reg):
        """build() works when col_ret is absent (live scoring)."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        live_df = pl.DataFrame({"period": [202501] * 25 + [202502] * 25})
        builder = PortfolioBuilder(n=5)
        portfolio = builder.build(live_df, clf, reg, X_te, Xr_te)
        assert isinstance(portfolio, pl.DataFrame)
        assert {"pred_prob", "pred_ret", "score"}.issubset(set(portfolio.columns))
        assert "fwd_log_ret" not in portfolio.columns

    def test_hourly_datetime_periods(self, fitted_clf_reg, dummy_hourly_df):
        """PortfolioBuilder handles pl.Datetime hourly periods correctly."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(
            n=5, col_period="ts", col_ret="fwd_log_ret", freq="hourly"
        )
        assert builder.freq == 1512
        portfolio = builder.build(dummy_hourly_df, clf, reg, X_te, Xr_te)
        assert isinstance(portfolio, pl.DataFrame)
        # 2 hourly periods, up to 5 stocks each
        counts = portfolio.group_by("ts").agg(pl.len().alias("cnt"))["cnt"]
        assert (counts <= 5).all()
        perf = builder.performance(portfolio, dummy_hourly_df)
        assert {"port_ret", "bench_ret", "excess"}.issubset(set(perf.columns))
        m = builder.metrics(perf, portfolio)
        assert set(m.keys()) == {"ann_port", "ann_bench", "sharpe", "ic"}

    def test_namespace_export(self):
        """Models must live in sentvols.models; PortfolioBuilder in sentvols.portfolio."""
        import sentvols.models as sm
        import sentvols.portfolio as sp
        import sentvols.utils as su

        assert "SentvolsClassifier" in sm.__all__
        assert "SentvolsRegressor" in sm.__all__
        assert "PortfolioBuilder" not in sm.__all__
        assert "PortfolioBuilder" in sp.__all__
        assert "SentvolsClassifier" not in su.__all__
        assert "SentvolsRegressor" not in su.__all__
        assert "PortfolioBuilder" not in su.__all__


# ---------------------------------------------------------------------------
# Serialisation — production save / load
# ---------------------------------------------------------------------------


class TestSentvolsClassifierSerialisation:
    def test_save_before_fit_raises(self, tmp_path):
        clf = SentvolsClassifier(estimator=_BASE_CLF)
        with pytest.raises(RuntimeError, match="fit"):
            clf.save(tmp_path / "clf.joblib")

    def test_save_load_roundtrip_predictions(self, clf_data, tmp_path):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        path = tmp_path / "clf.joblib"
        clf.save(path)

        clf2 = SentvolsClassifier.load(path)
        assert isinstance(clf2, SentvolsClassifier)
        np.testing.assert_array_equal(clf.predict(X_te), clf2.predict(X_te))
        np.testing.assert_array_almost_equal(
            clf.predict_proba(X_te), clf2.predict_proba(X_te)
        )

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib

        joblib.dump("not_a_classifier", tmp_path / "bad.joblib")
        with pytest.raises(TypeError, match="Expected SentvolsClassifier"):
            SentvolsClassifier.load(tmp_path / "bad.joblib")

    def test_load_preserves_best_params(self, clf_data, tmp_path):
        X_tr, y_tr, X_va, y_va, *_ = clf_data
        clf = SentvolsClassifier(estimator=_BASE_CLF, random_state=42)
        clf.fit(X_tr, y_tr, params=_FAST_CLF_PARAMS)
        path = tmp_path / "clf.joblib"
        clf.save(path)
        clf2 = SentvolsClassifier.load(path)
        assert clf2.best_params_ == clf.best_params_


class TestSentvolsRegressorSerialisation:
    def test_save_before_fit_raises(self, tmp_path):
        reg = SentvolsRegressor(estimator=_BASE_REG)
        with pytest.raises(RuntimeError, match="fit"):
            reg.save(tmp_path / "reg.joblib")

    def test_save_load_roundtrip_predictions(self, reg_data, tmp_path):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
        reg.fit(X_tr, y_tr, params=_FAST_REG_PARAMS)
        path = tmp_path / "reg.joblib"
        reg.save(path)

        reg2 = SentvolsRegressor.load(path)
        assert isinstance(reg2, SentvolsRegressor)
        np.testing.assert_array_almost_equal(reg.predict(X_te), reg2.predict(X_te))

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib

        joblib.dump("not_a_regressor", tmp_path / "bad.joblib")
        with pytest.raises(TypeError, match="Expected SentvolsRegressor"):
            SentvolsRegressor.load(tmp_path / "bad.joblib")

    def test_load_preserves_best_params(self, reg_data, tmp_path):
        X_tr, y_tr, X_va, y_va, *_ = reg_data
        reg = SentvolsRegressor(estimator=_BASE_REG, random_state=42)
        reg.fit(X_tr, y_tr, params=_FAST_REG_PARAMS)
        path = tmp_path / "reg.joblib"
        reg.save(path)
        reg2 = SentvolsRegressor.load(path)
        assert reg2.best_params_ == reg.best_params_


# ---------------------------------------------------------------------------
# PortfolioBuilder — weighting strategies
# ---------------------------------------------------------------------------


class TestPortfolioBuilderWeighting:
    """Tests for the four built-in weighting strategies and custom callables."""

    @pytest.fixture(scope="class")
    def base_portfolio(self, fitted_clf_reg, dummy_test_df):
        """Pre-built portfolio with default (equal) weighting."""
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10, weighting="equal")
        return builder, builder.build(dummy_test_df, clf, reg, X_te, Xr_te)

    def _build(self, fitted_clf_reg, dummy_test_df, weighting):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10, weighting=weighting)
        return builder.build(dummy_test_df, clf, reg, X_te, Xr_te)

    def test_weight_column_present(self, fitted_clf_reg, dummy_test_df):
        port = self._build(fitted_clf_reg, dummy_test_df, "equal")
        assert "weight" in port.columns

    def test_equal_weights_sum_to_one_per_period(self, fitted_clf_reg, dummy_test_df):
        port = self._build(fitted_clf_reg, dummy_test_df, "equal")
        sums = (
            port.group_by("period")
            .agg(pl.col("weight").sum().alias("w_sum"))["w_sum"]
            .to_numpy()
        )
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_score_weights_sum_to_one_per_period(self, fitted_clf_reg, dummy_test_df):
        port = self._build(fitted_clf_reg, dummy_test_df, "score")
        sums = (
            port.group_by("period")
            .agg(pl.col("weight").sum().alias("w_sum"))["w_sum"]
            .to_numpy()
        )
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_softmax_weights_sum_to_one_per_period(self, fitted_clf_reg, dummy_test_df):
        port = self._build(fitted_clf_reg, dummy_test_df, "softmax")
        sums = (
            port.group_by("period")
            .agg(pl.col("weight").sum().alias("w_sum"))["w_sum"]
            .to_numpy()
        )
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_rank_weights_sum_to_one_per_period(self, fitted_clf_reg, dummy_test_df):
        port = self._build(fitted_clf_reg, dummy_test_df, "rank")
        sums = (
            port.group_by("period")
            .agg(pl.col("weight").sum().alias("w_sum"))["w_sum"]
            .to_numpy()
        )
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_custom_callable_weighting(self, fitted_clf_reg, dummy_test_df):
        """Custom weighting fn: concentrate all weight on the single top scorer."""

        def winner_takes_all(scores: np.ndarray) -> np.ndarray:
            w = np.zeros(len(scores))
            w[np.argmax(scores)] = 1.0
            return w

        port = self._build(fitted_clf_reg, dummy_test_df, winner_takes_all)
        # Each period: exactly one weight == 1, rest == 0
        for period in port["period"].unique().to_list():
            grp_w = port.filter(pl.col("period") == period)["weight"].to_numpy()
            assert grp_w.sum() == pytest.approx(1.0)
            assert (grp_w == 1.0).sum() == 1

    def test_invalid_weighting_string_raises(self):
        with pytest.raises(ValueError, match="Unknown weighting"):
            PortfolioBuilder(weighting="magic")

    def test_weighted_performance_differs_from_equal(
        self, fitted_clf_reg, dummy_test_df
    ):
        """Softmax weighting must produce different port_ret from equal weighting."""
        clf, reg, X_te, Xr_te = fitted_clf_reg

        b_eq = PortfolioBuilder(n=10, weighting="equal")
        port_eq = b_eq.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf_eq = b_eq.performance(port_eq, dummy_test_df)

        b_sm = PortfolioBuilder(n=10, weighting="softmax")
        port_sm = b_sm.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf_sm = b_sm.performance(port_sm, dummy_test_df)

        # They won't be equal for generic data
        assert not np.allclose(
            perf_eq["port_ret"].to_numpy(), perf_sm["port_ret"].to_numpy()
        )


# ---------------------------------------------------------------------------
# PortfolioBuilder — scores-only (no ML models)
# ---------------------------------------------------------------------------


class TestPortfolioBuilderScoresPath:
    def test_build_with_precomputed_scores(self, dummy_test_df):
        """build() works without trained models when scores= is provided."""
        rng = np.random.default_rng(10)
        scores = rng.standard_normal(len(dummy_test_df))
        builder = PortfolioBuilder(n=5)
        portfolio = builder.build(dummy_test_df, scores=scores)
        assert isinstance(portfolio, pl.DataFrame)
        assert "score" in portfolio.columns
        assert "weight" in portfolio.columns
        assert "pred_prob" not in portfolio.columns

    def test_build_scores_top_n_per_period(self, dummy_test_df):
        rng = np.random.default_rng(11)
        scores = rng.standard_normal(len(dummy_test_df))
        builder = PortfolioBuilder(n=3)
        portfolio = builder.build(dummy_test_df, scores=scores)
        counts = portfolio.group_by("period").agg(pl.len().alias("cnt"))["cnt"]
        assert (counts <= 3).all()

    def test_build_without_models_or_scores_raises(self, dummy_test_df):
        builder = PortfolioBuilder(n=5)
        with pytest.raises(ValueError, match="clf.*reg.*scores"):
            builder.build(dummy_test_df)

    def test_build_scores_weights_sum_to_one(self, dummy_test_df):
        rng = np.random.default_rng(12)
        scores = np.abs(rng.standard_normal(len(dummy_test_df)))  # positive scores
        builder = PortfolioBuilder(n=5, weighting="score")
        portfolio = builder.build(dummy_test_df, scores=scores)
        sums = (
            portfolio.group_by("period")
            .agg(pl.col("weight").sum())["weight"]
            .to_numpy()
        )
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_build_scores_standalone_no_dataframe_required(self):
        """Minimal usage: only col_period must exist in the DataFrame."""
        rng = np.random.default_rng(13)
        df = pl.DataFrame({"period": [1] * 10 + [2] * 10})
        scores = rng.standard_normal(20)
        builder = PortfolioBuilder(n=3)
        portfolio = builder.build(df, scores=scores)
        assert isinstance(portfolio, pl.DataFrame)
        assert len(portfolio) <= 6  # at most 3 per period × 2 periods


# ---------------------------------------------------------------------------
# PortfolioManager
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    return PortfolioManager(
        initial_cash=100_000.0,
        col_ticker="ticker",
        col_price="price",
        col_period="period",
        transaction_cost=0.001,
    )


@pytest.fixture
def simple_portfolio():
    """Two-period portfolio with 3 tickers and weights summing to 1 per period."""
    return pl.DataFrame(
        {
            "period": [
                "2025-01",
                "2025-01",
                "2025-01",
                "2025-02",
                "2025-02",
                "2025-02",
            ],
            "ticker": ["AAPL", "MSFT", "GOOG"] * 2,
            "score": [0.9, 0.7, 0.5, 0.8, 0.6, 0.4],
            "weight": [0.5, 0.3, 0.2, 0.4, 0.4, 0.2],
        }
    )


@pytest.fixture
def prices_dict():
    return {"AAPL": 150.0, "MSFT": 300.0, "GOOG": 100.0}


class TestPortfolioManager:
    def test_initial_state(self, manager):
        assert manager.cash == pytest.approx(100_000.0)
        assert manager.positions == {}
        assert manager.snapshot()["n_positions"] == 0

    def test_trade_history_empty_schema(self, manager):
        hist = manager.trade_history
        assert isinstance(hist, pl.DataFrame)
        assert len(hist) == 0
        assert set(hist.columns) == {
            "period",
            "ticker",
            "action",
            "shares",
            "price",
            "cost",
            "cash_after",
        }

    def test_snapshot_keys(self, manager):
        snap = manager.snapshot()
        assert set(snap.keys()) == {"cash", "positions", "n_positions"}

    def test_portfolio_value_no_positions(self, manager, prices_dict):
        assert manager.portfolio_value(prices_dict) == pytest.approx(100_000.0)

    def test_rebalance_returns_dataframe(self, manager, simple_portfolio, prices_dict):
        trades = manager.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        assert isinstance(trades, pl.DataFrame)

    def test_rebalance_records_trades(self, manager, simple_portfolio, prices_dict):
        manager2 = PortfolioManager(
            initial_cash=100_000.0, col_ticker="ticker", col_period="period"
        )
        trades = manager2.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        assert len(trades) > 0
        assert {"buy", "sell"}.issuperset(set(trades["action"].to_list()))

    def test_rebalance_updates_positions(self, prices_dict, simple_portfolio):
        mgr = PortfolioManager(
            initial_cash=100_000.0, col_ticker="ticker", col_period="period"
        )
        mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        pos = mgr.positions
        # All three tickers should have a non-zero position
        assert all(ticker in pos for ticker in ["AAPL", "MSFT", "GOOG"])
        assert all(v > 0 for v in pos.values())

    def test_rebalance_cash_decreases_after_buy(self, prices_dict, simple_portfolio):
        mgr = PortfolioManager(
            initial_cash=100_000.0,
            col_ticker="ticker",
            col_period="period",
            transaction_cost=0.0,
        )
        mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        assert mgr.cash < 100_000.0

    def test_portfolio_value_after_rebalance(self, prices_dict, simple_portfolio):
        mgr = PortfolioManager(
            initial_cash=100_000.0,
            col_ticker="ticker",
            col_period="period",
            transaction_cost=0.0,
        )
        mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        value = mgr.portfolio_value(prices_dict)
        # Without costs, total value should be conserved (~100k)
        assert value == pytest.approx(100_000.0, rel=1e-6)

    def test_close_all_empties_positions(self, prices_dict, simple_portfolio):
        mgr = PortfolioManager(
            initial_cash=100_000.0,
            col_ticker="ticker",
            col_period="period",
            transaction_cost=0.0,
        )
        mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        assert len(mgr.positions) > 0
        mgr.close_all(prices_dict)
        assert mgr.positions == {}

    def test_close_all_no_positions_returns_empty(self, manager, prices_dict):
        trades = manager.close_all(prices_dict)
        assert isinstance(trades, pl.DataFrame)
        assert len(trades) == 0

    def test_rebalance_missing_weight_column_raises(self, manager, prices_dict):
        bad_portfolio = pl.DataFrame({"period": ["2025-01"], "ticker": ["AAPL"]})
        with pytest.raises(ValueError, match="weight"):
            manager.rebalance(bad_portfolio, prices_dict)

    def test_rebalance_missing_ticker_column_raises(self, manager, prices_dict):
        bad_portfolio = pl.DataFrame({"period": ["2025-01"], "weight": [1.0]})
        with pytest.raises(ValueError, match="ticker"):
            manager.rebalance(bad_portfolio, prices_dict)

    def test_prices_as_polars_dataframe(self, simple_portfolio):
        mgr = PortfolioManager(
            initial_cash=100_000.0, col_ticker="ticker", col_period="period"
        )
        prices_df = pl.DataFrame(
            {"ticker": ["AAPL", "MSFT", "GOOG"], "price": [150.0, 300.0, 100.0]}
        )
        trades = mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_df
        )
        assert isinstance(trades, pl.DataFrame)
        assert len(trades) > 0

    def test_save_load_roundtrip(self, tmp_path, simple_portfolio, prices_dict):
        mgr = PortfolioManager(
            initial_cash=50_000.0,
            col_ticker="ticker",
            col_period="period",
            transaction_cost=0.002,
        )
        mgr.rebalance(
            simple_portfolio.filter(pl.col("period") == "2025-01"), prices_dict
        )
        path = tmp_path / "manager.joblib"
        mgr.save(path)

        mgr2 = PortfolioManager.load(path)
        assert isinstance(mgr2, PortfolioManager)
        assert mgr2.cash == pytest.approx(mgr.cash)
        assert mgr2.positions == pytest.approx(mgr.positions)
        assert len(mgr2.trade_history) == len(mgr.trade_history)

    def test_load_wrong_type_raises(self, tmp_path):
        import joblib

        joblib.dump("not_a_manager", tmp_path / "bad.joblib")
        with pytest.raises(TypeError, match="Expected PortfolioManager"):
            PortfolioManager.load(tmp_path / "bad.joblib")

    def test_namespace_export(self):
        """PortfolioManager must be accessible from sentvols.portfolio."""
        import sentvols.portfolio as sp

        assert "PortfolioManager" in sp.__all__

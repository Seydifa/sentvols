"""Tests for sentvols.core.models (public: sentvols.models)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentvols.models import PortfolioBuilder, SentvolsClassifier, SentvolsRegressor


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
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 30,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}

_FAST_REG_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_estimators": 30,
    "learning_rate": 0.1,
    "num_leaves": 15,
    "random_state": 42,
    "n_jobs": 1,
}


# ---------------------------------------------------------------------------
# SentvolsClassifier
# ---------------------------------------------------------------------------


class TestSentvolsClassifier:
    def test_predict_before_fit_raises(self):
        clf = SentvolsClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(np.zeros((5, 14)))

    def test_predict_proba_before_fit_raises(self):
        clf = SentvolsClassifier()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict_proba(np.zeros((5, 14)))

    def test_feature_importances_before_fit_raises(self):
        clf = SentvolsClassifier()
        with pytest.raises(RuntimeError):
            _ = clf.feature_importances_

    def test_fit_without_params_raises(self):
        clf = SentvolsClassifier()
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.integers(0, 2, 20)
        with pytest.raises(RuntimeError, match="optimize"):
            clf.fit(X, y, X, y)

    def test_fit_predict_with_explicit_params(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(random_state=42)
        clf.fit(X_tr, y_tr, X_va, y_va, params=_FAST_CLF_PARAMS)
        preds = clf.predict(X_te)
        assert preds.shape == (len(y_te),)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(random_state=42)
        clf.fit(X_tr, y_tr, X_va, y_va, params=_FAST_CLF_PARAMS)
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(y_te), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_evaluate_keys(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(random_state=42)
        clf.fit(X_tr, y_tr, X_va, y_va, params=_FAST_CLF_PARAMS)
        metrics = clf.evaluate(X_te, y_te)
        assert set(metrics.keys()) == {"f1", "precision", "recall"}
        for v in metrics.values():
            assert 0.0 <= v <= 1.0

    def test_feature_importances_shape(self, clf_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
        clf = SentvolsClassifier(random_state=42)
        clf.fit(X_tr, y_tr, X_va, y_va, params=_FAST_CLF_PARAMS)
        imp = clf.feature_importances_
        assert imp.shape == (X_tr.shape[1],)
        assert (imp >= 0).all()


# ---------------------------------------------------------------------------
# SentvolsRegressor
# ---------------------------------------------------------------------------


class TestSentvolsRegressor:
    def test_predict_before_fit_raises(self):
        reg = SentvolsRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.zeros((5, 14)))

    def test_feature_importances_before_fit_raises(self):
        reg = SentvolsRegressor()
        with pytest.raises(RuntimeError):
            _ = reg.feature_importances_

    def test_fit_without_params_raises(self):
        reg = SentvolsRegressor()
        rng = np.random.default_rng(0)
        X, y = rng.standard_normal((20, 4)), rng.standard_normal(20)
        with pytest.raises(RuntimeError, match="optimize"):
            reg.fit(X, y, X, y)

    def test_fit_predict_with_explicit_params(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(random_state=42)
        reg.fit(X_tr, y_tr, X_va, y_va, params=_FAST_REG_PARAMS)
        preds = reg.predict(X_te)
        assert preds.shape == (len(y_te),)
        assert preds.dtype == np.float64

    def test_evaluate_keys(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(random_state=42)
        reg.fit(X_tr, y_tr, X_va, y_va, params=_FAST_REG_PARAMS)
        metrics = reg.evaluate(X_te, y_te)
        assert set(metrics.keys()) == {"rmse", "mae", "r2"}
        assert metrics["rmse"] >= 0
        assert metrics["mae"] >= 0

    def test_feature_importances_shape(self, reg_data):
        X_tr, y_tr, X_va, y_va, X_te, y_te = reg_data
        reg = SentvolsRegressor(random_state=42)
        reg.fit(X_tr, y_tr, X_va, y_va, params=_FAST_REG_PARAMS)
        imp = reg.feature_importances_
        assert imp.shape == (X_tr.shape[1],)


# ---------------------------------------------------------------------------
# PortfolioBuilder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted_clf_reg(clf_data, reg_data):
    X_tr, y_tr, X_va, y_va, X_te, y_te = clf_data
    Xr_tr, yr_tr, Xr_va, yr_va, Xr_te, yr_te = reg_data

    clf = SentvolsClassifier(random_state=42)
    clf.fit(X_tr, y_tr, X_va, y_va, params=_FAST_CLF_PARAMS)

    reg = SentvolsRegressor(random_state=42)
    reg.fit(Xr_tr, yr_tr, Xr_va, yr_va, params=_FAST_REG_PARAMS)

    return clf, reg, X_te, Xr_te


@pytest.fixture(scope="module")
def dummy_test_df():
    """Minimal df_model_test with 50 rows matching X_te/Xr_te from clf_data/reg_data."""
    rng = np.random.default_rng(2)
    periods = np.repeat([201901, 201902], 25)
    tickers = [f"T{i:03d}" for i in range(25)] * 2
    fwd = rng.standard_normal(50) * 0.02
    return pd.DataFrame({"period": periods, "ticker": tickers, "fwd_log_ret": fwd})


class TestPortfolioBuilder:
    def test_build_returns_dataframe(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        assert isinstance(portfolio, pd.DataFrame)

    def test_build_adds_expected_columns(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        assert {"pred_prob", "pred_ret", "score"}.issubset(portfolio.columns)

    def test_build_top_n_per_period(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        n = 5
        builder = PortfolioBuilder(n=n)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        counts = portfolio.groupby("period").size()
        assert (counts <= n).all()

    def test_performance_returns_expected_columns(self, fitted_clf_reg, dummy_test_df):
        clf, reg, X_te, Xr_te = fitted_clf_reg
        builder = PortfolioBuilder(n=10)
        portfolio = builder.build(dummy_test_df, clf, reg, X_te, Xr_te)
        perf = builder.performance(portfolio, dummy_test_df)
        assert {"port_ret", "bench_ret", "excess", "cum_port", "cum_bench"}.issubset(
            perf.columns
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

    def test_namespace_export(self):
        """Models must live in sentvols.models, not sentvols.utils."""
        import sentvols.models as sm
        import sentvols.utils as su

        assert "SentvolsClassifier" in sm.__all__
        assert "SentvolsRegressor" in sm.__all__
        assert "PortfolioBuilder" in sm.__all__
        assert "SentvolsClassifier" not in su.__all__
        assert "SentvolsRegressor" not in su.__all__
        assert "PortfolioBuilder" not in su.__all__

from __future__ import annotations

import pathlib
from typing import Any, Callable

import joblib
import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .exports import registration


def _default_clf_score(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def _default_reg_score(y_true, y_pred) -> float:
    return -float(np.sqrt(mean_squared_error(y_true, y_pred)))


@registration(module="models")
class SentvolsClassifier:
    """Framework-agnostic classifier wrapper with optional Optuna HPO.

    Parameters
    ----------
    estimator :
        Any sklearn-compatible classifier *instance*
        (e.g. ``RandomForestClassifier()``, ``LGBMClassifier()``,
        ``CatBoostClassifier()``, ``XGBClassifier()``).
    random_state : int
        Seed used by the Optuna sampler.
    """

    def __init__(self, estimator: Any, random_state: int = 42) -> None:
        self.estimator = estimator
        self.random_state = random_state
        self._model: Any | None = None
        self.best_params_: dict = {}

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    def optimize(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        search_space: Callable,
        n_trials: int = 50,
        score_fn: Callable | None = None,
        fit_params: dict | None = None,
    ) -> dict:
        """Run Optuna HPO over the estimator's hyperparameters.

        Parameters
        ----------
        search_space :
            Callable ``(trial) -> dict`` whose return value is forwarded to
            ``estimator.set_params(**kwargs)`` for every trial.
        score_fn :
            ``(y_true, y_pred) -> float`` metric to *maximise*.
            Defaults to macro-F1.
        fit_params :
            Extra kwargs forwarded verbatim to ``estimator.fit()``.
        """
        _score = score_fn if score_fn is not None else _default_clf_score
        _fit_params = fit_params or {}

        def _objective(trial):
            params = search_space(trial)
            m = clone(self.estimator)
            m.set_params(**params)
            m.fit(X_train, y_train, **_fit_params)
            return _score(y_val, m.predict(X_val))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        return {"best_score": study.best_value, "best_params": self.best_params_}

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        params: dict | None = None,
        fit_params: dict | None = None,
    ) -> "SentvolsClassifier":
        """Train a fresh clone of the estimator.

        Parameters
        ----------
        params :
            Hyperparameters passed to ``estimator.set_params(**params)``.
            If omitted, ``best_params_`` from a prior ``optimize()`` call is used.
        fit_params :
            Extra kwargs forwarded verbatim to ``estimator.fit()``.
        """
        p = params if params is not None else self.best_params_
        if not p:
            raise RuntimeError("Call optimize() first or pass params explicitly.")
        self._model = clone(self.estimator)
        self._model.set_params(**p)
        self._model.fit(X_train, y_train, **(fit_params or {}))
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the fitted wrapper to *path* (joblib format)."""
        if self._model is None:
            raise RuntimeError("Call fit() before save().")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "SentvolsClassifier":
        """Load a previously saved wrapper from *path*."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        if not hasattr(self._model, "predict_proba"):
            raise AttributeError(
                f"{type(self._model).__name__} does not support predict_proba()."
            )
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation & introspection
    # ------------------------------------------------------------------

    def evaluate(self, X, y) -> dict:
        preds = self.predict(X)
        return {
            "f1": float(f1_score(y, preds, zero_division=0)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        if hasattr(self._model, "feature_importances_"):
            return np.asarray(self._model.feature_importances_)
        if hasattr(self._model, "coef_"):
            return np.abs(self._model.coef_).ravel()
        raise AttributeError(
            f"{type(self._model).__name__} does not expose feature importances "
            "via 'feature_importances_' or 'coef_'."
        )


@registration(module="models")
class SentvolsRegressor:
    """Framework-agnostic regressor wrapper with optional Optuna HPO.

    Parameters
    ----------
    estimator :
        Any sklearn-compatible regressor *instance*.
    random_state : int
        Seed used by the Optuna sampler.
    """

    def __init__(self, estimator: Any, random_state: int = 42) -> None:
        self.estimator = estimator
        self.random_state = random_state
        self._model: Any | None = None
        self.best_params_: dict = {}

    # ------------------------------------------------------------------
    # Optuna tuning
    # ------------------------------------------------------------------

    def optimize(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        search_space: Callable,
        n_trials: int = 50,
        score_fn: Callable | None = None,
        fit_params: dict | None = None,
    ) -> dict:
        """Run Optuna HPO over the estimator's hyperparameters.

        Parameters
        ----------
        search_space :
            Callable ``(trial) -> dict`` whose return value is forwarded to
            ``estimator.set_params(**kwargs)`` for every trial.
        score_fn :
            ``(y_true, y_pred) -> float`` to *maximise*.
            Defaults to ``-RMSE``.
        fit_params :
            Extra kwargs forwarded verbatim to ``estimator.fit()``.
        """
        _score = score_fn if score_fn is not None else _default_reg_score
        _fit_params = fit_params or {}

        def _objective(trial):
            params = search_space(trial)
            m = clone(self.estimator)
            m.set_params(**params)
            m.fit(X_train, y_train, **_fit_params)
            return _score(y_val, m.predict(X_val))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        return {"best_score": study.best_value, "best_params": self.best_params_}

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train,
        y_train,
        params: dict | None = None,
        fit_params: dict | None = None,
    ) -> "SentvolsRegressor":
        """Train a fresh clone of the estimator.

        Parameters
        ----------
        params :
            Hyperparameters passed to ``estimator.set_params(**params)``.
            If omitted, ``best_params_`` from a prior ``optimize()`` call is used.
        fit_params :
            Extra kwargs forwarded verbatim to ``estimator.fit()``.
        """
        p = params if params is not None else self.best_params_
        if not p:
            raise RuntimeError("Call optimize() first or pass params explicitly.")
        self._model = clone(self.estimator)
        self._model.set_params(**p)
        self._model.fit(X_train, y_train, **(fit_params or {}))
        return self

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str | pathlib.Path) -> None:
        """Persist the fitted wrapper to *path* (joblib format)."""
        if self._model is None:
            raise RuntimeError("Call fit() before save().")
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "SentvolsRegressor":
        """Load a previously saved wrapper from *path*."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def evaluate(self, X, y) -> dict:
        y_hat = self.predict(X)
        return {
            "rmse": float(np.sqrt(mean_squared_error(y, y_hat))),
            "mae": float(mean_absolute_error(y, y_hat)),
            "r2": float(r2_score(y, y_hat)),
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        if hasattr(self._model, "feature_importances_"):
            return np.asarray(self._model.feature_importances_)
        if hasattr(self._model, "coef_"):
            return np.abs(self._model.coef_).ravel()
        raise AttributeError(
            f"{type(self._model).__name__} does not expose feature importances "
            "via 'feature_importances_' or 'coef_'."
        )

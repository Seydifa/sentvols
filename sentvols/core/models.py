from __future__ import annotations

import numpy as np
import lightgbm as lgb
import optuna
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from .exports import registration


@registration(module="models")
class SentvolsClassifier:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._model: lgb.LGBMClassifier | None = None
        self.best_params_: dict = {}

    def _objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        return float(f1_score(y_val, m.predict(X_val)))

    def optimize(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials: int = 50,
    ) -> dict:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        self.best_params_ = {
            **study.best_params,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "class_weight": "balanced",
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        return {"best_f1": study.best_value, "best_params": self.best_params_}

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        params: dict | None = None,
    ) -> "SentvolsClassifier":
        p = params if params is not None else self.best_params_
        if not p:
            raise RuntimeError("Call optimize() first or pass params explicitly.")
        self._model = lgb.LGBMClassifier(**p)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        return self

    def predict(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self._model.predict_proba(X)

    def evaluate(self, X, y) -> dict:
        preds = self.predict(X)
        return {
            "f1": float(f1_score(y, preds)),
            "precision": float(precision_score(y, preds)),
            "recall": float(recall_score(y, preds)),
        }

    @property
    def feature_importances_(self) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted yet.")
        return self._model.feature_importances_


@registration(module="models")
class SentvolsRegressor:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._model: lgb.LGBMRegressor | None = None
        self.best_params_: dict = {}

    def _objective(self, trial, X_train, y_train, X_val, y_val) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        y_hat = m.predict(X_val)
        return -float(np.sqrt(mean_squared_error(y_val, y_hat)))

    def optimize(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        n_trials: int = 50,
    ) -> dict:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        self.best_params_ = {
            **study.best_params,
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "random_state": self.random_state,
            "n_jobs": -1,
        }
        return {"best_rmse": -study.best_value, "best_params": self.best_params_}

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        params: dict | None = None,
    ) -> "SentvolsRegressor":
        p = params if params is not None else self.best_params_
        if not p:
            raise RuntimeError("Call optimize() first or pass params explicitly.")
        self._model = lgb.LGBMRegressor(**p)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        return self

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
        return self._model.feature_importances_


@registration(module="models")
class PortfolioBuilder:
    def __init__(self, n: int = 50) -> None:
        self.n = n

    def build(
        self,
        df_test,
        clf: SentvolsClassifier,
        reg: SentvolsRegressor,
        X_test_clf_sc,
        X_test_reg_sc,
    ):
        import pandas as pd

        df = df_test.copy()
        pred_prob = clf.predict_proba(X_test_clf_sc)[:, 1]
        pred_ret = reg.predict(X_test_reg_sc)
        df["pred_prob"] = pred_prob
        df["pred_ret"] = pred_ret
        df["score"] = pred_prob * pred_ret
        portfolio = pd.concat(
            [grp.nlargest(self.n, "score") for _, grp in df.groupby("period")],
            ignore_index=True,
        )
        return portfolio

    def performance(self, portfolio, df_all_test):
        perf = (
            portfolio.groupby("period")["fwd_log_ret"]
            .mean()
            .reset_index()
            .rename(columns={"fwd_log_ret": "port_ret"})
        )
        bench = (
            df_all_test.groupby("period")["fwd_log_ret"]
            .mean()
            .reset_index()
            .rename(columns={"fwd_log_ret": "bench_ret"})
        )
        perf = perf.merge(bench, on="period")
        perf["excess"] = perf["port_ret"] - perf["bench_ret"]
        perf["cum_port"] = (1 + perf["port_ret"]).cumprod() - 1
        perf["cum_bench"] = (1 + perf["bench_ret"]).cumprod() - 1
        return perf

    def metrics(self, perf, portfolio) -> dict:
        ann_port = float(perf["port_ret"].mean() * 12)
        ann_bench = float(perf["bench_ret"].mean() * 12)
        sharpe = float(
            perf["port_ret"].mean() / (perf["port_ret"].std() + 1e-9) * np.sqrt(12)
        )
        ic = float(
            np.corrcoef(
                portfolio["score"].values,
                portfolio["fwd_log_ret"].values,
            )[0, 1]
        )
        return {
            "ann_port": ann_port,
            "ann_bench": ann_bench,
            "sharpe": sharpe,
            "ic": ic,
        }

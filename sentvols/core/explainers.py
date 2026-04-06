from __future__ import annotations

import numpy as np
from scipy import stats

from .exports import registration


@registration(module="explainers")
def test_alpha(perf) -> dict:
    t_stat, p_val = stats.ttest_1samp(perf["excess"], popmean=0, alternative="greater")
    mean_excess = float(perf["excess"].mean())
    return {
        "mean_excess_monthly": mean_excess,
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


@registration(module="explainers")
def test_sentiment_correlation(df_pd) -> dict:
    df_clean = df_pd.dropna(subset=["sent_sum_mean", "sent_mean_avg", "target_return"])
    r_sum, p_sum = stats.pearsonr(
        df_clean["sent_sum_mean"].values, df_clean["target_return"].values
    )
    r_avg, p_avg = stats.pearsonr(
        df_clean["sent_mean_avg"].values, df_clean["target_return"].values
    )
    return {
        "n_obs": len(df_clean),
        "r_sum": float(r_sum),
        "p_sum": float(p_sum),
        "r_avg": float(r_avg),
        "p_avg": float(p_avg),
        "significant": bool(p_sum < 0.05 or p_avg < 0.05),
    }


@registration(module="explainers")
def test_classifier_permutation(
    clf,
    X_test_sc,
    y_cls_test,
    n_permu: int = 10_000,
    seed: int = 42,
) -> dict:
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    baseline_f1 = float(f1_score(y_cls_test, clf.predict(X_test_sc)))
    perm_f1s = np.array(
        [
            f1_score(rng.permutation(y_cls_test), clf.predict(X_test_sc))
            for _ in range(n_permu)
        ]
    )
    p_perm = float((perm_f1s >= baseline_f1).mean())
    return {
        "baseline_f1": baseline_f1,
        "perm_f1_mean": float(perm_f1s.mean()),
        "p_value": p_perm,
        "perm_f1s": perm_f1s,
        "significant": bool(p_perm < 0.05),
    }


@registration(module="explainers")
def test_diebold_mariano(reg, X_test_reg_sc, y_reg_test) -> dict:
    y_pred = reg.predict(X_test_reg_sc)
    y_naive = np.zeros_like(y_reg_test)
    e_lgbm = (y_reg_test - y_pred) ** 2
    e_naive = (y_reg_test - y_naive) ** 2
    dm_diff = e_naive - e_lgbm
    t_dm, p_dm = stats.ttest_1samp(dm_diff, popmean=0, alternative="greater")
    return {
        "mse_naive": float(e_naive.mean()),
        "mse_lgbm": float(e_lgbm.mean()),
        "t_statistic": float(t_dm),
        "p_value": float(p_dm),
        "significant": bool(p_dm < 0.05),
    }


@registration(module="explainers")
def run_hypothesis_tests(
    perf,
    df_pd,
    clf,
    reg,
    X_test_sc,
    X_test_reg_sc,
    y_cls_test,
    y_reg_test,
    n_permu: int = 10_000,
    seed: int = 42,
) -> dict:
    alpha_result = test_alpha(perf)
    corr_result = test_sentiment_correlation(df_pd)
    perm_result = test_classifier_permutation(
        clf, X_test_sc, y_cls_test, n_permu=n_permu, seed=seed
    )
    dm_result = test_diebold_mariano(reg, X_test_reg_sc, y_reg_test)
    return {
        "alpha": alpha_result,
        "correlation": corr_result,
        "permutation": perm_result,
        "diebold_mariano": dm_result,
    }

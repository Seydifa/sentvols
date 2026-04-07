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


# ---------------------------------------------------------------------------
# OLS news-impact test (market-adjusted framework)
# ---------------------------------------------------------------------------


@registration(module="explainers")
def test_ols_sentiment_impact(
    df,
    feature_cols: list[str],
    col_dep: str = "abnormal_ret",
    intercept: bool = False,
    alpha_level: float = 0.05,
) -> dict:
    """OLS test: do sentiment features significantly predict abnormal returns?

    The market-model residual (abnormal return) is used as the dependent
    variable so that the coefficients measure the **pure news effect** after
    removing systematic market risk.  This avoids the classic confound where
    sentiment is correlated with the market cycle.

    Model (no-intercept variant, ``intercept=False``):

    .. math::

        AR_{i,t} = \\sum_j \\hat{\\beta}_j X_{j,i,t} + \\varepsilon_{i,t}

    Under the null :math:`H_0: \\beta_j = 0` for all *j* — i.e. news
    sentiment has *no* marginal explanatory power for abnormal returns.

    Parameters
    ----------
    df : pl.DataFrame or pd.DataFrame
        Feature panel.  Must contain ``feature_cols`` and ``col_dep``.
        NaN / null rows are dropped before fitting.
    feature_cols : list[str]
        Sentiment feature columns to include as regressors.  Typically a
        subset of :data:`sentvols.features.SENTIMENT_FEATURE_COLS`.
    col_dep : str
        Dependent variable column.  Default ``"abnormal_ret"`` (output of
        :func:`sentvols.features.add_abnormal_returns`).  Can be set to
        ``"ret"`` for a raw-return regression (market effects NOT removed).
    intercept : bool
        If ``True``, prepend a column of ones to the design matrix.
        Default ``False`` — assumes market adjustment has centred the
        dependent variable around zero.
    alpha_level : float
        Significance level for the ``significant`` flag per coefficient.

    Returns
    -------
    dict
        ``n_obs``       — number of observations used.
        ``feature_cols`` — list of regressor names (matches coefficient order).
        ``coefs``       — OLS coefficient estimates, shape (k,).
        ``se``          — heteroskedasticity-robust (HC3) standard errors, shape (k,).
        ``t_stats``     — t-statistics, shape (k,).
        ``p_values``    — two-sided p-values, shape (k,).
        ``significant`` — bool array: p_value < alpha_level, shape (k,).
        ``r_squared``   — coefficient of determination.
        ``f_stat``      — F-statistic for joint significance (all β = 0).
        ``f_pvalue``    — p-value for F-test.
        ``summary``     — human-readable dict {feature: {coef, se, t, p, sig}}.
    """
    import polars as pl

    # --- data prep ---
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    cols_needed = feature_cols + [col_dep]
    pdf = pdf[cols_needed].dropna()
    n = len(pdf)
    if n < len(feature_cols) + 2:
        raise ValueError(
            f"test_ols_sentiment_impact: only {n} complete observations for "
            f"{len(feature_cols)} regressors — too few to fit."
        )

    y = pdf[col_dep].values.astype(float)
    X_raw = pdf[feature_cols].values.astype(float)
    X = np.column_stack([np.ones(n), X_raw]) if intercept else X_raw
    k = X.shape[1]
    reg_names = (["_intercept"] + feature_cols) if intercept else feature_cols

    # --- OLS via normal equations ---
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        coefs = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    y_hat = X @ coefs
    residuals = y - y_hat
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    dof = n - k

    # --- HC3 robust standard errors (MacKinnon & White 1985) ---
    h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(XtX), X)  # hat values
    e_hc3 = residuals / (1 - h).clip(min=1e-10)  # inflation factor
    Omega = np.diag(e_hc3**2)
    meat = X.T @ Omega @ X
    bread = np.linalg.pinv(XtX)
    cov_hc3 = bread @ meat @ bread
    se = np.sqrt(np.diag(cov_hc3).clip(min=0))

    t_stats = coefs / np.where(se > 0, se, np.nan)
    p_values = np.array(
        [
            float(2 * stats.t.sf(abs(t), df=dof)) if not np.isnan(t) else 1.0
            for t in t_stats
        ]
    )
    significant = p_values < alpha_level

    # --- F-test for joint significance ---
    if dof > 0:
        f_stat = (
            float((r_squared / k) / ((1 - r_squared) / dof))
            if r_squared < 1
            else float("inf")
        )
        f_pvalue = float(stats.f.sf(f_stat, dfn=k, dfd=dof))
    else:
        f_stat, f_pvalue = float("nan"), float("nan")

    summary = {
        name: {
            "coef": float(coefs[i]),
            "se": float(se[i]),
            "t": float(t_stats[i]) if not np.isnan(t_stats[i]) else None,
            "p": float(p_values[i]),
            "significant": bool(significant[i]),
        }
        for i, name in enumerate(reg_names)
    }

    return {
        "n_obs": n,
        "feature_cols": reg_names,
        "coefs": coefs,
        "se": se,
        "t_stats": t_stats,
        "p_values": p_values,
        "significant": significant,
        "r_squared": r_squared,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "summary": summary,
    }

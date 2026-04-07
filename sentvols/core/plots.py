from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Patch
from scipy import stats as scipy_stats

from .exports import registration


@registration(module="plots")
def plot_sentiment_distribution(df_news_pd, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Analyse de Sentiment Financier (VADER + Loughran-McDonald)",
        fontsize=13,
        fontweight="bold",
    )

    sns.countplot(
        data=df_news_pd,
        x="sentiment_label",
        order=["positif", "neutre", "négatif"],
        ax=axes[0],
        palette={"positif": "#2ecc71", "neutre": "#95a5a6", "négatif": "#e74c3c"},
    )
    axes[0].set_title("Distribution des sentiments")
    axes[0].set_xlabel("Sentiment")
    axes[0].set_ylabel("Nombre d'articles")

    sns.histplot(
        data=df_news_pd,
        x="sentiment_score",
        bins=40,
        kde=True,
        ax=axes[1],
        color="#0055A4",
    )
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1, label="Neutralité (0)")
    axes[1].set_title("Distribution des scores de sentiment financier")
    axes[1].set_xlabel("Score VADER+LM (−1 = très négatif → +1 = très positif)")
    axes[1].set_ylabel("Densité")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_descriptive_dashboard(df_plot, save_path: str | None = None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Sentivol — Distributions des Variables Clés", fontsize=14, fontweight="bold"
    )

    axes[0, 0].hist(
        df_plot["sentiment_sum"], bins=80, color="#0055A4", edgecolor="white"
    )
    axes[0, 0].axvline(
        0, color="red", linestyle="--", linewidth=1.2, label="Neutralité"
    )
    axes[0, 0].axvline(
        df_plot["sentiment_sum"].mean(),
        color="orange",
        linestyle="-.",
        label=f"Moyenne : {df_plot['sentiment_sum'].mean():.3f}",
    )
    axes[0, 0].set_title("Score de Sentiment Cumulé (∑s)")
    axes[0, 0].set_xlabel("∑ scores VADER+LM par jour")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(
        df_plot["sentiment_mean"], bins=80, color="#27ae60", edgecolor="white"
    )
    axes[0, 1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0, 1].axvline(
        df_plot["sentiment_mean"].mean(),
        color="orange",
        linestyle="-.",
        label=f"Moyenne : {df_plot['sentiment_mean'].mean():.3f}",
    )
    axes[0, 1].set_title("Score de Sentiment Moyen (s̄)")
    axes[0, 1].set_xlabel("Score VADER+LM moyen par article")
    axes[0, 1].legend(fontsize=8)

    lr_clean = df_plot["log_return"].dropna()
    axes[0, 2].hist(
        lr_clean,
        bins=100,
        color="#DCE7F7",
        edgecolor="#0055A4",
        density=True,
        label="Empirique",
    )
    xmin = lr_clean.quantile(0.001)
    xmax = lr_clean.quantile(0.999)
    x_norm = np.linspace(xmin, xmax, 300)
    axes[0, 2].plot(
        x_norm,
        scipy_stats.norm.pdf(x_norm, lr_clean.mean(), lr_clean.std()),
        "r-",
        linewidth=2,
        label="Normale théorique",
    )
    axes[0, 2].set_xlim(xmin, xmax)
    axes[0, 2].set_title("Distribution des Log-Rendements")
    axes[0, 2].set_xlabel("log(P_t / P_{t-1})")
    axes[0, 2].legend(fontsize=8)

    axes[1, 0].hist(
        df_plot["n_articles"].clip(upper=30),
        bins=30,
        color="#e67e22",
        edgecolor="white",
    )
    axes[1, 0].set_title("Nombre d'Articles par (Date × Ticker)")
    axes[1, 0].set_xlabel("Nombre d'articles (tronqué à 30)")
    axes[1, 0].set_ylabel("Fréquence")

    df_melt = df_plot[["pct_positif", "pct_negatif"]].melt(
        var_name="Type", value_name="Proportion"
    )
    df_melt["Type"] = df_melt["Type"].map(
        {"pct_positif": "Positifs", "pct_negatif": "Négatifs"}
    )
    sns.violinplot(
        data=df_melt,
        x="Type",
        y="Proportion",
        palette={"Positifs": "#2ecc71", "Négatifs": "#e74c3c"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("Répartition Articles Positifs vs Négatifs")
    axes[1, 1].set_ylabel("Proportion journalière")

    (osm, osr), (slope, intercept, r) = scipy_stats.probplot(lr_clean, dist="norm")
    axes[1, 2].scatter(osm, osr, s=2, alpha=0.3, color="#0055A4", label="Données")
    axes[1, 2].plot(
        osm,
        slope * np.array(osm) + intercept,
        "r-",
        linewidth=1.5,
        label="Droite normale",
    )
    axes[1, 2].set_title(f"QQ-Plot des Log-Rendements  (r={r:.3f})")
    axes[1, 2].set_xlabel("Quantiles théoriques")
    axes[1, 2].set_ylabel("Quantiles empiriques")
    axes[1, 2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_correlation_analysis(df_corr_pd, save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Sentivol — Distribution des Corrélations Intra-Titre",
        fontsize=13,
        fontweight="bold",
    )

    for ax, col, label, color in [
        (axes[0], "corr_sum", "Sentiment Cumulé (Sum s) vs Rendement", "#0055A4"),
        (axes[1], "corr_mean", "Sentiment Moyen (s̄) vs Rendement", "#27ae60"),
    ]:
        data = df_corr_pd[col].dropna()
        med = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        pct_pos = (data > 0).mean()
        ax.hist(data, bins=50, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="r = 0")
        ax.axvline(med, color="orange", linewidth=1.5, label=f"Médiane : {med:+.3f}")
        ax.axvline(
            q1, color="gray", linewidth=1.0, linestyle=":", label=f"Q1 : {q1:+.3f}"
        )
        ax.axvline(
            q3, color="gray", linewidth=1.0, linestyle=":", label=f"Q3 : {q3:+.3f}"
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Corrélation de Pearson")
        ax.set_ylabel("Nombre de tickers")
        ax.legend(fontsize=8)
        ax.text(
            0.97,
            0.93,
            f"{pct_pos:.1%} des tickers\nont r > 0",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.9),
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_intrastock_scatter(df_corr_pd, save_path: str | None = None):
    df_c = df_corr_pd.dropna(subset=["corr_sum"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        "Sentivol — Fiabilité du Signal vs Taille de l'Échantillon",
        fontsize=13,
        fontweight="bold",
    )

    for ax, col, label, color in [
        (axes[0], "corr_sum", "Corrél. ∑s", "#0055A4"),
        (axes[1], "corr_mean", "Corrél. s̄", "#27ae60"),
    ]:
        sc = ax.scatter(
            df_c["n_points"],
            df_c[col],
            alpha=0.35,
            s=10,
            c=df_c[col],
            cmap="RdYlGn",
            vmin=-0.4,
            vmax=0.4,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        z = np.polyfit(df_c["n_points"].fillna(0), df_c[col].fillna(0), 1)
        xfit = np.linspace(df_c["n_points"].min(), df_c["n_points"].max(), 200)
        ax.plot(
            xfit,
            np.polyval(z, xfit),
            "r-",
            linewidth=1.5,
            label=f"Tendance (pente {z[0]:+.5f})",
        )
        plt.colorbar(sc, ax=ax, label="Corrélation")
        ax.set_title(f"{label} vs Nbre d'observations")
        ax.set_xlabel("Nombre de jours de cotation")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_portfolio_performance(
    perf,
    ann_port: float,
    ann_bench: float,
    sharpe: float,
    ic: float,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    step = max(1, len(perf) // 6)

    ax = axes[0]
    ax.plot(
        perf["period"].astype(str),
        perf["cum_port"],
        marker="o",
        color="#0055A4",
        linewidth=2,
        markersize=5,
        label="Portefeuille Top-50",
    )
    ax.plot(
        perf["period"].astype(str),
        perf["cum_bench"],
        marker="s",
        color="#e74c3c",
        linewidth=2,
        markersize=5,
        linestyle="--",
        label="Benchmark",
    )
    ax.fill_between(
        range(len(perf)),
        perf["cum_port"].values,
        perf["cum_bench"].values,
        where=perf["cum_port"].values >= perf["cum_bench"].values,
        alpha=0.15,
        color="#27ae60",
        label="Alpha positif",
    )
    ax.fill_between(
        range(len(perf)),
        perf["cum_port"].values,
        perf["cum_bench"].values,
        where=perf["cum_port"].values < perf["cum_bench"].values,
        alpha=0.15,
        color="#e74c3c",
        label="Alpha négatif",
    )
    ax.set_title("Rendements Cumulés (période test)", fontweight="bold")
    ax.set_xlabel("Période")
    ax.set_ylabel("Rendement cumulé")
    ax.set_xticks(range(0, len(perf), step))
    ax.set_xticklabels(
        [str(p) for p in perf["period"].values[::step]], rotation=45, ha="right"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    x_pos = np.arange(len(perf))
    width = 0.35
    ax.bar(
        x_pos - width / 2,
        perf["port_ret"] * 100,
        width,
        color="#0055A4",
        alpha=0.8,
        label="Portefeuille Top-50",
    )
    ax.bar(
        x_pos + width / 2,
        perf["bench_ret"] * 100,
        width,
        color="#e74c3c",
        alpha=0.8,
        label="Benchmark",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Rendements Mensuels (%)", fontweight="bold")
    ax.set_xlabel("Période")
    ax.set_ylabel("Log-rendement (%)")
    ax.set_xticks(x_pos[::step])
    ax.set_xticklabels(
        [str(p) for p in perf["period"].values[::step]], rotation=45, ha="right"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    metrics_txt = (
        f"Rdt. port. annualisé : {ann_port:+.2%}  |  "
        f"Benchmark : {ann_bench:+.2%}  |  "
        f"Sharpe : {sharpe:.3f}  |  IC : {ic:.4f}"
    )
    fig.text(
        0.5,
        -0.04,
        metrics_txt,
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor="#0055A4"),
    )
    fig.suptitle(
        "Portefeuille Top-50 — LightGBM Regressor + Classifier (période test 2019–2020)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_feature_importance(
    clf, reg, feature_cols: list[str], save_path: str | None = None
):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, model, title in [
        (axes[0], reg, "LGBM Regressor"),
        (axes[1], clf, "LGBM Classifier"),
    ]:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1]
        sorted_feats = [feature_cols[i] for i in idx]
        sorted_imps = importances[idx]
        colors = ["#27ae60" if "sent" in f else "#0055A4" for f in sorted_feats]
        bars = ax.barh(
            range(len(sorted_feats)), sorted_imps[::-1], color=colors[::-1], alpha=0.85
        )
        ax.set_yticks(range(len(sorted_feats)))
        ax.set_yticklabels(sorted_feats[::-1], fontsize=9)
        ax.set_xlabel("Importance (gain moyen)", fontsize=9)
        ax.set_title(title, fontweight="bold")
        ax.grid(alpha=0.3, axis="x")
        for bar, val in zip(bars, sorted_imps[::-1]):
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}",
                va="center",
                fontsize=7,
            )

    legend_handles = [
        Patch(facecolor="#27ae60", label="Feature Sentiment (VADER/LM)"),
        Patch(facecolor="#0055A4", label="Feature Financière (ret/vol)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=9,
    )
    fig.suptitle(
        "Feature Importance (XAI — Gain) : LGBM Regressor & LGBM Classifier",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_hypothesis_permutation(
    perm_f1s,
    baseline_f1: float,
    p_perm: float,
    save_path: str | None = None,
):
    fig, ax = plt.subplots(figsize=(8, 4))
    n_permu = len(perm_f1s)
    ax.hist(
        perm_f1s,
        bins=40,
        color="#0055A4",
        alpha=0.7,
        edgecolor="white",
        label=f"Distribution permutations (n={n_permu})",
    )
    ax.axvline(
        baseline_f1,
        color="#e74c3c",
        linewidth=2.5,
        label=f"F1 observé = {baseline_f1:.4f}",
    )
    ax.axvline(
        np.percentile(perm_f1s, 95),
        color="#27ae60",
        linewidth=1.5,
        linestyle="--",
        label="Seuil 5% (p = 0.05)",
    )
    ax.set_xlabel("Score F1")
    ax.set_ylabel("Fréquences")
    ax.set_title(
        f"Test de Permutation — F1 LGBM vs Distribution Nulle\np-valeur = {p_perm:.4f}",
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_score_explanation(
    explanation: dict, top_n: int = 15, save_path: str | None = None
):
    """Visualise the word- and phrase-level contributions for a single text.

    Parameters
    ----------
    explanation : dict
        Output of ``FinancialVADERAnnotator.explain(text)``.
    top_n : int, default 15
        Maximum number of word/phrase tokens to display.
    save_path : str | None
        If provided, the figure is saved to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    word_hits = explanation.get("word_hits", [])
    phrase_hits = explanation.get("phrase_hits", [])
    base = explanation.get("base_vader_score", 0.0)
    final = explanation.get("final_score", 0.0)
    label = explanation.get("label", "")
    text_preview = explanation.get("text", "")[:80]

    # Build rows: words + phrases combined, sorted by |valence|, clipped to top_n
    rows = [
        {"token": wh["word"], "valence": wh["valence"], "source": "word"}
        for wh in word_hits
    ] + [
        {"token": ph["phrase"], "valence": ph["adjustment"], "source": "phrase"}
        for ph in phrase_hits
    ]
    rows = sorted(rows, key=lambda r: abs(r["valence"]), reverse=True)[:top_n]

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No lexicon signals found in text.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    tokens = [r["token"] for r in rows]
    valences = [r["valence"] for r in rows]
    sources = [r["source"] for r in rows]

    colors = []
    for v, src in zip(valences, sources):
        if src == "phrase":
            colors.append(
                "#8e44ad" if v >= 0 else "#d35400"
            )  # purple/orange for phrases
        else:
            colors.append("#27ae60" if v >= 0 else "#c0392b")  # green/red for words

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(4, len(rows) * 0.45 + 2)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # --- left: horizontal bar chart of contributions ---
    ax = axes[0]
    y_pos = range(len(tokens))
    bars = ax.barh(list(y_pos), valences, color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"[phrase] {t}" if s == "phrase" else t for t, s in zip(tokens, sources)],
        fontsize=9,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Lexicon valence / phrase adjustment")
    ax.set_title(
        f'Score explanation\n"{text_preview}{"…" if len(explanation.get("text", "")) > 80 else ""}"',
        fontsize=9,
        loc="left",
    )
    ax.grid(axis="x", alpha=0.3)

    # value labels on bars
    for bar, val in zip(bars, valences):
        x_off = 0.05 if val >= 0 else -0.05
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + x_off,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}",
            va="center",
            ha=ha,
            fontsize=7.5,
        )

    # legend
    legend_handles = [
        Patch(facecolor="#27ae60", label="word (positive)"),
        Patch(facecolor="#c0392b", label="word (negative)"),
        Patch(facecolor="#8e44ad", label="phrase (positive)"),
        Patch(facecolor="#d35400", label="phrase (negative)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    # --- right: score summary panel ---
    ax2 = axes[1]
    ax2.axis("off")
    label_color = (
        "#27ae60"
        if label == "positif"
        else "#c0392b"
        if label == "négatif"
        else "#7f8c8d"
    )
    summary = (
        f"Base VADER\n{base:+.4f}\n\n"
        f"Phrase adj.\n{explanation.get('phrase_adjustment', 0.0):+.4f}\n\n"
        f"Final score\n{final:+.4f}"
    )
    ax2.text(
        0.5,
        0.65,
        summary,
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", edgecolor="#cccccc"),
    )
    ax2.text(
        0.5,
        0.18,
        label.upper(),
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=14,
        fontweight="bold",
        color=label_color,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_sentiment_features_overview(feat_df_pd, save_path: str | None = None):
    """Distribution overview of sentiment feature columns.

    Parameters
    ----------
    feat_df_pd : pd.DataFrame
        Output of :func:`sentvols.features.build_sentiment_features` (or
        :func:`build_full_feature_set`) converted to pandas.  Must contain
        at least a subset of the columns defined in
        :data:`sentvols.features.SENTIMENT_FEATURE_COLS`.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sentvols.features import SENTIMENT_FEATURE_COLS

    feature_cols = [c for c in SENTIMENT_FEATURE_COLS if c in feat_df_pd.columns]
    if not feature_cols:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5,
            0.5,
            "No sentiment feature columns found.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    n_cols = 4
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes_flat = np.array(axes).flatten()

    for i, col in enumerate(feature_cols):
        ax = axes_flat[i]
        data = feat_df_pd[col].dropna()
        if data.nunique() <= 3:
            counts = data.value_counts().sort_index()
            ax.bar(
                counts.index.astype(str),
                counts.values,
                color="#0055A4",
                alpha=0.8,
                edgecolor="white",
            )
        else:
            ax.hist(data, bins=30, color="#0055A4", alpha=0.75, edgecolor="white")
            if data.std() > 0:
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 200)
                ax.plot(
                    x,
                    scipy_stats.norm.pdf(x, data.mean(), data.std())
                    * len(data)
                    * (data.max() - data.min())
                    / 30,
                    color="#e74c3c",
                    linewidth=1.5,
                    linestyle="--",
                )
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Fréquence", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, axis="y")

    for j in range(len(feature_cols), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        "Distribution des Features de Sentiment",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_market_betas(beta_df_pd, save_path: str | None = None):
    """Beta and alpha distribution across tickers from the market model.

    Parameters
    ----------
    beta_df_pd : pd.DataFrame
        Output of :func:`sentvols.features.compute_market_betas` converted
        to pandas.  Expected columns: ``ticker``, ``beta``, ``alpha``,
        ``n_obs``.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    betas = beta_df_pd["beta"].dropna()
    ax.hist(betas, bins=30, color="#0055A4", alpha=0.8, edgecolor="white")
    ax.axvline(
        1.0, color="#e74c3c", linewidth=1.5, linestyle="--", label="β = 1 (marché)"
    )
    ax.axvline(
        betas.mean(),
        color="#27ae60",
        linewidth=1.5,
        label=f"Moyenne β = {betas.mean():.3f}",
    )
    ax.set_title("Distribution des Betas (β)", fontweight="bold")
    ax.set_xlabel("Beta")
    ax.set_ylabel("Nombre de titres")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    alphas = beta_df_pd["alpha"].dropna() * 100
    ax.hist(alphas, bins=30, color="#27ae60", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(
        alphas.mean(),
        color="#e74c3c",
        linewidth=1.5,
        label=f"Moyenne α = {alphas.mean():.3f}%",
    )
    ax.set_title("Distribution des Alphas (α) [%]", fontweight="bold")
    ax.set_xlabel("Alpha (%)")
    ax.set_ylabel("Nombre de titres")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    n_obs_col = (
        beta_df_pd["n_obs"]
        if "n_obs" in beta_df_pd.columns
        else np.ones(len(beta_df_pd))
    )
    sc = ax.scatter(
        beta_df_pd["beta"],
        beta_df_pd["alpha"] * 100,
        c=n_obs_col,
        cmap="viridis",
        alpha=0.6,
        s=30,
        edgecolors="none",
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(1, color="#e74c3c", linewidth=0.8, linestyle="--", label="β = 1")
    plt.colorbar(sc, ax=ax, label="n_obs")
    ax.set_title("Alpha vs Beta par Titre", fontweight="bold")
    ax.set_xlabel("Beta (β)")
    ax.set_ylabel("Alpha (α) [%]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Estimation du Modèle de Marché (OLS)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_abnormal_returns(
    df_pd,
    col_ret: str = "ret",
    col_ar: str = "abnormal_ret",
    save_path: str | None = None,
):
    """Overlay of raw returns vs abnormal returns distributions.

    Parameters
    ----------
    df_pd : pd.DataFrame
        Panel DataFrame returned by :func:`sentvols.features.add_abnormal_returns`
        converted to pandas.  Must contain ``col_ret`` and ``col_ar``.
    col_ret : str
        Raw return column name.
    col_ar : str
        Abnormal return column name.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, col, label, color in [
        (axes[0], col_ret, "Rendement brut", "#0055A4"),
        (axes[1], col_ar, "Rendement anormal (AR)", "#27ae60"),
    ]:
        data = df_pd[col].dropna()
        ax.hist(
            data,
            bins=50,
            color=color,
            alpha=0.7,
            edgecolor="white",
            density=True,
            label=label,
        )
        if data.std() > 0:
            xmin, xmax = data.min(), data.max()
            x = np.linspace(xmin, xmax, 300)
            ax.plot(
                x,
                scipy_stats.norm.pdf(x, data.mean(), data.std()),
                color="#e74c3c",
                linewidth=2,
                label="Normal fit",
            )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("Rendement")
        ax.set_ylabel("Densité")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        stats_txt = (
            f"μ={data.mean():.4f}  σ={data.std():.4f}\n"
            f"skew={scipy_stats.skew(data):.3f}  kurt={scipy_stats.kurtosis(data):.3f}"
        )
        ax.text(
            0.97,
            0.97,
            stats_txt,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4ff", edgecolor="#ccc"),
        )

    fig.suptitle(
        "Distribution des Rendements : Bruts vs Anormaux",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_ols_sentiment_results(ols_result: dict, save_path: str | None = None):
    """Coefficient dot-plot with HC3 confidence intervals for OLS results.

    Parameters
    ----------
    ols_result : dict
        Output of :func:`sentvols.explainers.test_ols_sentiment_impact`.
        Expected keys: ``feature_cols``, ``coefs``, ``se``, ``t_stats``,
        ``p_values``, ``significant``, ``r_squared``, ``f_stat``,
        ``f_pvalue``.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    feature_cols = ols_result["feature_cols"]
    coefs = np.asarray(ols_result["coefs"])
    se = np.asarray(ols_result["se"])
    significant = np.asarray(ols_result["significant"])
    r_sq = ols_result.get("r_squared", float("nan"))
    f_stat = ols_result.get("f_stat", float("nan"))
    f_pvalue = ols_result.get("f_pvalue", float("nan"))
    n_obs = ols_result.get("n_obs", "?")

    ci_95 = 1.96 * se

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, max(5, len(feature_cols) * 0.55 + 2)),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    ax = axes[0]
    y_pos = np.arange(len(feature_cols))

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_cols, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient OLS (IC 95 % HC3)", fontsize=9)
    ax.set_title(
        "Impact des Features de Sentiment sur le Rendement Anormal\n(OLS — HC3 robust SE)",
        fontweight="bold",
        fontsize=10,
    )
    ax.grid(alpha=0.3, axis="x")

    max_coef = np.nanmax(np.abs(coefs)) if len(coefs) > 0 else 1.0
    for idx, (coef, y, sig) in enumerate(zip(coefs, y_pos, significant)):
        ec = "#e74c3c" if sig else "#bdc3c7"
        ax.errorbar(
            coef, y, xerr=ci_95[idx], fmt="none", ecolor=ec, elinewidth=1.2, capsize=4
        )
        x_off = ci_95[idx] + max_coef * 0.02
        ax.text(
            coef + x_off if coef >= 0 else coef - x_off,
            y,
            f"{coef:+.4f}",
            va="center",
            ha="left" if coef >= 0 else "right",
            fontsize=7.5,
        )

    colors = ["#e74c3c" if sig else "#95a5a6" for sig in significant]
    ax.scatter(
        coefs, y_pos, c=colors, s=80, zorder=5, edgecolors="white", linewidths=0.5
    )

    alpha_level = ols_result.get("alpha_level", 0.05)
    legend_handles = [
        Patch(facecolor="#e74c3c", label=f"Significatif (p < {alpha_level})"),
        Patch(facecolor="#95a5a6", label="Non significatif"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

    ax2 = axes[1]
    ax2.axis("off")
    n_sig = int(significant.sum())
    summary_txt = (
        f"n_obs = {n_obs}\n\n"
        f"R² = {r_sq:.4f}\n\n"
        f"F = {f_stat:.3f}\n"
        f"p(F) = {f_pvalue:.4f}\n\n"
        f"Sig. features:\n{n_sig} / {len(feature_cols)}"
    )
    ax2.text(
        0.5,
        0.65,
        summary_txt,
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f0f0", edgecolor="#ccc"),
    )
    sig_color = "#27ae60" if n_sig > 0 else "#7f8c8d"
    ax2.text(
        0.5,
        0.15,
        f"{n_sig} SIG.",
        ha="center",
        va="center",
        transform=ax2.transAxes,
        fontsize=13,
        fontweight="bold",
        color=sig_color,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_portfolio_weights(
    portfolio_pd,
    col_period: str = "period",
    col_weight: str = "weight",
    col_ticker: str = "ticker",
    save_path: str | None = None,
):
    """Visualise portfolio weight distributions across periods.

    Parameters
    ----------
    portfolio_pd : pd.DataFrame
        Output of :meth:`sentvols.portfolio.PortfolioBuilder.build` converted
        to pandas.  Expected columns: ``col_period``, ``col_ticker``,
        ``col_weight``.
    col_period, col_weight, col_ticker : str
        Column name overrides.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    weights = portfolio_pd[col_weight].dropna()
    ax.hist(weights, bins=40, color="#0055A4", alpha=0.8, edgecolor="white")
    ax.axvline(
        weights.mean(),
        color="#e74c3c",
        linewidth=1.5,
        linestyle="--",
        label=f"Moyenne = {weights.mean():.4f}",
    )
    ax.set_title("Distribution Globale des Poids", fontweight="bold")
    ax.set_xlabel("Poids")
    ax.set_ylabel("Fréquence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    periods = sorted(portfolio_pd[col_period].dropna().unique())
    data_per_period = [
        portfolio_pd.loc[portfolio_pd[col_period] == p, col_weight].dropna().values
        for p in periods
    ]
    max_periods = 20
    if len(periods) > max_periods:
        step = max(1, len(periods) // max_periods)
        periods = periods[::step]
        data_per_period = data_per_period[::step]

    ax.boxplot(
        data_per_period,
        tick_labels=[str(p) for p in periods],
        patch_artist=True,
        boxprops=dict(facecolor="#d5e8f7", color="#0055A4"),
        medianprops=dict(color="#e74c3c", linewidth=1.5),
        whiskerprops=dict(color="#0055A4"),
        capprops=dict(color="#0055A4"),
        flierprops=dict(marker=".", color="#0055A4", alpha=0.4, markersize=3),
    )
    ax.set_title("Poids par Période (Boxplot)", fontweight="bold")
    ax.set_xlabel("Période")
    ax.set_ylabel("Poids")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Distribution des Poids du Portefeuille", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


@registration(module="plots")
def plot_portfolio_manager_history(
    trade_history_pd,
    col_period: str = "period",
    col_action: str = "action",
    col_cash: str = "cash_after",
    col_ticker: str = "ticker",
    save_path: str | None = None,
):
    """Visualise PortfolioManager trade history: cash curve, buy/sell events, positions.

    Parameters
    ----------
    trade_history_pd : pd.DataFrame
        Output of :attr:`sentvols.portfolio.PortfolioManager.trade_history`
        converted to pandas.  Expected columns: ``col_period``,
        ``col_action`` (``"buy"`` / ``"sell"``), ``col_cash``, ``col_ticker``.
    col_period, col_action, col_cash, col_ticker : str
        Column name overrides.
    save_path : str | None
        If provided, the figure is saved to this path at 150 dpi.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if trade_history_pd.empty:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No trades recorded.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )
        ax.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=False)

    periods = sorted(trade_history_pd[col_period].unique())
    period_idx = {p: i for i, p in enumerate(periods)}

    # --- cash curve ---
    ax = axes[0]
    cash_per_period = trade_history_pd.groupby(col_period)[col_cash].last()
    cp_x = [period_idx[p] for p in cash_per_period.index]
    ax.plot(
        cp_x,
        cash_per_period.values,
        color="#0055A4",
        linewidth=2,
        marker="o",
        markersize=4,
        label="Cash après opération",
    )
    ax.fill_between(cp_x, cash_per_period.values, alpha=0.15, color="#0055A4")
    ax.set_title("Évolution de la Trésorerie", fontweight="bold")
    ax.set_ylabel("Cash (€)")
    step_x = max(1, len(cp_x) // 15)
    ax.set_xticks(cp_x[::step_x])
    ax.set_xticklabels(
        [str(periods[i]) for i in cp_x[::step_x]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- buy/sell event markers ---
    ax = axes[1]
    buys = trade_history_pd[trade_history_pd[col_action] == "buy"]
    sells = trade_history_pd[trade_history_pd[col_action] == "sell"]
    buy_x = [period_idx[p] for p in buys[col_period]]
    sell_x = [period_idx[p] for p in sells[col_period]]
    ax.scatter(
        buy_x,
        np.ones(len(buy_x)),
        marker="^",
        color="#27ae60",
        s=60,
        alpha=0.7,
        label=f"Achat ({len(buys)})",
    )
    ax.scatter(
        sell_x,
        np.zeros(len(sell_x)),
        marker="v",
        color="#e74c3c",
        s=60,
        alpha=0.7,
        label=f"Vente ({len(sells)})",
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Vente", "Achat"])
    ax.set_title("Événements Buy / Sell", fontweight="bold")
    ax.set_xticks(cp_x[::step_x])
    ax.set_xticklabels(
        [str(periods[i]) for i in cp_x[::step_x]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")

    # --- position count per period ---
    ax = axes[2]
    n_pos_per_period = trade_history_pd.groupby(col_period)[col_ticker].nunique()
    pp_x = [period_idx[p] for p in n_pos_per_period.index]
    ax.bar(pp_x, n_pos_per_period.values, color="#8e44ad", alpha=0.8, edgecolor="white")
    ax.set_title("Nombre de Positions Distinctes par Période", fontweight="bold")
    ax.set_ylabel("Nb. titres")
    step_pp = max(1, len(pp_x) // 15)
    ax.set_xticks(pp_x[::step_pp])
    ax.set_xticklabels(
        [str(periods[i]) for i in pp_x[::step_pp]],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "Historique du Gestionnaire de Portefeuille",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig

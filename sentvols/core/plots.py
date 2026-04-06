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

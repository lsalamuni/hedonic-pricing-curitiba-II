"""Principal Component Analysis of Curitiba's neighborhoods.

The public function ``run_pca_analysis`` takes the raw census/neighborhood
Excel file, runs factorability tests, PCA, weighted-score stratification,
and produces tables and figures needed for the paper.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.decomposition import PCA

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_PCA_VARIABLES: list[str] = ["inc", "pop_1/2_sm", "lit", "grow", "dens", "pop"]

_VARIABLE_LABELS: dict[str, str] = {
    "inc": "Income",
    "pop_1/2_sm": "Pop 1/2 SM",
    "lit": "Literacy",
    "grow": "Growth",
    "dens": "Density",
    "pop": "Population",
}

_STRAT_LOW_THRESHOLD: float = -0.01
_STRAT_HIGH_THRESHOLD: float = 1.00

# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _load_pca_data(path) -> pd.DataFrame:
    """Read the PCA Excel file and normalise column names."""
    df = pd.read_excel(path, sheet_name="Data")
    df.columns = df.columns.str.strip().str.lower()
    return df


def _select_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 6 PCA variables as a clean numeric DataFrame."""
    return df[_PCA_VARIABLES].copy()


def _compute_correlation(var_df: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix for the PCA variables."""
    return var_df.corr(method="pearson")


def _run_kmo(var_array: np.ndarray) -> tuple[np.ndarray, float]:
    """Run Kaiser-Meyer-Olkin test. Returns (per-variable KMO, overall KMO)."""
    return calculate_kmo(var_array)


def _run_bartlett(var_array: np.ndarray) -> tuple[float, float]:
    """Run Bartlett's test of sphericity. Returns (chi², p-value)."""
    return calculate_bartlett_sphericity(var_array)


def _standardize(var_df: pd.DataFrame) -> pd.DataFrame:
    """Center and scale using sample std (ddof=1)."""
    return (var_df - var_df.mean()) / var_df.std(ddof=1)


def _fit_pca(var_df: pd.DataFrame) -> PCA:
    """Fit PCA on pre-scaled data."""
    pca = PCA()
    pca.fit(var_df.to_numpy())
    return pca


def _align_signs(pca: PCA) -> None:
    """Ensure deterministic eigenvector signs.

    PCA eigenvectors are only defined up to sign. This flips any component
    whose first loading is negative, ensuring reproducible results.
    Mutates ``pca.components_`` in place.
    """
    for i in range(pca.n_components_):
        if pca.components_[i, 0] < 0:
            pca.components_[i] *= -1


def _get_loadings(pca: PCA, var_names: list[str]) -> pd.DataFrame:
    """Extract factor loadings (rotation matrix) as a labelled DataFrame."""
    n_components = pca.n_components_
    cols = [f"PC{i + 1}" for i in range(n_components)]
    return pd.DataFrame(pca.components_.T, index=var_names, columns=cols)


def _get_scores(pca: PCA, var_df: pd.DataFrame, locations: pd.Series) -> pd.DataFrame:
    """Project data onto principal components and attach location labels."""
    n_components = pca.n_components_
    cols = [f"PC{i + 1}" for i in range(n_components)]
    scores = pd.DataFrame(pca.transform(var_df.to_numpy()), columns=cols)
    scores.insert(0, "loc", locations.to_numpy())
    return scores


def _get_variance_table(pca: PCA) -> pd.DataFrame:
    """Build a variance-explained importance table."""
    eigenvalues = pca.explained_variance_
    prop_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(prop_var)
    n = len(eigenvalues)
    cols = [f"PC{i + 1}" for i in range(n)]
    return pd.DataFrame(
        {
            "eigenvalue": eigenvalues,
            "proportion_of_variance": prop_var,
            "cumulative_proportion": cum_var,
        },
        index=cols,
    )


def _compute_weighted_score(
    scores: pd.DataFrame,
    variance_ratios: np.ndarray,
) -> pd.Series:
    """Weighted sum of PC1 and PC2 scores using variance proportions."""
    return (
        scores["PC1"] * variance_ratios[0] + scores["PC2"] * variance_ratios[1]
    )


def _classify_neighborhoods(
    locations: pd.Series,
    final_scores: pd.Series,
) -> pd.DataFrame:
    """Assign Low / Mid / High tier based on thresholds from Favero (2005)."""
    tier = pd.Series("mid", index=final_scores.index)
    tier[final_scores < _STRAT_LOW_THRESHOLD] = "low"
    tier[final_scores > _STRAT_HIGH_THRESHOLD] = "high"
    return pd.DataFrame({
        "loc": locations.to_numpy(),
        "final_score": final_scores.to_numpy(),
        "tier": tier.to_numpy(),
    })


# ------------------------------------------------------------------ #
# Plots
# ------------------------------------------------------------------ #


def _plot_correlation_heatmap(corr: pd.DataFrame, path) -> None:
    """Correlation heatmap with diverging color scheme."""
    labels = [_VARIABLE_LABELS.get(c, c) for c in corr.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr.values,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(240, 10, s=90, l=50, as_cmap=True),
        center=0,
        vmin=-1,
        vmax=1,
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_scree(pca: PCA, path) -> None:
    """Scree plot (explained variance per component)."""
    n = pca.n_components_
    prop = pca.explained_variance_ratio_ * 100
    idx = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(idx, prop, color="#4682B4", edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, prop, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            fontsize=9,
        )
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Percentage of Explained Variance")
    ax.set_title("Explained Variance for each Principal Component")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"PC{i}" for i in idx])
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_loadings_heatmap(loadings: pd.DataFrame, path) -> None:
    """Heatmap of factor loadings (rotation matrix)."""
    labels = [_VARIABLE_LABELS.get(c, c) for c in loadings.index]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        loadings.values,
        annot=True,
        fmt=".3f",
        cmap=sns.diverging_palette(240, 10, s=90, l=50, as_cmap=True),
        center=0,
        xticklabels=loadings.columns.tolist(),
        yticklabels=labels,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Factor Loadings")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_pca_biplot(
    pca: PCA,
    var_names: list[str],
    path,
) -> None:
    """PCA variable plot colored by cos².

    Plots variable coordinates (loadings scaled by the standard deviation
    of each PC). Arrows that reach the unit circle indicate a perfect
    correlation with the component.
    """
    sdev = np.sqrt(pca.explained_variance_[:2])
    coords = pca.components_[:2].T * sdev
    prop = pca.explained_variance_ratio_[:2] * 100

    cos2 = coords[:, 0] ** 2 + coords[:, 1] ** 2
    labels = [_VARIABLE_LABELS.get(v, v) for v in var_names]

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=cos2,
        cmap=sns.diverging_palette(240, 10, s=90, l=50, as_cmap=True),
        s=80,
        edgecolors="black",
        linewidths=0.5,
        vmin=0,
        vmax=1,
        zorder=3,
    )
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (coords[i, 0], coords[i, 1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
        )
        ax.annotate(
            "",
            xy=(coords[i, 0], coords[i, 1]),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "grey", "lw": 1.2},
        )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("cos²  (Relevance)")
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.set_xlabel(f"PC1  ({prop[0]:.1f}%)", fontweight="bold")
    ax.set_ylabel(f"PC2  ({prop[1]:.1f}%)", fontweight="bold")
    ax.set_title("PCA Plot")
    circle = plt.Circle(
        (0, 0), 1, fill=False, color="grey", linestyle="--", linewidth=0.5,
    )
    ax.add_patch(circle)
    ax.set_aspect("equal")
    lim = max(abs(coords).max() * 1.3, 1.1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ #
# LaTeX table helpers
# ------------------------------------------------------------------ #


def _df_to_latex(df: pd.DataFrame, path, caption: str, label: str) -> None:
    """Write a DataFrame as a LaTeX table file."""
    latex = df.to_latex(
        float_format="%.4f",
        caption=caption,
        label=label,
        position="htbp",
    )
    latex = re.sub(r"\\textbackslash\{\}", r"\\", latex)
    Path(path).write_text(latex, encoding="utf-8")


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def run_pca_analysis(
    data_path,
    data_dir,
    analysis_dir,
    images_dir,
) -> dict[str, Any]:
    """Execute the full PCA pipeline and save all outputs.

    Args:
        data_path: Path to pca.xlsx input file.
        data_dir: Directory for data outputs (.parquet, .xlsx).
        analysis_dir: Directory for analysis outputs (.tex).
        images_dir: Directory for image outputs (.png).

    Returns:
        Dict with keys: classification, loadings, scores, variance_table,
        kmo, bartlett.

    """
    for d in (data_dir, analysis_dir, images_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load & select
    df = _load_pca_data(data_path)
    var_df = _select_variables(df)
    var_array = var_df.to_numpy()

    # Factorability tests
    _kmo_per_var, kmo_overall = _run_kmo(var_array)
    bartlett_chi2, bartlett_p = _run_bartlett(var_array)

    # Correlation
    corr = _compute_correlation(var_df)
    _plot_correlation_heatmap(corr, images_dir / "correlation_heatmap.png")
    _df_to_latex(
        corr,
        analysis_dir / "correlation_matrix.tex",
        caption="Pearson Correlation Matrix",
        label="tab:correlation_matrix",
    )

    # PCA
    var_scaled = _standardize(var_df)

    pca = _fit_pca(var_scaled)
    _align_signs(pca)
    loadings_full = _get_loadings(pca, _PCA_VARIABLES)
    scores_full = _get_scores(pca, var_scaled, df["loc"])
    variance_table = _get_variance_table(pca)

    # Retain PC1 & PC2 (Kaiser criterion: eigenvalue > 1)
    loadings = loadings_full[["PC1", "PC2"]]
    scores = scores_full[["loc", "PC1", "PC2"]]

    # Plots
    _plot_scree(pca, images_dir / "scree_plot.png")
    _plot_loadings_heatmap(loadings, images_dir / "loadings_heatmap.png")
    _plot_pca_biplot(pca, _PCA_VARIABLES, images_dir / "pca_biplot.png")

    # Weighted score & stratification
    final_scores = _compute_weighted_score(scores, pca.explained_variance_ratio_)
    classification = _classify_neighborhoods(df["loc"], final_scores)

    # Save data
    loadings.to_parquet(data_dir / "factor_loadings.parquet")
    scores.to_parquet(data_dir / "pca_scores.parquet")
    classification.to_parquet(data_dir / "neighborhood_classification.parquet")
    classification.to_excel(
        data_dir / "neighborhood_classification.xlsx", index=False,
    )

    # Save LaTeX tables
    _df_to_latex(
        loadings,
        analysis_dir / "factor_loadings.tex",
        caption="Factor Loadings (PC1 and PC2)",
        label="tab:factor_loadings",
    )
    _df_to_latex(
        variance_table,
        analysis_dir / "variance_explained.tex",
        caption="PCA Variance Explained",
        label="tab:variance_explained",
    )
    _df_to_latex(
        classification,
        analysis_dir / "neighborhood_classification.tex",
        caption="Neighborhood Classification",
        label="tab:neighborhood_classification",
    )

    return {
        "classification": classification,
        "loadings": loadings,
        "scores": scores,
        "variance_table": variance_table,
        "kmo_overall": kmo_overall,
        "bartlett_chi2": bartlett_chi2,
        "bartlett_p": bartlett_p,
    }

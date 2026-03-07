"""Rosen's Two-Stage Hedonic Regression for Curitiba's housing market.

Replicates the two-stage hedonic pricing procedure:

1. **Box-Cox test** to select functional form (log-log).
2. **First stage** = log-log OLS of price on intrinsic attributes per tier,
   with Conley HAC standard errors for spatial autocorrelation.
3. **Diagnostic tests** = Jarque-Bera normality, White heteroskedasticity,
   Moran's I spatial autocorrelation, VIF multicollinearity.
4. **Second stage** = implicit-price regression on intrinsic + extrinsic
   (location) attributes per tier, again with Conley SEs.
5. **Wald test** for market segmentation (pooled model with tier interactions).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import boxcox as scipy_boxcox
from scipy.stats import boxcox_llf, jarque_bera, norm
from scipy.stats import t as t_dist
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_INTRINSIC_CONTINUOUS: list[str] = ["total_area_m2"]
_INTRINSIC_LOG_PLUS_ONE: list[str] = ["age_years"]

_INTRINSIC_BINARY: list[str] = [
    "offplan",
    "party_room",
    "game_room",
    "gym",
    "pool",
    "sauna",
    "bbq",
    "gourmet_space",
    "sports_court",
    "guardhouse",
    "cameras",
    "balcony",
    "playground",
    "parking_1",
    "parking_2",
    "bedroom_2",
    "bedroom_3",
    "bedroom_4",
    "bathroom_1",
    "bathroom_2",
    "bathroom_3",
    "bathroom_4",
]

_SUPPLY_SHIFTERS_LOG: list[str] = ["density", "population"]

_LOCATION_BINARY: list[str] = [
    "green_area",
    "cicloways",
    "hospitals",
    "terminals",
    "private_schools",
    "public_schools",
    "culture_facilities",
    "shoppings",
]

# High tier drops PRIVATE_SCHOOLS and CULTURE_FACILITIES due to multicollinearity
_LOCATION_BINARY_HIGH: list[str] = [
    "green_area",
    "cicloways",
    "hospitals",
    "terminals",
    "public_schools",
    "shoppings",
]

_FIRST_STAGE_DROP: list[str] = [
    "neighborhood",
    "apartment",
    "usable_area_m2",
    "bedroom_1",
    "population",
    "density",
    "cicloways",
    "green_area",
    "hospitals",
    "terminals",
    "private_schools",
    "public_schools",
    "culture_facilities",
    "shoppings",
    "latitude",
    "longitude",
    "full_address",
]

_CONLEY_DIST_CUTOFF_KM: float = 2.0
_EARTH_RADIUS_KM: float = 6371.0
_BOXCOX_LAMBDA_RANGE: tuple[float, float] = (-2.0, 2.0)

# ------------------------------------------------------------------ #
# Private helpers, Conley standard errors
# ------------------------------------------------------------------ #


def _haversine_km(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distances in km from (lat, lon) in degrees."""
    rad = np.deg2rad(coords)
    lat = rad[:, 0:1]
    lon = rad[:, 1:2]

    dlat = lat - lat.T
    dlon = lon - lon.T

    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _conley_vcov(
    x: np.ndarray,
    residuals: np.ndarray,
    coords: np.ndarray,
    dist_cutoff: float,
) -> np.ndarray:
    """Compute Conley HAC variance-covariance matrix.

    Uses a Bartlett (triangular) kernel with the Haversine distance.
    """
    n, k = x.shape
    dist_mat = _haversine_km(coords)

    # Bartlett kernel weights
    weights = np.maximum(0.0, 1.0 - dist_mat / dist_cutoff)

    # Weighted outer products of moment conditions
    meat = np.zeros((k, k))
    xu = x * residuals[:, np.newaxis]
    for i in range(n):
        for j in range(n):
            if weights[i, j] > 0:
                meat += weights[i, j] * np.outer(xu[i], xu[j])

    # Bread: (X'X)^{-1}
    bread = np.linalg.inv(x.T @ x)
    return bread @ meat @ bread


def _conley_regression(
    y: np.ndarray,
    x: np.ndarray,
    coords: np.ndarray,
    feature_names: list[str],
    dist_cutoff: float = _CONLEY_DIST_CUTOFF_KM,
) -> pd.DataFrame:
    """OLS regression with Conley HAC standard errors."""
    x_with_const = np.column_stack([np.ones(len(x)), x])
    names = ["(Intercept)", *feature_names]

    beta = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
    residuals = y - x_with_const @ beta
    vcov = _conley_vcov(x_with_const, residuals, coords, dist_cutoff)

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, beta / se, 0.0)
    n = len(y)
    k = len(beta)
    p_val = 2 * (1 - t_dist.cdf(np.abs(t_stat), df=n - k))

    return pd.DataFrame(
        {
            "coefficient": beta,
            "std_error": se,
            "t_value": t_stat,
            "p_value": p_val,
        },
        index=names,
    )


# ------------------------------------------------------------------ #
# Private helpers, Diagnostics
# ------------------------------------------------------------------ #


def _run_jarque_bera(residuals: np.ndarray) -> dict:
    """Jarque-Bera normality test on residuals."""
    stat, p = jarque_bera(residuals)
    return {"statistic": float(stat), "p_value": float(p)}


def _run_white_test(model: sm.OLS) -> dict:
    """White's test for heteroskedasticity."""
    result = model.fit()
    stat, p, _f_stat, _f_p = het_white(result.resid, result.model.exog)
    return {"statistic": float(stat), "p_value": float(p)}


def _compute_vif(x_df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factors for each regressor.

    Columns with zero variance are excluded to avoid division by zero.
    """
    varying = x_df.loc[:, x_df.std() > 0]
    x_with_const = sm.add_constant(varying)
    vif_data = []
    for i, name in enumerate(x_with_const.columns):
        if name == "const":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vif_data.append(
                {
                    "variable": name,
                    "vif": variance_inflation_factor(x_with_const.values, i),
                }
            )
    return pd.DataFrame(vif_data)


def _run_moran_test(
    residuals: np.ndarray,
    coords: np.ndarray,
    k_neighbors: int = 5,
) -> dict:
    """Moran's I test for spatial autocorrelation using KNN weights."""
    n = len(residuals)
    dist_mat = _haversine_km(coords)

    # Build KNN weight matrix (row-standardised)
    w = np.zeros((n, n))
    for i in range(n):
        neighbors = np.argsort(dist_mat[i])
        neighbors = neighbors[neighbors != i][:k_neighbors]
        w[i, neighbors] = 1.0
    row_sums = w.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    w = w / row_sums

    # Moran's I
    z = residuals - residuals.mean()
    numerator = (z[:, np.newaxis] * z[np.newaxis, :] * w).sum()
    denominator = (z**2).sum()
    s0 = w.sum()
    moran_i = (n / s0) * (numerator / denominator)

    # Expected value and variance under randomisation
    e_i = -1.0 / (n - 1)

    s1 = 0.5 * ((w + w.T) ** 2).sum()
    s2 = ((w.sum(axis=1) + w.sum(axis=0)) ** 2).sum()

    k = (1.0 / n) * ((z**4).sum() / ((z**2).sum() / n) ** 2)

    a = n * ((n**2 - 3 * n + 3) * s1 - n * s2 + 3 * s0**2)
    b = k * ((n**2 - n) * s1 - 2 * n * s2 + 6 * s0**2)
    c = (n - 1) * (n - 2) * (n - 3) * s0**2

    var_i = (a - b) / c - e_i**2

    z_score = (moran_i - e_i) / np.sqrt(var_i)
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    return {
        "moran_i": float(moran_i),
        "expected": float(e_i),
        "variance": float(var_i),
        "z_score": float(z_score),
        "p_value": float(p_value),
    }


# ------------------------------------------------------------------ #
# Private helpers, Box-Cox
# ------------------------------------------------------------------ #


def _boxcox_optimal_lambda(y: np.ndarray) -> float:
    """Find optimal Box-Cox lambda for the positive response variable."""
    y_pos = y[y > 0]
    _, lam = scipy_boxcox(y_pos)
    return float(lam)


# ------------------------------------------------------------------ #
# Private helpers, Data preparation
# ------------------------------------------------------------------ #


def _prepare_first_stage(tier_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare first-stage data: log transforms, drop non-regression columns."""
    df = tier_df.copy()

    # Log transforms
    df["price"] = np.log(df["price"].astype(float))
    df["total_area_m2"] = np.log(df["total_area_m2"].astype(float))
    df["age_years"] = np.log(df["age_years"].astype(float) + 1)

    # Replace -inf with 0
    for col in ["price", "total_area_m2"]:
        df[col] = df[col].replace(-np.inf, 0)

    # Drop columns not used in first-stage regression
    extra_drop = [
        "id",
        "url",
        "URL",
        "Outlier",
        "outlier",
        "address",
        "tier",
        "category",
        "final_score",
    ]
    cols_to_drop = [c for c in [*_FIRST_STAGE_DROP, *extra_drop] if c in df.columns]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    y = df.pop("price")
    return df, y


def _prepare_second_stage_dep_var(
    tier_df: pd.DataFrame,
    beta_area: float,
) -> pd.Series:
    """Compute OLS second-stage dependent variable: beta_area * Price / Area."""
    price = tier_df["price"].astype(float)
    area = tier_df["total_area_m2"].astype(float)
    return beta_area * (price / area)


def _prepare_second_stage_dep_var_conley(
    tier_df: pd.DataFrame,
    beta_area: float,
) -> pd.Series:
    """Compute Conley second-stage dep var: beta_area * ln(P) / ln(A)."""
    log_price = np.log(tier_df["price"].astype(float))
    log_area = np.log(tier_df["total_area_m2"].astype(float))
    return beta_area * (log_price / log_area)


def _build_second_stage_x(
    tier_df: pd.DataFrame,
    tier_name: str,
) -> pd.DataFrame:
    """Build second-stage regressor matrix with log-transformed continuous vars."""
    location_binary = _LOCATION_BINARY_HIGH if tier_name == "high" else _LOCATION_BINARY

    # Intrinsic continuous (log)
    x_parts = [
        pd.DataFrame(
            {
                "log_total_area_m2": np.log(tier_df["total_area_m2"].astype(float)),
                "log_age_years": np.log(tier_df["age_years"].astype(float) + 1),
            }
        )
    ]

    # Intrinsic binary
    intrinsic = [c for c in _INTRINSIC_BINARY if c in tier_df.columns]
    x_parts.extend(tier_df[[c]].astype(float) for c in intrinsic)

    # Supply shifters (log)
    x_parts.extend(
        pd.DataFrame({f"log_{c}": np.log(tier_df[c].astype(float))})
        for c in _SUPPLY_SHIFTERS_LOG
        if c in tier_df.columns
    )

    # Location binary
    loc_cols = [c for c in location_binary if c in tier_df.columns]
    x_parts.extend(tier_df[[c]].astype(float) for c in loc_cols)

    return pd.concat(x_parts, axis=1)


def _prepare_conley_data(
    tier_df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter to rows with valid coordinates for Conley regression."""
    mask = tier_df["latitude"].notna() & tier_df["longitude"].notna()
    df = tier_df[mask].copy()
    coords = df[["latitude", "longitude"]].to_numpy()
    return df, coords


# ------------------------------------------------------------------ #
# Private helpers, Plots
# ------------------------------------------------------------------ #


def _plot_residual_diagnostics(
    residuals: np.ndarray,
    fitted: np.ndarray,
    tier_name: str,
    stage: str,
    images_dir: Path,
) -> None:
    """Generate residual diagnostic plots (residuals vs fitted, histogram)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs Fitted
    axes[0].scatter(fitted, residuals, alpha=0.3, s=10, color="#4682B4")
    axes[0].axhline(0, color="red", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Fitted Values")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title(f"Residuals vs Fitted ({tier_name.title()} Tier, {stage})")

    # Histogram of residuals
    axes[1].hist(residuals, bins=40, color="#4682B4", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Residual Distribution ({tier_name.title()} Tier, {stage})")

    fig.tight_layout()
    fig.savefig(
        images_dir / f"residuals_{stage}_{tier_name}.png",
        dpi=300,
    )
    plt.close(fig)


def _plot_boxcox(
    y: np.ndarray,
    tier_name: str,
    images_dir: Path,
) -> float:
    """Plot Box-Cox log-likelihood and return optimal lambda."""
    y_pos = y[y > 0]
    lambdas = np.linspace(-2.0, 2.0, 201)
    llf = [boxcox_llf(lam, y_pos) for lam in lambdas]

    _, opt_lambda = scipy_boxcox(y_pos)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, llf, color="#4682B4", linewidth=1.5)
    ax.axvline(
        opt_lambda,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"Optimal \u03bb = {opt_lambda:.2f}",
    )
    ax.set_xlabel("\u03bb")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title(f"Box-Cox Transformation ({tier_name.title()} Tier)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(images_dir / f"boxcox_{tier_name}.png", dpi=300)
    plt.close(fig)

    return float(opt_lambda)


# ------------------------------------------------------------------ #
# Private helpers, LaTeX tables
# ------------------------------------------------------------------ #


_SIG_FOOTNOTE = (
    "\\textit{Signif. codes:} *** $p<0.001$, ** $p<0.01$, * $p<0.05$, . $p<0.1$"
)

_SIG_THRESHOLDS: tuple[tuple[float, str], ...] = (
    (0.001, "***"),
    (0.01, "**"),
    (0.05, "*"),
    (0.1, "."),
)


def _significance_stars(p: float) -> str:
    """Return significance stars for a given p-value."""
    for threshold, stars in _SIG_THRESHOLDS:
        if p < threshold:
            return stars
    return ""


def _add_stars_to_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add significance stars to coefficient columns based on p-value columns.

    Handles both per-tier tables (columns: coefficient, p_value) and combined
    tables (columns: {tier}_coef, {tier}_p).
    """
    out = df.copy()

    # Per-tier table: coefficient + p_value
    if "coefficient" in out.columns and "p_value" in out.columns:
        out["coefficient"] = [
            f"{c:.4f}{_significance_stars(p)}"
            for c, p in zip(out["coefficient"], out["p_value"], strict=True)
        ]
        out = out.drop(columns=["t_value", "p_value"], errors="ignore")
        return out

    # Combined table: {tier}_coef + {tier}_p
    for col in list(out.columns):
        if col.endswith("_coef"):
            tier = col.removesuffix("_coef")
            p_col = f"{tier}_p"
            if p_col in out.columns:
                out[col] = [
                    f"{c:.4f}{_significance_stars(p)}"
                    for c, p in zip(out[col], out[p_col], strict=True)
                ]
                out = out.drop(columns=[p_col])

    return out


def _coef_table_to_latex(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> None:
    """Write coefficient table as a LaTeX file."""
    starred = _add_stars_to_table(df)
    latex = starred.to_latex(
        float_format="%.4f",
        caption=caption,
        label=label,
        position="htbp",
    )
    # Insert significance footnote before \end{table}
    if _SIG_FOOTNOTE not in latex and any(
        c for c in starred.columns if c == "coefficient" or c.endswith("_coef")
    ):
        latex = latex.replace(
            "\\end{table}",
            f"\\vspace{{0.3em}}\n{_SIG_FOOTNOTE}\n\\end{{table}}",
        )
    path.write_text(latex, encoding="utf-8")


# ------------------------------------------------------------------ #
# Private helpers, paper-formatted LaTeX tables
# ------------------------------------------------------------------ #

_FIRST_STAGE_PAPER_VARS: list[tuple[str, str]] = [
    ("total_area_m2", "Total area"),
    ("age_years", "Age"),
    ("offplan", "Off-plan"),
    ("party_room", "Party room"),
    ("game_room", "Game room"),
    ("gym", "Gym"),
    ("pool", "Pool"),
    ("sauna", "Sauna"),
    ("bbq", "BBQ"),
    ("gourmet_space", "Gourmet space"),
    ("sports_court", "Sports court"),
    ("guardhouse", "Guardhouse"),
    ("cameras", "Cameras"),
    ("balcony", "Balcony"),
    ("playground", "Playground"),
    ("parking_1", "Parking (1 spot)"),
    ("parking_2", "Parking (2+ spots)"),
    ("bedroom_2", "Bedroom (2)"),
    ("bedroom_3", "Bedroom (3)"),
    ("bedroom_4", "Bedroom (4+)"),
    ("bathroom_1", "Bathroom (1)"),
    ("bathroom_2", "Bathroom (2)"),
    ("bathroom_3", "Bathroom (3)"),
    ("bathroom_4", "Bathroom (4+)"),
]

_SECOND_STAGE_PAPER_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Intrinsic Characteristics",
        [
            ("log_total_area_m2", "Log(total area)"),
            ("log_age_years", "Log(age)"),
            ("offplan", "Off-plan"),
            ("party_room", "Party room"),
            ("game_room", "Game room"),
            ("gym", "Gym"),
            ("pool", "Pool"),
            ("sauna", "Sauna"),
            ("bbq", "BBQ"),
            ("gourmet_space", "Gourmet space"),
            ("sports_court", "Sports court"),
            ("guardhouse", "Guardhouse"),
            ("cameras", "Cameras"),
            ("balcony", "Balcony"),
            ("playground", "Playground"),
            ("parking_1", "Parking (1 spot)"),
            ("parking_2", "Parking (2+ spots)"),
            ("bedroom_2", "Bedroom (2)"),
            ("bedroom_3", "Bedroom (3)"),
            ("bedroom_4", "Bedroom (4+)"),
            ("bathroom_1", "Bathroom (1)"),
            ("bathroom_2", "Bathroom (2)"),
            ("bathroom_3", "Bathroom (3)"),
            ("bathroom_4", "Bathroom (4+)"),
        ],
    ),
    (
        "Supply Shifters",
        [
            ("log_density", "Log(density)"),
            ("log_population", "Log(population)"),
        ],
    ),
    (
        "Locational Attributes",
        [
            ("green_area", "Green area"),
            ("cicloways", "Cycleways"),
            ("hospitals", "Hospitals"),
            ("terminals", "Terminals"),
            ("private_schools", "Private schools"),
            ("public_schools", "Public schools"),
            ("culture_facilities", "Culture facilities"),
            ("shoppings", "Shopping centers"),
        ],
    ),
]


_LATEX_SIG_MAP: dict[str, str] = {
    "***": "$^{***}$",
    "**": "$^{**}$",
    "*": "$^{*}$",
    ".": "$^{\\cdot}$",
}


def _latex_sig(p: float) -> str:
    """Return LaTeX-formatted significance stars."""
    if np.isnan(p):
        return ""
    stars = _significance_stars(p)
    return _LATEX_SIG_MAP.get(stars, "")


def _fmt_first_stage_cell(coef: float, se: float, p: float) -> str:
    """Format first-stage cell: 3 decimal places."""
    if np.isnan(coef):
        return "---"
    stars = _latex_sig(p)
    sign = "$-$" if coef < 0 else ""
    return f"{sign}{abs(coef):.3f}{stars} ({se:.3f})"


def _fmt_second_stage_cell(coef: float, se: float, p: float) -> str:
    """Format second-stage cell: 4 decimal places."""
    if np.isnan(coef):
        return "---"
    stars = _latex_sig(p)
    sign = "$-$" if coef < 0 else ""
    return f"{sign}{abs(coef):.4f}{stars} ({se:.4f})"


def _write_first_stage_paper_table(results: dict, path: Path) -> None:
    """Write paper-formatted first-stage LaTeX table."""
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{First Stage Hedonic Regression Results "
        r"(Log-Log Specification).}\label{tab:first-stage}",
        r"\begin{threeparttable}",
        r"\scriptsize",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Variable & Low Tier & Mid Tier & High Tier \\",
        r"\midrule",
    ]

    # Intercept row
    icells = []
    for tier in ("low", "mid", "high"):
        table = results[tier]["first_stage"]["conley_table"]
        if "(Intercept)" in table.index:
            row = table.loc["(Intercept)"]
            icells.append(
                _fmt_first_stage_cell(
                    row["coefficient"], row["std_error"], row["p_value"]
                )
            )
        else:
            icells.append("---")
    lines.append(f"Intercept & {' & '.join(icells)} \\\\")

    for var_name, display_name in _FIRST_STAGE_PAPER_VARS:
        cells = []
        for tier in ("low", "mid", "high"):
            table = results[tier]["first_stage"]["conley_table"]
            if var_name in table.index:
                row = table.loc[var_name]
                cells.append(
                    _fmt_first_stage_cell(
                        row["coefficient"], row["std_error"], row["p_value"]
                    )
                )
            else:
                cells.append("---")
        lines.append(f"{display_name} & {' & '.join(cells)} \\\\")

    lines.append(r"\midrule")

    r2_cells = []
    adj_r2_cells = []
    n_cells = []
    for tier in ("low", "mid", "high"):
        ols = results[tier]["first_stage"]["ols_result"]
        r2_cells.append(f"{ols.rsquared:.3f}")
        adj_r2_cells.append(f"{ols.rsquared_adj:.3f}")
        n_cells.append(f"{int(ols.nobs):,}")
    lines.append(f"$R^2$ & {' & '.join(r2_cells)} \\\\")
    lines.append(f"Adj.\\ $R^2$ & {' & '.join(adj_r2_cells)} \\\\")
    lines.append(f"$N$ & {' & '.join(n_cells)} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\scriptsize",
            r"\item \textit{Note:} Conley (1999) HAC standard errors in parentheses.",
            r"\item $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$;"
            r" $^{\cdot}\,p<0.10$.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_second_stage_paper_table(results: dict, path: Path) -> None:
    """Write paper-formatted second-stage LaTeX table."""
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Second Stage Hedonic Regression Results "
        r"(Implicit Prices).}\label{tab:second-stage}",
        r"\begin{threeparttable}",
        r"\scriptsize",
        r"\begin{tabular}{l ccc}",
        r"\toprule",
        r"Variable & Low Tier & Mid Tier & High Tier \\",
        r"\midrule",
    ]

    # Intercept row
    icells = []
    for tier in ("low", "mid", "high"):
        table = results[tier]["second_stage"]["conley_table"]
        if "(Intercept)" in table.index:
            row = table.loc["(Intercept)"]
            icells.append(
                _fmt_second_stage_cell(
                    row["coefficient"], row["std_error"], row["p_value"]
                )
            )
        else:
            icells.append("---")
    lines.append(f"Intercept & {' & '.join(icells)} \\\\")

    for section_name, variables in _SECOND_STAGE_PAPER_SECTIONS:
        if section_name != "Intrinsic Characteristics":
            lines.append(r"\midrule")
        lines.append(f"\\textit{{{section_name}}} & & & \\\\")

        for var_name, display_name in variables:
            cells = []
            for tier in ("low", "mid", "high"):
                table = results[tier]["second_stage"]["conley_table"]
                if var_name in table.index:
                    row = table.loc[var_name]
                    cells.append(
                        _fmt_second_stage_cell(
                            row["coefficient"], row["std_error"], row["p_value"]
                        )
                    )
                else:
                    cells.append("---")
            lines.append(f"{display_name} & {' & '.join(cells)} \\\\")

    lines.append(r"\midrule")

    r2_cells = []
    adj_r2_cells = []
    n_cells = []
    for tier in ("low", "mid", "high"):
        ols = results[tier]["second_stage"]["ols_result"]
        r2_cells.append(f"{ols.rsquared:.3f}")
        adj_r2_cells.append(f"{ols.rsquared_adj:.3f}")
        n_cells.append(f"{int(ols.nobs):,}")
    lines.append(f"$R^2$ & {' & '.join(r2_cells)} \\\\")
    lines.append(f"Adj.\\ $R^2$ & {' & '.join(adj_r2_cells)} \\\\")
    lines.append(f"$N$ & {' & '.join(n_cells)} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{tablenotes}",
            r"\scriptsize",
            r"\item \textit{Note:} Conley (1999) HAC standard errors in parentheses."
            r" Dependent variable: $\hat{p} = \partial p / \partial\text{Area}$.",
            r"\item $^{***}p<0.001$; $^{**}p<0.01$; $^{*}p<0.05$;"
            r" $^{\cdot}\,p<0.10$."
            r" \textemdash\ indicates variable dropped due to multicollinearity.",
            r"\end{tablenotes}",
            r"\end{threeparttable}",
            r"\end{table}",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ #
# Core analysis, single-tier runners
# ------------------------------------------------------------------ #


def _run_first_stage(
    tier_df: pd.DataFrame,
    tier_name: str,
    images_dir: Path,
) -> dict:
    """Run first-stage regression for a single tier."""
    x_df, y = _prepare_first_stage(tier_df)

    # OLS fit
    x_with_const = sm.add_constant(x_df.astype(float))
    model = sm.OLS(y.astype(float), x_with_const)
    result = model.fit()

    # Diagnostic plots
    _plot_residual_diagnostics(
        result.resid.to_numpy(),
        result.fittedvalues.to_numpy(),
        tier_name,
        "first_stage",
        images_dir,
    )

    # Diagnostics
    jb = _run_jarque_bera(result.resid.to_numpy())
    white = _run_white_test(model)
    vif = _compute_vif(x_df.astype(float))

    # Moran's I (requires coords)
    mask = tier_df["latitude"].notna() & tier_df["longitude"].notna()
    moran = None
    if mask.sum() == len(tier_df):
        coords = tier_df[["latitude", "longitude"]].to_numpy()
        moran = _run_moran_test(result.resid.to_numpy(), coords)

    # Conley SEs
    conley_df, conley_coords = _prepare_conley_data(tier_df)
    conley_x_df, conley_y = _prepare_first_stage(conley_df)
    conley_result = _conley_regression(
        conley_y.to_numpy(),
        conley_x_df.to_numpy().astype(float),
        conley_coords,
        list(conley_x_df.columns),
    )

    return {
        "ols_summary": result.summary2(),
        "ols_result": result,
        "conley_table": conley_result,
        "jarque_bera": jb,
        "white_test": white,
        "vif": vif,
        "moran": moran,
        "beta_area": float(result.params.get("total_area_m2", 0)),
        "conley_beta_area": float(
            conley_result.loc["total_area_m2", "coefficient"]
            if "total_area_m2" in conley_result.index
            else 0
        ),
    }


def _run_second_stage(
    tier_df: pd.DataFrame,
    tier_name: str,
    beta_area_ols: float,
    beta_area_conley: float,
    images_dir: Path,
) -> dict:
    """Run second-stage regression for a single tier."""
    # OLS second stage
    price_m2 = _prepare_second_stage_dep_var(tier_df, beta_area_ols)
    x_df = _build_second_stage_x(tier_df, tier_name)

    x_with_const = sm.add_constant(x_df)
    model = sm.OLS(price_m2.astype(float), x_with_const.astype(float))
    result = model.fit()

    # Diagnostics
    _plot_residual_diagnostics(
        result.resid.to_numpy(),
        result.fittedvalues.to_numpy(),
        tier_name,
        "second_stage",
        images_dir,
    )
    jb = _run_jarque_bera(result.resid.to_numpy())
    white = _run_white_test(model)
    vif = _compute_vif(x_df.astype(float))

    # Conley SEs (second stage)
    conley_df, conley_coords = _prepare_conley_data(tier_df)
    conley_price_m2 = _prepare_second_stage_dep_var_conley(conley_df, beta_area_conley)
    conley_x_df = _build_second_stage_x(conley_df, tier_name)

    conley_result = _conley_regression(
        conley_price_m2.to_numpy(),
        conley_x_df.to_numpy().astype(float),
        conley_coords,
        list(conley_x_df.columns),
    )

    return {
        "ols_summary": result.summary2(),
        "ols_result": result,
        "conley_table": conley_result,
        "jarque_bera": jb,
        "white_test": white,
        "vif": vif,
    }


# ------------------------------------------------------------------ #
# Core analysis, Wald test for market segmentation
# ------------------------------------------------------------------ #


def _run_wald_test(
    tiers: dict[str, pd.DataFrame],
) -> dict:
    """Robust Wald test for market segmentation (tier interactions).

    Pools all tiers, fits an unrestricted model with tier interactions on
    Total_area_m2 and Age_years, and tests H0: all tier interactions = 0.
    """
    pooled = pd.concat(
        [tiers[t].assign(tier=t) for t in ("low", "mid", "high")], ignore_index=True
    )

    mask = pooled["latitude"].notna() & pooled["longitude"].notna()
    pooled = pooled[mask].copy()

    # Log transforms
    pooled["log_price"] = np.log(pooled["price"].astype(float))
    pooled["log_area"] = np.log(pooled["total_area_m2"].astype(float))
    pooled["log_age"] = np.log(pooled["age_years"].astype(float) + 1)

    # Replace -inf with 0
    for col in ["log_price", "log_area"]:
        pooled[col] = pooled[col].replace(-np.inf, 0)

    # Create tier dummies and interactions
    pooled["tier_mid"] = (pooled["tier"] == "mid").astype(float)
    pooled["tier_high"] = (pooled["tier"] == "high").astype(float)
    pooled["tier_mid_area"] = pooled["tier_mid"] * pooled["log_area"]
    pooled["tier_high_area"] = pooled["tier_high"] * pooled["log_area"]
    pooled["tier_mid_age"] = pooled["tier_mid"] * pooled["log_age"]
    pooled["tier_high_age"] = pooled["tier_high"] * pooled["log_age"]

    # Build regressor matrix
    x_cols = [
        "tier_mid",
        "tier_high",
        "log_area",
        "log_age",
        "tier_mid_area",
        "tier_high_area",
        "tier_mid_age",
        "tier_high_age",
        *_INTRINSIC_BINARY,
    ]
    x_cols = [c for c in x_cols if c in pooled.columns]

    y = pooled["log_price"].to_numpy()
    x_df = pooled[x_cols].astype(float)
    x_with_const = sm.add_constant(x_df)

    model = sm.OLS(y, x_with_const)
    result = model.fit(cov_type="HC3")

    # Test H0: all tier interaction terms = 0
    interaction_terms = [c for c in x_with_const.columns if c.startswith("tier_")]
    restriction_str = ", ".join(f"{t} = 0" for t in interaction_terms)

    wald = result.wald_test(restriction_str, use_f=True, scalar=True)

    return {
        "wald_statistic": float(wald.statistic),
        "p_value": float(wald.pvalue),
        "df_num": int(wald.df_num),
        "df_denom": int(wald.df_denom) if wald.df_denom is not None else None,
        "interaction_terms": interaction_terms,
        "model_summary": result.summary2(),
    }


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def run_rosen_analysis(
    tiers: dict[str, pd.DataFrame],
    data_dir: Path,
    analysis_dir: Path,
    images_dir: Path,
) -> dict:
    """Execute the full Rosen two-stage hedonic analysis.

    Args:
        tiers: Dict with keys ``"low"``, ``"mid"``, ``"high"`` mapping to
            tier DataFrames (output of ``merge_location_attributes``).
        data_dir: Directory for data outputs.
        analysis_dir: Directory for LaTeX tables.
        images_dir: Directory for plot outputs.

    Returns:
        Nested dict with results for each tier and the Wald test.

    """
    for d in (data_dir, analysis_dir, images_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    results = {}
    first_stage_tables = {}
    second_stage_tables = {}

    for tier_name in ("low", "mid", "high"):
        tier_df = tiers[tier_name]

        # Box-Cox
        opt_lambda = _plot_boxcox(
            tier_df["price"].to_numpy().astype(float),
            tier_name,
            images_dir,
        )

        # First stage
        first = _run_first_stage(tier_df, tier_name, images_dir)

        # Second stage
        second = _run_second_stage(
            tier_df,
            tier_name,
            first["beta_area"],
            first["conley_beta_area"],
            images_dir,
        )

        results[tier_name] = {
            "boxcox_lambda": opt_lambda,
            "first_stage": first,
            "second_stage": second,
        }

        first_stage_tables[tier_name] = first["conley_table"]
        second_stage_tables[tier_name] = second["conley_table"]

        # Save per-tier LaTeX tables
        _coef_table_to_latex(
            first["conley_table"],
            analysis_dir / f"first_stage_{tier_name}.tex",
            caption=f"First Stage Conley Regression ({tier_name.title()} Tier)",
            label=f"tab:first_stage_{tier_name}",
        )
        _coef_table_to_latex(
            second["conley_table"],
            analysis_dir / f"second_stage_{tier_name}.tex",
            caption=f"Second Stage Conley Regression ({tier_name.title()} Tier)",
            label=f"tab:second_stage_{tier_name}",
        )

        # Save VIF tables
        _coef_table_to_latex(
            first["vif"],
            analysis_dir / f"vif_first_stage_{tier_name}.tex",
            caption=f"VIF First Stage ({tier_name.title()} Tier)",
            label=f"tab:vif_first_stage_{tier_name}",
        )

    # Combined first-stage table (all tiers side by side)
    combined_first = _combine_tier_tables(first_stage_tables)
    combined_first.to_parquet(data_dir / "first_stage_results.parquet")
    _coef_table_to_latex(
        combined_first,
        analysis_dir / "first_stage_all_tiers.tex",
        caption="First Stage Conley Regression (All Tiers)",
        label="tab:first_stage_all_tiers",
    )

    # Combined second-stage table
    combined_second = _combine_tier_tables(second_stage_tables)
    combined_second.to_parquet(data_dir / "second_stage_results.parquet")
    _coef_table_to_latex(
        combined_second,
        analysis_dir / "second_stage_all_tiers.tex",
        caption="Second Stage Conley Regression (All Tiers)",
        label="tab:second_stage_all_tiers",
    )

    # Paper-formatted tables
    _write_first_stage_paper_table(results, analysis_dir / "first_stage_paper.tex")
    _write_second_stage_paper_table(results, analysis_dir / "second_stage_paper.tex")

    # Wald test
    wald = _run_wald_test(tiers)
    results["wald_test"] = wald

    # Save diagnostics summary
    diag = _build_diagnostics_summary(results)
    diag.to_parquet(data_dir / "diagnostics_summary.parquet")
    _coef_table_to_latex(
        diag,
        analysis_dir / "diagnostics_summary.tex",
        caption="Diagnostic Tests Summary",
        label="tab:diagnostics",
    )

    return results


def _combine_tier_tables(
    tier_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Combine per-tier coefficient tables into one wide table."""
    dfs = []
    for tier_name, table in tier_tables.items():
        renamed = table[["coefficient", "std_error", "p_value"]].copy()
        renamed.columns = [
            f"{tier_name}_coef",
            f"{tier_name}_se",
            f"{tier_name}_p",
        ]
        dfs.append(renamed)
    return pd.concat(dfs, axis=1)


def _build_diagnostics_summary(results: dict) -> pd.DataFrame:
    """Build a summary table of diagnostic tests across tiers."""
    rows = []
    for tier_name in ("low", "mid", "high"):
        r = results[tier_name]
        fs = r["first_stage"]
        ss = r["second_stage"]
        row = {
            "tier": tier_name,
            "boxcox_lambda": r["boxcox_lambda"],
            "jb_first_stat": fs["jarque_bera"]["statistic"],
            "jb_first_p": fs["jarque_bera"]["p_value"],
            "white_first_stat": fs["white_test"]["statistic"],
            "white_first_p": fs["white_test"]["p_value"],
            "jb_second_stat": ss["jarque_bera"]["statistic"],
            "jb_second_p": ss["jarque_bera"]["p_value"],
            "white_second_stat": ss["white_test"]["statistic"],
            "white_second_p": ss["white_test"]["p_value"],
        }
        if fs["moran"] is not None:
            row["moran_i"] = fs["moran"]["moran_i"]
            row["moran_p"] = fs["moran"]["p_value"]
        rows.append(row)

    wald = results["wald_test"]
    rows.append(
        {
            "tier": "wald_test",
            "boxcox_lambda": None,
            "jb_first_stat": wald["wald_statistic"],
            "jb_first_p": wald["p_value"],
        }
    )

    return pd.DataFrame(rows)

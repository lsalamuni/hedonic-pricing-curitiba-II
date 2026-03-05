from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hedonic_analysis.analysis.pca_housing import (
    _build_adequacy_table,
    _classify_neighborhoods,
    _compute_correlation,
    _compute_weighted_score,
    _fit_pca,
    _get_loadings,
    _get_scores,
    _get_variance_table,
    _kmo_interpretation,
    _select_variables,
    _standardize,
    run_pca_analysis,
)

_N_VARS = 6
_N_OBS = 20
_N_BAIRROS = 75
_KMO_MIN = 0.5
_BARTLETT_MAX_P = 0.05


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(42)
    n = _N_OBS
    return pd.DataFrame(
        {
            "loc": [f"Bairro_{i}" for i in range(n)],
            "pop": rng.integers(1000, 100_000, size=n),
            "dom": rng.integers(500, 50_000, size=n),
            "dom_part": rng.integers(500, 50_000, size=n),
            "area": rng.uniform(1, 20, size=n),
            "pop_urb": rng.integers(1000, 100_000, size=n),
            "urb": np.ones(n, dtype=int),
            "dens": rng.uniform(5, 120, size=n),
            "pop_2010": rng.integers(1000, 100_000, size=n),
            "grow": rng.uniform(-0.2, 0.5, size=n),
            "inc": rng.uniform(500, 4000, size=n),
            "pop_1/2_sm": rng.uniform(0.001, 0.03, size=n),
            "pop_10_sm": rng.uniform(0.4, 0.65, size=n),
            "lit": rng.uniform(0.9, 1.0, size=n),
        },
    )


@pytest.fixture
def var_df(sample_df):
    return _select_variables(sample_df)


@pytest.fixture
def scaled_df(var_df):
    return _standardize(var_df)


@pytest.fixture
def fitted_pca(scaled_df):
    return _fit_pca(scaled_df)


def test_select_variables_columns(sample_df):
    result = _select_variables(sample_df)
    assert list(result.columns) == [
        "inc",
        "pop_1/2_sm",
        "lit",
        "grow",
        "dens",
        "pop",
    ]


def test_select_variables_shape(sample_df):
    result = _select_variables(sample_df)
    assert result.shape == (_N_OBS, _N_VARS)


def test_correlation_is_square(var_df):
    corr = _compute_correlation(var_df)
    assert corr.shape == (_N_VARS, _N_VARS)


def test_correlation_diagonal_is_one(var_df):
    corr = _compute_correlation(var_df)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), 1.0)


def test_correlation_is_symmetric(var_df):
    corr = _compute_correlation(var_df)
    np.testing.assert_allclose(corr.to_numpy(), corr.to_numpy().T)


def test_pca_n_components(fitted_pca):
    assert fitted_pca.n_components_ == _N_VARS


def test_pca_explained_variance_sums_to_one(fitted_pca):
    assert pytest.approx(fitted_pca.explained_variance_ratio_.sum(), abs=1e-6) == 1.0


def test_loadings_shape(fitted_pca):
    loadings = _get_loadings(
        fitted_pca,
        ["inc", "pop_1/2_sm", "lit", "grow", "dens", "pop"],
    )
    assert loadings.shape == (_N_VARS, _N_VARS)


def test_loadings_columns(fitted_pca):
    loadings = _get_loadings(
        fitted_pca,
        ["inc", "pop_1/2_sm", "lit", "grow", "dens", "pop"],
    )
    assert list(loadings.columns) == [
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5",
        "PC6",
    ]


def test_scores_shape(fitted_pca, scaled_df, sample_df):
    scores = _get_scores(fitted_pca, scaled_df, sample_df["loc"])
    assert scores.shape == (_N_OBS, _N_VARS + 1)


def test_scores_has_loc_column(fitted_pca, scaled_df, sample_df):
    scores = _get_scores(fitted_pca, scaled_df, sample_df["loc"])
    assert scores.columns[0] == "loc"


def test_variance_table_shape(fitted_pca):
    vt = _get_variance_table(fitted_pca)
    assert vt.shape == (_N_VARS, 3)


def test_variance_cumulative_ends_at_one(fitted_pca):
    vt = _get_variance_table(fitted_pca)
    assert pytest.approx(vt["cumulative_proportion"].iloc[-1], abs=1e-6) == 1.0


def test_weighted_score_length(fitted_pca, scaled_df, sample_df):
    scores = _get_scores(fitted_pca, scaled_df, sample_df["loc"])
    ws = _compute_weighted_score(scores, fitted_pca.explained_variance_ratio_)
    assert len(ws) == _N_OBS


def test_classify_low():
    locs = pd.Series(["A", "B", "C"])
    scores = pd.Series([-1.0, 0.5, 2.0])
    result = _classify_neighborhoods(locs, scores)
    assert result["tier"].iloc[0] == "low"


def test_classify_mid():
    locs = pd.Series(["A", "B", "C"])
    scores = pd.Series([-1.0, 0.5, 2.0])
    result = _classify_neighborhoods(locs, scores)
    assert result["tier"].iloc[1] == "mid"


def test_classify_high():
    locs = pd.Series(["A", "B", "C"])
    scores = pd.Series([-1.0, 0.5, 2.0])
    result = _classify_neighborhoods(locs, scores)
    assert result["tier"].iloc[2] == "high"


def test_classify_columns():
    locs = pd.Series(["A"])
    scores = pd.Series([0.0])
    result = _classify_neighborhoods(locs, scores)
    assert list(result.columns) == ["loc", "final_score", "tier"]


def test_kmo_interpretation():
    assert _kmo_interpretation(0.95) == "Marvelous adequacy"
    assert _kmo_interpretation(0.85) == "Meritorious adequacy"
    assert _kmo_interpretation(0.73) == "Middling adequacy"
    assert _kmo_interpretation(0.65) == "Mediocre adequacy"
    assert _kmo_interpretation(0.55) == "Miserable adequacy"
    assert _kmo_interpretation(0.40) == "Unacceptable"


def test_build_adequacy_table_shape():
    df = _build_adequacy_table(kmo_overall=0.734, bartlett_p=0.001)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Test", "Result", "Interpretation"]


def test_build_adequacy_table_content():
    df = _build_adequacy_table(kmo_overall=0.734, bartlett_p=0.001)
    assert df["Test"].iloc[0] == "Kaiser-Meyer-Olkin"
    assert df["Test"].iloc[1] == "Bartlett"
    assert "0.734" in df["Result"].iloc[0]


def test_run_pca_analysis_on_real_data(tmp_path):
    data_path = (
        Path(__file__).parent.parent.parent
        / "src"
        / "hedonic_analysis"
        / "data"
        / "pca.xlsx"
    )
    if not data_path.exists():
        pytest.skip("pca.xlsx not found")

    data_dir = tmp_path / "data"
    analysis_dir = tmp_path / "analysis"
    images_dir = tmp_path / "images"
    result = run_pca_analysis(data_path, data_dir, analysis_dir, images_dir)

    assert result["classification"].shape[0] == _N_BAIRROS
    assert set(result["classification"]["tier"].unique()) <= {"low", "mid", "high"}
    assert result["loadings"].shape == (_N_VARS, 2)
    assert result["kmo_overall"] > _KMO_MIN
    assert result["bartlett_p"] < _BARTLETT_MAX_P
    assert (images_dir / "correlation_heatmap.png").exists()
    assert (images_dir / "scree_plot.png").exists()
    assert (images_dir / "pca_biplot.png").exists()
    assert (images_dir / "loadings_heatmap.png").exists()
    assert (analysis_dir / "factor_loadings.tex").exists()
    assert (analysis_dir / "adequacy_tests.tex").exists()
    assert (data_dir / "neighborhood_classification.xlsx").exists()
    assert (data_dir / "neighborhood_classification.parquet").exists()

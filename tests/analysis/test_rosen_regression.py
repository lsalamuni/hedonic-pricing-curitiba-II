import numpy as np
import pandas as pd
import pytest

from hedonic_analysis.analysis.rosen_regression import (
    _boxcox_optimal_lambda,
    _build_second_stage_x,
    _conley_regression,
    _haversine_km,
    _prepare_first_stage,
    _prepare_second_stage_dep_var,
    _run_jarque_bera,
)

_N_OBS = 100


@pytest.fixture
def sample_tier():
    rng = np.random.default_rng(42)
    n = _N_OBS
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "price": rng.integers(200_000, 3_000_000, size=n),
            "total_area_m2": rng.integers(40, 400, size=n),
            "usable_area_m2": rng.integers(30, 350, size=n),
            "age_years": rng.integers(0, 50, size=n),
            "off_plan": rng.integers(0, 2, size=n),
            "party_room": rng.integers(0, 2, size=n),
            "game_room": rng.integers(0, 2, size=n),
            "gym": rng.integers(0, 2, size=n),
            "pool": rng.integers(0, 2, size=n),
            "sauna": rng.integers(0, 2, size=n),
            "bbq": rng.integers(0, 2, size=n),
            "gourmet_space": rng.integers(0, 2, size=n),
            "sports_court": rng.integers(0, 2, size=n),
            "guardhouse": rng.integers(0, 2, size=n),
            "cameras": rng.integers(0, 2, size=n),
            "balcony": rng.integers(0, 2, size=n),
            "playground": rng.integers(0, 2, size=n),
            "parking_1": rng.integers(0, 2, size=n),
            "parking_2": rng.integers(0, 2, size=n),
            "bedroom_1": rng.integers(0, 2, size=n),
            "bedroom_2": rng.integers(0, 2, size=n),
            "bedroom_3": rng.integers(0, 2, size=n),
            "bedroom_4": rng.integers(0, 2, size=n),
            "bathroom_1": rng.integers(0, 2, size=n),
            "bathroom_2": rng.integers(0, 2, size=n),
            "bathroom_3": rng.integers(0, 2, size=n),
            "bathroom_4": rng.integers(0, 2, size=n),
            "neighborhood": ["Centro"] * n,
            "apartment": rng.integers(0, 2, size=n),
            "latitude": rng.uniform(-25.55, -25.35, size=n),
            "longitude": rng.uniform(-49.35, -49.20, size=n),
            "population": [38671] * n,
            "density": [117.79] * n,
            "cicloways": [1] * n,
            "hospitals": [1] * n,
            "terminals": [1] * n,
            "private_schools": [1] * n,
            "public_schools": [1] * n,
            "culture_facilities": [1] * n,
            "green_area": [1] * n,
            "shoppings": [1] * n,
        }
    )


def test_haversine_km_self_is_zero():
    coords = np.array([[-25.43, -49.27], [-25.44, -49.29]])
    d = _haversine_km(coords)
    assert d[0, 0] == pytest.approx(0, abs=1e-10)
    assert d[1, 1] == pytest.approx(0, abs=1e-10)


def test_haversine_km_symmetric():
    coords = np.array([[-25.43, -49.27], [-25.44, -49.29]])
    d = _haversine_km(coords)
    assert d[0, 1] == pytest.approx(d[1, 0], abs=1e-10)


def test_haversine_km_positive():
    coords = np.array([[-25.43, -49.27], [-25.44, -49.29]])
    d = _haversine_km(coords)
    assert d[0, 1] > 0


def test_prepare_first_stage_log_transforms(sample_tier):
    x_df, y = _prepare_first_stage(sample_tier)
    assert "total_area_m2" in x_df.columns
    assert "age_years" in x_df.columns
    # price should be log-transformed
    assert y.max() < np.log(3_000_001)


def test_prepare_first_stage_drops_extrinsic(sample_tier):
    x_df, _ = _prepare_first_stage(sample_tier)
    for col in ["population", "density", "shoppings", "neighborhood", "latitude"]:
        assert col not in x_df.columns


def test_prepare_second_stage_dep_var(sample_tier):
    beta_area = 0.5
    result = _prepare_second_stage_dep_var(sample_tier, beta_area)
    expected = beta_area * (sample_tier["price"] / sample_tier["total_area_m2"])
    pd.testing.assert_series_equal(result, expected.astype(float))


def test_build_second_stage_x_has_log_columns(sample_tier):
    x_df = _build_second_stage_x(sample_tier, "low")
    assert "log_total_area_m2" in x_df.columns
    assert "log_age_years" in x_df.columns
    assert "log_density" in x_df.columns
    assert "log_population" in x_df.columns


def test_build_second_stage_x_has_location_vars(sample_tier):
    x_df = _build_second_stage_x(sample_tier, "low")
    for var in ["green_area", "cicloways", "hospitals", "shoppings"]:
        assert var in x_df.columns


def test_build_second_stage_high_drops_multicollinear(sample_tier):
    x_df = _build_second_stage_x(sample_tier, "high")
    assert "private_schools" not in x_df.columns
    assert "culture_facilities" not in x_df.columns


def test_build_second_stage_low_keeps_all_location(sample_tier):
    x_df = _build_second_stage_x(sample_tier, "low")
    assert "private_schools" in x_df.columns
    assert "culture_facilities" in x_df.columns


def test_jarque_bera_returns_dict():
    rng = np.random.default_rng(42)
    resid = rng.normal(size=100)
    result = _run_jarque_bera(resid)
    assert "statistic" in result
    assert "p_value" in result


def test_conley_regression_returns_dataframe():
    rng = np.random.default_rng(42)
    n = 50
    x = rng.normal(size=(n, 2))
    y = x @ np.array([1.0, 2.0]) + rng.normal(size=n) * 0.1
    coords = np.column_stack(
        [
            rng.uniform(-25.55, -25.35, size=n),
            rng.uniform(-49.35, -49.20, size=n),
        ]
    )
    result = _conley_regression(y, x, coords, ["x1", "x2"])
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["coefficient", "std_error", "t_value", "p_value"]
    assert "(Intercept)" in result.index
    assert "x1" in result.index
    assert "x2" in result.index


def test_conley_regression_recovers_true_coefficients():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.normal(size=(n, 2))
    y = x @ np.array([3.0, -1.5]) + 2.0 + rng.normal(size=n) * 0.5
    coords = np.column_stack(
        [
            rng.uniform(-25.55, -25.35, size=n),
            rng.uniform(-49.35, -49.20, size=n),
        ]
    )
    result = _conley_regression(y, x, coords, ["x1", "x2"])
    assert result.loc["(Intercept)", "coefficient"] == pytest.approx(2.0, abs=0.5)
    assert result.loc["x1", "coefficient"] == pytest.approx(3.0, abs=0.5)
    assert result.loc["x2", "coefficient"] == pytest.approx(-1.5, abs=0.5)


_BOXCOX_TOL = 0.5


def test_boxcox_lambda_near_zero():
    rng = np.random.default_rng(42)
    y = np.exp(rng.normal(12, _BOXCOX_TOL, size=500))
    lam = _boxcox_optimal_lambda(y)
    assert -_BOXCOX_TOL < lam < _BOXCOX_TOL

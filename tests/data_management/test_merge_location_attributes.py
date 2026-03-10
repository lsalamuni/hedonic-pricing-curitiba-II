from pathlib import Path

import pandas as pd
import pytest

from hedonic_analysis.data_management.merge_location_attributes import (
    _add_apartment_flag,
    _normalise_neighborhood_name,
    merge_location_attributes,
)

pytestmark = pytest.mark.unit

_INFO_PATH = (
    Path(__file__).parent.parent.parent
    / "src"
    / "hedonic_analysis"
    / "data"
    / "info_neighborhoods.xlsx"
)
_CLASSIFICATION_PATH = (
    Path(__file__).parent.parent.parent
    / "bld"
    / "data"
    / "neighborhood_classification.parquet"
)

_N_LOCATION_VARS = 10


@pytest.fixture
def sample_housing():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "neighborhood": ["Centro", "Batel", "Xaxim"],
            "price": [500_000, 1_200_000, 350_000],
            "total_area_m2": [80, 150, 60],
            "latitude": [-25.43, -25.44, -25.51],
            "longitude": [-49.27, -49.29, -49.26],
        }
    )


def test_normalise_name():
    s = pd.Series(["  centro ", "Batel"])
    result = _normalise_neighborhood_name(s)
    assert result.iloc[0] == "CENTRO"
    assert result.iloc[1] == "BATEL"


def test_add_apartment_flag_from_category():
    df = pd.DataFrame({"category": ["Apartamento", "Casa", "apartamento"]})
    result = _add_apartment_flag(df)
    assert "apartment" in result.columns
    assert list(result["apartment"]) == [1, 0, 1]


def test_add_apartment_flag_no_category():
    df = pd.DataFrame({"price": [100]})
    result = _add_apartment_flag(df)
    assert "apartment" not in result.columns


@pytest.mark.integration
@pytest.mark.skipif(
    not _INFO_PATH.exists() or not _CLASSIFICATION_PATH.exists(),
    reason="Data files not found",
)
def test_merge_returns_all_tiers(sample_housing):
    result = merge_location_attributes(
        sample_housing,
        _INFO_PATH,
        _CLASSIFICATION_PATH,
    )
    assert "all" in result
    assert "low" in result
    assert "mid" in result
    assert "high" in result


@pytest.mark.integration
@pytest.mark.skipif(
    not _INFO_PATH.exists() or not _CLASSIFICATION_PATH.exists(),
    reason="Data files not found",
)
def test_merge_adds_tier_column(sample_housing):
    result = merge_location_attributes(
        sample_housing,
        _INFO_PATH,
        _CLASSIFICATION_PATH,
    )
    assert "tier" in result["all"].columns


@pytest.mark.integration
@pytest.mark.skipif(
    not _INFO_PATH.exists() or not _CLASSIFICATION_PATH.exists(),
    reason="Data files not found",
)
def test_merge_adds_location_vars(sample_housing):
    result = merge_location_attributes(
        sample_housing,
        _INFO_PATH,
        _CLASSIFICATION_PATH,
    )
    for var in ["population", "density", "hospitals", "shoppings"]:
        assert var in result["all"].columns


@pytest.mark.integration
@pytest.mark.skipif(
    not _INFO_PATH.exists() or not _CLASSIFICATION_PATH.exists(),
    reason="Data files not found",
)
def test_merge_tier_split_is_exhaustive(sample_housing):
    result = merge_location_attributes(
        sample_housing,
        _INFO_PATH,
        _CLASSIFICATION_PATH,
    )
    total = sum(len(result[t]) for t in ("low", "mid", "high"))
    assert total == len(result["all"])


@pytest.mark.integration
@pytest.mark.skipif(
    not _INFO_PATH.exists() or not _CLASSIFICATION_PATH.exists(),
    reason="Data files not found",
)
def test_unmatched_neighborhood_dropped():
    df = pd.DataFrame(
        {
            "id": [1],
            "neighborhood": ["FAKE_NEIGHBORHOOD"],
            "price": [100_000],
            "total_area_m2": [50],
            "latitude": [-25.43],
            "longitude": [-49.27],
        }
    )
    result = merge_location_attributes(df, _INFO_PATH, _CLASSIFICATION_PATH)
    assert len(result["all"]) == 0

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from hedonic_analysis.data_management.geocode_housing import (
    _build_full_address,
    geocode_housing,
)

_N_NON_OUTLIER = 3
_OUTLIER_ID = 4
_GEO = "hedonic_analysis.data_management.geocode_housing"


@pytest.fixture(autouse=True)
def _no_sleep():
    with patch(f"{_GEO}.time.sleep"):
        yield


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, _OUTLIER_ID],
            "address": [
                "Rua XV de Novembro, 100",
                "Avenida Batel, 200",
                "Rua Marechal Deodoro, 300",
                "Rua das Flores, 400",
            ],
            "neighborhood": ["Centro", "Batel", "Centro", "Centro"],
            "price": [500_000.0, 800_000.0, 600_000.0, 700_000.0],
            "usable_area_m2": [80.0, 120.0, 90.0, 100.0],
            "outlier": [0, 0, 0, 1],
        }
    )


def _make_location(lat, lon):
    loc = MagicMock()
    loc.latitude = lat
    loc.longitude = lon
    return loc


def test_build_full_address(sample_df):
    result = _build_full_address(sample_df)
    expected = "Rua XV de Novembro, 100, Centro, Curitiba, Paraná, Brazil"
    assert result.iloc[0] == expected


def test_build_full_address_all_rows(sample_df):
    result = _build_full_address(sample_df)
    assert len(result) == len(sample_df)
    assert all("Curitiba" in addr for addr in result)


@patch(f"{_GEO}.ArcGIS")
@patch(f"{_GEO}.Nominatim")
def test_geocode_housing_filters_outliers(
    mock_nominatim_cls,
    mock_arcgis_cls,
    sample_df,
):
    mock_geocoder = MagicMock()
    mock_geocoder.geocode.return_value = _make_location(-25.43, -49.27)
    mock_nominatim_cls.return_value = mock_geocoder
    mock_arcgis_cls.return_value = mock_geocoder

    result = geocode_housing(sample_df)

    assert len(result) == _N_NON_OUTLIER
    assert _OUTLIER_ID not in result["id"].to_numpy()


@patch(f"{_GEO}.ArcGIS")
@patch(f"{_GEO}.Nominatim")
def test_geocode_housing_adds_coordinates(
    mock_nominatim_cls,
    mock_arcgis_cls,
    sample_df,
):
    mock_geocoder = MagicMock()
    mock_geocoder.geocode.return_value = _make_location(-25.43, -49.27)
    mock_nominatim_cls.return_value = mock_geocoder
    mock_arcgis_cls.return_value = mock_geocoder

    result = geocode_housing(sample_df)

    assert "latitude" in result.columns
    assert "longitude" in result.columns
    assert result["latitude"].notna().all()
    assert result["longitude"].notna().all()


@patch(f"{_GEO}.ArcGIS")
@patch(f"{_GEO}.Nominatim")
def test_arcgis_fallback(
    mock_nominatim_cls,
    mock_arcgis_cls,
    sample_df,
):
    mock_nominatim = MagicMock()
    mock_nominatim.geocode.return_value = None
    mock_nominatim_cls.return_value = mock_nominatim

    mock_arcgis = MagicMock()
    mock_arcgis.geocode.return_value = _make_location(-25.43, -49.27)
    mock_arcgis_cls.return_value = mock_arcgis

    result = geocode_housing(sample_df)

    assert mock_arcgis.geocode.call_count == _N_NON_OUTLIER
    assert result["latitude"].notna().all()


@patch(f"{_GEO}.ArcGIS")
@patch(f"{_GEO}.Nominatim")
def test_cache_prevents_re_geocoding(
    mock_nominatim_cls,
    mock_arcgis_cls,
    sample_df,
    tmp_path,
):
    mock_geocoder = MagicMock()
    mock_geocoder.geocode.return_value = _make_location(-25.43, -49.27)
    mock_nominatim_cls.return_value = mock_geocoder
    mock_arcgis_cls.return_value = mock_geocoder

    cache_path = tmp_path / "cache.parquet"

    geocode_housing(sample_df, cache_path=cache_path)
    first_run_calls = mock_geocoder.geocode.call_count

    mock_geocoder.geocode.reset_mock()
    geocode_housing(sample_df, cache_path=cache_path)
    second_run_calls = mock_geocoder.geocode.call_count

    assert first_run_calls == _N_NON_OUTLIER
    assert second_run_calls == 0


@patch(f"{_GEO}.ArcGIS")
@patch(f"{_GEO}.Nominatim")
def test_output_columns(
    mock_nominatim_cls,
    mock_arcgis_cls,
    sample_df,
):
    mock_geocoder = MagicMock()
    mock_geocoder.geocode.return_value = _make_location(-25.43, -49.27)
    mock_nominatim_cls.return_value = mock_geocoder
    mock_arcgis_cls.return_value = mock_geocoder

    result = geocode_housing(sample_df)

    expected_cols = {
        "id",
        "address",
        "neighborhood",
        "price",
        "usable_area_m2",
        "outlier",
        "latitude",
        "longitude",
    }
    assert expected_cols.issubset(set(result.columns))

from unittest.mock import patch

import folium
import pandas as pd
import pytest

from hedonic_analysis.final.interactive_map import (
    _enrich_neighborhoods,
    _normalize_key,
    _strip_accents,
    build_interactive_map,
)

pytestmark = pytest.mark.unit

_MOD = "hedonic_analysis.final.interactive_map"

_POP_AGUA = 52_000
_POP_CIC = 180_000
_DENS_CIC = 35.0


# =====================================================================
# Fixtures
# =====================================================================


def _fake_props(nome: str) -> dict:
    """Build properties dict with every field any layer tooltip uses."""
    return {
        "NOME": nome,
        "TEXTO_MAPA": nome,
        "TIPO": "test",
        "CATEG_2000": "test",
        "TEXTO": "test",
        "NOME_COMPL": nome,
        "NOME_MAPA": nome,
    }


@pytest.fixture
def sample_geojson():
    ring_a = [
        [-49.3, -25.4],
        [-49.2, -25.4],
        [-49.2, -25.5],
        [-49.3, -25.4],
    ]
    ring_b = [
        [-49.3, -25.3],
        [-49.2, -25.3],
        [-49.2, -25.4],
        [-49.3, -25.3],
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring_a],
                },
                "properties": _fake_props("\u00c1GUA VERDE"),
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring_b],
                },
                "properties": _fake_props(
                    "CIDADE INDUSTRIAL DE CURITIBA",
                ),
            },
        ],
    }


@pytest.fixture
def sample_housing_df():
    return pd.DataFrame(
        {
            "address": ["Rua A, 100", "Rua B, 200", "Rua C, 300"],
            "neighborhood": ["\u00c1gua Verde", "Centro", "Batel"],
            "price": [350_000.0, 500_000.0, 750_000.0],
            "total_area_m2": [65.0, 80.0, 120.0],
            "latitude": [-25.44, -25.43, -25.45],
            "longitude": [-49.28, -49.27, -49.29],
        }
    )


@pytest.fixture
def cls_df(tmp_path):
    df = pd.DataFrame(
        {
            "loc": ["AGUA VERDE", "CIC"],
            "final_score": [1.5, -0.3],
            "tier": ["mid", "low"],
        }
    )
    path = tmp_path / "classification.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture
def pca_xlsx(tmp_path):
    df = pd.DataFrame(
        {
            "Loc": ["AGUA VERDE", "CIC"],
            "Pop": [_POP_AGUA, _POP_CIC],
            "Dens": [80.0, _DENS_CIC],
        }
    )
    path = tmp_path / "pca.xlsx"
    df.to_excel(path, sheet_name="Data", index=False)
    return path


# =====================================================================
# Tests
# =====================================================================


def test_strip_accents_removes_diacritics():
    assert _strip_accents("\u00c1GUA VERDE") == "AGUA VERDE"
    assert _strip_accents("BOQUEIR\u00c3O") == "BOQUEIRAO"
    assert _strip_accents("CENTRO") == "CENTRO"


def test_normalize_key_handles_cic():
    assert _normalize_key("CIC") == "CIDADE INDUSTRIAL DE CURITIBA"


def test_normalize_key_strips_accents_and_uppercases():
    assert _normalize_key("\u00c1gua Verde") == "AGUA VERDE"


@pytest.mark.integration
def test_enrich_neighborhoods(sample_geojson, cls_df, pca_xlsx):
    enriched = _enrich_neighborhoods(
        sample_geojson,
        cls_df,
        pca_xlsx,
    )
    props_agua = enriched["features"][0]["properties"]
    assert props_agua["tier"] == "Mid"
    assert props_agua["pop"] == _POP_AGUA

    props_cic = enriched["features"][1]["properties"]
    assert props_cic["tier"] == "Low"
    assert props_cic["pop"] == _POP_CIC
    assert props_cic["dens"] == _DENS_CIC


@pytest.mark.integration
def test_build_returns_folium_map(
    sample_geojson,
    sample_housing_df,
    cls_df,
    pca_xlsx,
    tmp_path,
):
    housing_path = tmp_path / "housing.parquet"
    sample_housing_df.to_parquet(housing_path)

    with patch(f"{_MOD}._read_shapefile", return_value=sample_geojson):
        result = build_interactive_map(
            housing_path=housing_path,
            classification_path=cls_df,
            pca_path=pca_xlsx,
            shapefiles_dir=tmp_path,
        )

    assert isinstance(result, folium.Map)


@pytest.mark.integration
def test_map_html_contains_layer_names(
    sample_geojson,
    sample_housing_df,
    cls_df,
    pca_xlsx,
    tmp_path,
):
    housing_path = tmp_path / "housing.parquet"
    sample_housing_df.to_parquet(housing_path)

    with patch(f"{_MOD}._read_shapefile", return_value=sample_geojson):
        result = build_interactive_map(
            housing_path=housing_path,
            classification_path=cls_df,
            pca_path=pca_xlsx,
            shapefiles_dir=tmp_path,
        )

    html = result._repr_html_()
    expected_groups = (
        "Neighborhoods",
        "Housing Listings",
        "Population",
        "Density",
        "Neighborhood Types",
    )
    for group in expected_groups:
        assert group in html


@pytest.mark.integration
def test_housing_popup_format(
    sample_geojson,
    sample_housing_df,
    cls_df,
    pca_xlsx,
    tmp_path,
):
    housing_path = tmp_path / "housing.parquet"
    sample_housing_df.to_parquet(housing_path)

    with patch(f"{_MOD}._read_shapefile", return_value=sample_geojson):
        result = build_interactive_map(
            housing_path=housing_path,
            classification_path=cls_df,
            pca_path=pca_xlsx,
            shapefiles_dir=tmp_path,
        )

    html = result._repr_html_()
    assert "R$" in html
    assert "m\u00b2" in html

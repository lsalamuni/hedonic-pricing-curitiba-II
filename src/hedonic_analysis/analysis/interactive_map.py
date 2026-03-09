"""Build an interactive Folium map of Curitiba housing and urban amenities.

Public function: ``build_interactive_map``.
"""

from __future__ import annotations

import copy
import unicodedata
from pathlib import Path
from typing import Any

import branca.colormap as cm
import folium
import folium.plugins
import pandas as pd
import shapefile
from pyproj import CRS, Transformer

# =====================================================================
# Constants
# =====================================================================

_CURITIBA_CENTER: list[float] = [-25.43, -49.27]
_DEFAULT_ZOOM: int = 12
_MIN_COORD_LEN: int = 2

_SHAPEFILES: dict[str, tuple[str, str]] = {
    "neighborhoods": ("Neighborhoods", "DIVISA_DE_BAIRROS.shp"),
    "parks": ("Parks", "PARQUES_E_BOSQUES.shp"),
    "squares": ("Squares", "PRACAS_E_JARDINETES.shp"),
    "cicleways": ("Cicleways", "CICLOVIA_OFICIAL.shp"),
    "cicleroutes": ("Cicleroutes", "CICLORROTA.shp"),
    "railroads": ("Railroads", "RRFSA_FERROVIAS.shp"),
    "sport": ("Sport_leisure", "CENTRO_DE_ESPORTE_E_LAZER.shp"),
    "cemiteries": ("Cemiteries", "CEMITERIOS.shp"),
    "cidadania": ("Cidadania", "RUA_DA_CIDADANIA.shp"),
    "terminals": ("Terminals", "TERMINAL_DE_TRANSPORTE.shp"),
    "dentists": ("Dentist_units", "CENTRO_DE_ESPECIALIDADES_ODONTOLOGICAS.shp"),
    "emergency": ("Emergency", "UNIDADE_DE_PRONTO_ATENDIMENTO.shp"),
    "health_units": ("Health_units", "UNIDADE_DE_SAUDE.shp"),
    "hospitals": ("Hospitals", "HOSPITAL.shp"),
    "med_units": ("Medical_units", "CENTRO_DE_ESPECIALIDADES_MEDICAS.shp"),
    "irregular": ("Irregular", "OCUPACAO_IRREGULAR.shp"),
    "lakes": ("Lakes", "HIDRO_LAGOS_LAGOAS_REPRESAS.shp"),
    "rivers": ("Rivers", "HIDRO_RIOS_LN.shp"),
    "schools": ("Schools", "ESCOLA_MUNICIPAL.shp"),
}

_NAME_ALIASES: dict[str, str] = {"CIC": "CIDADE INDUSTRIAL DE CURITIBA"}

_TIER_COLORS: dict[str, str] = {"Low": "blue", "Mid": "gold", "High": "red"}


# =====================================================================
# Shapefiles
# =====================================================================


def _strip_accents(text: str) -> str:
    """Remove combining diacritical marks from *text*."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if unicodedata.category(c) != "Mn")


def _transform_coords(
    coords: Any,
    transformer: Transformer,
) -> Any:
    """Recursively transform nested coordinate lists."""
    is_point = (
        isinstance(coords, (list, tuple))
        and len(coords) >= _MIN_COORD_LEN
        and isinstance(coords[0], (int, float))
    )
    if is_point:
        x, y = transformer.transform(coords[0], coords[1])
        return [x, y, *coords[2:]]
    return [_transform_coords(c, transformer) for c in coords]


_SHP_ENCODING = "latin-1"


def _read_shapefile(shp_path: Path) -> dict:
    """Read a shapefile and return a GeoJSON FeatureCollection in WGS84."""
    prj_path = shp_path.with_suffix(".prj")
    if prj_path.exists():
        source_crs = CRS.from_wkt(prj_path.read_text())
    else:
        source_crs = CRS.from_epsg(31982)
    target_crs = CRS.from_epsg(4326)

    needs_transform = not source_crs.equals(target_crs)
    transformer = (
        Transformer.from_crs(source_crs, target_crs, always_xy=True)
        if needs_transform
        else None
    )

    reader = shapefile.Reader(str(shp_path), encoding=_SHP_ENCODING)
    geojson = copy.deepcopy(reader.__geo_interface__)

    if transformer is not None:
        for feature in geojson["features"]:
            geom = feature["geometry"]
            geom["coordinates"] = _transform_coords(geom["coordinates"], transformer)

    return geojson


# =====================================================================
# Data loading & joining
# =====================================================================


def _normalize_key(name: str) -> str:
    """Normalize a neighbourhood name for joining: strip accents + uppercase."""
    key = _strip_accents(name).strip().upper()
    return _NAME_ALIASES.get(key, key)


def _enrich_neighborhoods(
    geojson: dict,
    classification_path: Path,
    pca_path: Path,
) -> dict:
    """Inject ``tier``, ``pop``, and ``dens`` into neighbourhood GeoJSON features."""
    cls_df = pd.read_parquet(classification_path)
    cls_lookup: dict[str, str] = {}
    for _, row in cls_df.iterrows():
        cls_lookup[_normalize_key(str(row["loc"]))] = str(row["tier"]).capitalize()

    pca_df = pd.read_excel(pca_path, sheet_name="Data")
    pca_df.columns = pca_df.columns.str.lower()
    pop_lookup: dict[str, float] = {}
    dens_lookup: dict[str, float] = {}
    for _, row in pca_df.iterrows():
        key = _normalize_key(str(row["loc"]))
        pop_lookup[key] = float(row["pop"]) if pd.notna(row["pop"]) else 0.0
        dens_lookup[key] = float(row["dens"]) if pd.notna(row["dens"]) else 0.0

    for feature in geojson["features"]:
        nome = feature["properties"].get("NOME", "")
        key = _normalize_key(nome)
        feature["properties"]["tier"] = cls_lookup.get(key, "")
        feature["properties"]["pop"] = pop_lookup.get(key, 0.0)
        feature["properties"]["dens"] = dens_lookup.get(key, 0.0)

    return geojson


# =====================================================================
# Layer builders
# =====================================================================


def _highlight_kwargs(color: str = "white", weight: int = 2) -> dict[str, Any]:
    return {"color": color, "weight": weight, "bringToFront": True}


# ---- Polygon and polyline layers ----


def _add_neighborhoods(m: folium.Map, geojson: dict) -> None:
    fg = folium.FeatureGroup(name="Neighborhoods", show=True)
    folium.GeoJson(
        geojson,
        style_function=lambda _: {
            "fillColor": "transparent",
            "color": "black",
            "weight": 1,
            "opacity": 1,
        },
        highlight_function=lambda _: _highlight_kwargs(),
        tooltip=folium.GeoJsonTooltip(
            fields=["NOME", "pop"],
            aliases=["Neighborhood:", "Population:"],
            style=_tooltip_style_str(),
        ),
    ).add_to(fg)
    fg.add_to(m)


def _add_polygon_layer(
    m: folium.Map,
    geojson: dict,
    group: str,
    fill_color: str,
    tooltip_fields: list[str] | None = None,
    tooltip_aliases: list[str] | None = None,
    *,
    show: bool = False,
) -> None:
    fg = folium.FeatureGroup(name=group, show=show)
    kwargs: dict[str, Any] = {
        "style_function": lambda _, fc=fill_color: {
            "fillColor": fc,
            "color": "transparent",
            "fillOpacity": 0.7,
            "stroke": False,
        },
        "highlight_function": lambda _: _highlight_kwargs(color="white", weight=3),
    }
    if tooltip_fields:
        kwargs["tooltip"] = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases or tooltip_fields,
            style=_tooltip_style_str(),
        )
    folium.GeoJson(geojson, **kwargs).add_to(fg)
    fg.add_to(m)


def _add_polyline_layer(
    m: folium.Map,
    geojson: dict,
    group: str,
    color: str,
    tooltip_fields: list[str] | None = None,
    tooltip_aliases: list[str] | None = None,
    *,
    fg: folium.FeatureGroup | None = None,
) -> folium.FeatureGroup:
    if fg is None:
        fg = folium.FeatureGroup(name=group, show=False)
    kwargs: dict[str, Any] = {
        "style_function": lambda _, c=color: {
            "color": c,
            "weight": 1.5,
            "opacity": 0.7,
        },
        "highlight_function": lambda _: _highlight_kwargs(weight=5),
    }
    if tooltip_fields:
        kwargs["tooltip"] = folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases or tooltip_fields,
            style=_tooltip_style_str(),
        )
    folium.GeoJson(geojson, **kwargs).add_to(fg)
    fg.add_to(m)
    return fg


# ---- Marker layers ----


def _get_point_coords(
    feature: dict,
) -> tuple[float, float] | None:
    """Return (lat, lon) from a GeoJSON point feature, or None."""
    if feature["geometry"]["type"] != "Point":
        return None
    coords = feature["geometry"]["coordinates"]
    return coords[1], coords[0]


def _add_marker_layer(
    m: folium.Map,
    geojson: dict,
    group: str,
    popup_field: str,
    icon_name: str,
    marker_color: str,
    *,
    prefix: str = "glyphicon",
    fg: folium.FeatureGroup | None = None,
) -> folium.FeatureGroup:
    if fg is None:
        fg = folium.FeatureGroup(name=group, show=False)
    for feature in geojson["features"]:
        coords = _get_point_coords(feature)
        if coords is None:
            continue
        lat, lon = coords
        popup_text = str(feature["properties"].get(popup_field, ""))
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=250),
            icon=folium.Icon(
                icon=icon_name,
                prefix=prefix,
                color=marker_color,
                icon_color="white",
            ),
        ).add_to(fg)
    fg.add_to(m)
    return fg


# ---- Housing listings ----


def _add_housing(m: folium.Map, housing_df: pd.DataFrame) -> None:
    fg = folium.FeatureGroup(name="Housing Listings", show=False)
    cluster = folium.plugins.MarkerCluster().add_to(fg)
    for _, row in housing_df.iterrows():
        if pd.isna(row["latitude"]) or pd.isna(row["longitude"]):
            continue
        popup_html = (
            f"<b>{row['address']}</b><br>"
            f"Neighborhood: {row['neighborhood']}<br>"
            f"Price: R$ {row['price']:,.0f}<br>"
            f"Area: {row['total_area_m2']} m\u00b2"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=6,
            color="#6C3BAA",
            fill_color="#A47DAB",
            fill_opacity=1,
            weight=1,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(cluster)
    fg.add_to(m)


# ---- Choropleth layers ----


def _add_choropleth(
    m: folium.Map,
    geojson: dict,
    group: str,
    prop: str,
    colormap: cm.LinearColormap,
    tooltip_fields: list[str],
    tooltip_aliases: list[str],
) -> None:
    fg = folium.FeatureGroup(name=group, show=False)

    def style_fn(
        feature: dict,
        *,
        _prop: str = prop,
        _cm: cm.LinearColormap = colormap,
    ) -> dict:
        val = feature["properties"].get(_prop, 0)
        return {
            "fillColor": _cm(val) if val else "transparent",
            "color": "black",
            "weight": 1,
            "opacity": 1,
            "fillOpacity": 0.5,
        }

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        highlight_function=lambda _: _highlight_kwargs(),
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_fields,
            aliases=tooltip_aliases,
            style=_tooltip_style_str(),
        ),
    ).add_to(fg)
    fg.add_to(m)


def _add_tier_layer(m: folium.Map, geojson: dict) -> None:
    fg = folium.FeatureGroup(name="Neighborhood Types", show=False)

    def style_fn(feature: dict) -> dict:
        tier = feature["properties"].get("tier", "")
        return {
            "fillColor": _TIER_COLORS.get(tier, "gray"),
            "color": "black",
            "weight": 1,
            "opacity": 1,
            "fillOpacity": 0.5,
        }

    folium.GeoJson(
        geojson,
        style_function=style_fn,
        highlight_function=lambda _: _highlight_kwargs(),
        tooltip=folium.GeoJsonTooltip(
            fields=["NOME", "tier"],
            aliases=["Neighborhood:", "Type:"],
            style=_tooltip_style_str(),
        ),
    ).add_to(fg)
    fg.add_to(m)


# ---- Legends ----

_LEGEND_BOX = (
    "position:fixed;bottom:{bottom}px;right:10px;z-index:1000;"
    "background:white;padding:10px;border-radius:5px;"
    "border:2px solid grey;font-size:13px;display:none"
)

_SQ = "width:12px;height:12px;display:inline-block"


def _gradient_bar(colors: list[str]) -> str:
    """Build an inline CSS gradient bar from a list of hex colours."""
    grad = ",".join(colors)
    return (
        f'<div style="width:120px;height:12px;'
        f"background:linear-gradient(to right,{grad});"
        f'border:1px solid #999;margin:4px 0"></div>'
    )


def _add_legends(
    m: folium.Map,
    pop_range: tuple[float, float],
    dens_range: tuple[float, float],
) -> None:
    pop_colors = ["#FFFFD4", "#FED98E", "#FE9929", "#D95F0E", "#993404"]
    dens_colors = ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"]

    tier_html = (
        f'<div id="legend-type" style="{_LEGEND_BOX.format(bottom=30)}">'
        "<b>Type</b><br>"
        f'<i style="background:blue;{_SQ}"></i> Low<br>'
        f'<i style="background:gold;{_SQ}"></i> Mid<br>'
        f'<i style="background:red;{_SQ}"></i> High'
        "</div>"
    )

    pop_html = (
        f'<div id="legend-pop" style="{_LEGEND_BOX.format(bottom=30)}">'
        "<b>Population</b><br>"
        f"{_gradient_bar(pop_colors)}"
        f'<span style="font-size:11px">'
        f"{pop_range[0]:,.0f} &mdash; {pop_range[1]:,.0f}"
        "</span></div>"
    )

    dens_html = (
        f'<div id="legend-dens" style="{_LEGEND_BOX.format(bottom=30)}">'
        "<b>Density</b><br>"
        f"{_gradient_bar(dens_colors)}"
        f'<span style="font-size:11px">'
        f"{dens_range[0]:,.1f} &mdash; {dens_range[1]:,.1f}"
        "</span></div>"
    )

    toggle_js = """
    <script>
    document.addEventListener("DOMContentLoaded", function() {
        var map = Object.values(window).find(
            function(v) { return v instanceof L.Map; }
        );
        if (!map) return;
        var legends = {
            "Neighborhood Types": document.getElementById("legend-type"),
            "Population": document.getElementById("legend-pop"),
            "Density": document.getElementById("legend-dens")
        };
        map.on("overlayadd", function(e) {
            var el = legends[e.name];
            if (el) el.style.display = "block";
        });
        map.on("overlayremove", function(e) {
            var el = legends[e.name];
            if (el) el.style.display = "none";
        });
    });
    </script>
    """

    root = m.get_root().html
    root.add_child(folium.Element(tier_html))
    root.add_child(folium.Element(pop_html))
    root.add_child(folium.Element(dens_html))
    root.add_child(folium.Element(toggle_js))


# ---- Tooltip style helper ----


def _tooltip_style_str() -> str:
    return "font-weight:normal;padding:3px 8px;font-size:13px"


# =====================================================================
# Public API
# =====================================================================


def build_interactive_map(
    housing_path: Path,
    classification_path: Path,
    pca_path: Path,
    shapefiles_dir: Path,
) -> folium.Map:
    """Build the interactive Curitiba housing map.

    Args:
        housing_path: Path to ``housing_geocoded.parquet``.
        classification_path: Path to ``neighborhood_classification.parquet``.
        pca_path: Path to ``pca.xlsx`` with population and density data.
        shapefiles_dir: Directory containing shapefile subdirectories.

    Returns:
        A ``folium.Map`` ready to be saved as HTML.
    """
    m = folium.Map(
        location=_CURITIBA_CENTER,
        zoom_start=_DEFAULT_ZOOM,
        tiles=None,
    )

    # Base tile layers (OpenStreetMap added first = default selected)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Esri.WorldImagery",
        show=False,
    ).add_to(m)

    # Load shapefiles
    shp = {
        key: _read_shapefile(shapefiles_dir / sub / fname)
        for key, (sub, fname) in _SHAPEFILES.items()
    }

    # Classification and PCA data
    shp["neighborhoods"] = _enrich_neighborhoods(
        shp["neighborhoods"],
        classification_path,
        pca_path,
    )

    # Compute ranges for colormaps
    pops = [f["properties"]["pop"] for f in shp["neighborhoods"]["features"]]
    denss = [f["properties"]["dens"] for f in shp["neighborhoods"]["features"]]
    pop_range = (min(pops), max(pops))
    dens_range = (min(denss), max(denss))
    pop_cm = cm.LinearColormap(
        ["#FFFFD4", "#FED98E", "#FE9929", "#D95F0E", "#993404"],
        vmin=pop_range[0],
        vmax=pop_range[1],
    )
    dens_cm = cm.LinearColormap(
        ["#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"],
        vmin=dens_range[0],
        vmax=dens_range[1],
    )

    # Neighborhoods boundary
    _add_neighborhoods(m, shp["neighborhoods"])

    # Green Area (parks + squares)
    fg_green = folium.FeatureGroup(name="Green Area", show=False)
    for key, tt_fields, tt_aliases in [
        ("parks", ["TEXTO_MAPA", "TIPO"], ["Name:", "Type:"]),
        ("squares", ["NOME", "TIPO"], ["Name:", "Type:"]),
    ]:
        folium.GeoJson(
            shp[key],
            style_function=lambda _: {
                "fillColor": "green",
                "color": "transparent",
                "fillOpacity": 0.7,
                "stroke": False,
            },
            highlight_function=lambda _: _highlight_kwargs(color="darkgreen", weight=3),
            tooltip=folium.GeoJsonTooltip(
                fields=tt_fields,
                aliases=tt_aliases,
                style=_tooltip_style_str(),
            ),
        ).add_to(fg_green)
    fg_green.add_to(m)

    # Hydrography
    fg_hydro = folium.FeatureGroup(name="Hydrography", show=False)
    folium.GeoJson(
        shp["lakes"],
        style_function=lambda _: {
            "fillColor": "blue",
            "color": "transparent",
            "fillOpacity": 0.7,
            "stroke": False,
        },
        highlight_function=lambda _: _highlight_kwargs(color="darkblue", weight=3),
        tooltip=folium.GeoJsonTooltip(
            fields=["TIPO"],
            aliases=["Type:"],
            style=_tooltip_style_str(),
        ),
    ).add_to(fg_hydro)
    folium.GeoJson(
        shp["rivers"],
        style_function=lambda _: {"color": "blue", "weight": 1.5, "opacity": 0.7},
        highlight_function=lambda _: _highlight_kwargs(color="darkblue", weight=5),
        tooltip=folium.GeoJsonTooltip(
            fields=["TIPO"],
            aliases=["Type:"],
            style=_tooltip_style_str(),
        ),
    ).add_to(fg_hydro)
    fg_hydro.add_to(m)

    # Railroads
    _add_polyline_layer(
        m,
        shp["railroads"],
        "Railroads",
        "purple",
        tooltip_fields=["TEXTO", "TIPO"],
        tooltip_aliases=["Name:", "Type:"],
    )

    # Cycling
    fg_cycling = folium.FeatureGroup(name="Cycling", show=False)
    for key in ("cicleways", "cicleroutes"):
        folium.GeoJson(
            shp[key],
            style_function=lambda _: {"color": "red", "weight": 1.5, "opacity": 0.7},
            highlight_function=lambda _: _highlight_kwargs(weight=5),
            tooltip=folium.GeoJsonTooltip(
                fields=["TIPO"],
                aliases=["Type:"],
                style=_tooltip_style_str(),
            ),
        ).add_to(fg_cycling)
    fg_cycling.add_to(m)

    # Irregular Settlements
    _add_polygon_layer(
        m,
        shp["irregular"],
        "Irregular Settlements",
        "darkorange",
        tooltip_fields=["NOME", "CATEG_2000"],
        tooltip_aliases=["Name:", "Type:"],
    )

    # Hospitals and Clinics
    fg_health = folium.FeatureGroup(name="Hospitals & Clinics", show=False)
    for key in ("hospitals", "dentists", "med_units", "emergency"):
        _add_marker_layer(
            m,
            shp[key],
            "Hospitals & Clinics",
            "NOME_COMPL",
            "plus",
            "red",
            fg=fg_health,
        )

    # Cemiteries
    _add_polygon_layer(
        m,
        shp["cemiteries"],
        "Cemiteries",
        "#5A5A5A",
        tooltip_fields=["NOME"],
        tooltip_aliases=["Name:"],
    )

    # Bus Terminals
    _add_marker_layer(
        m,
        shp["terminals"],
        "Bus Terminals",
        "NOME_COMPL",
        "bus",
        "blue",
        prefix="fa",
    )

    # Public Schools
    _add_marker_layer(
        m,
        shp["schools"],
        "Public Schools",
        "NOME_MAPA",
        "book",
        "orange",
    )

    # Cultural Centers
    _add_marker_layer(
        m,
        shp["cidadania"],
        "Cultural Centers",
        "NOME_MAPA",
        "star",
        "pink",
    )

    # Sport Centers
    _add_marker_layer(
        m,
        shp["sport"],
        "Sport Centers",
        "NOME_MAPA",
        "futbol",
        "lightgreen",
        prefix="fa",
    )

    # Housing Listings
    housing_df = pd.read_parquet(housing_path)
    _add_housing(m, housing_df)

    # Population choropleth
    _add_choropleth(
        m,
        shp["neighborhoods"],
        "Population",
        "pop",
        pop_cm,
        tooltip_fields=["NOME", "pop"],
        tooltip_aliases=["Neighborhood:", "Population:"],
    )

    # Density choropleth
    _add_choropleth(
        m,
        shp["neighborhoods"],
        "Density",
        "dens",
        dens_cm,
        tooltip_fields=["NOME", "dens"],
        tooltip_aliases=["Name:", "Density:"],
    )

    # Choropleth: Neighborhood Types
    _add_tier_layer(m, shp["neighborhoods"])

    # Legends
    _add_legends(m, pop_range, dens_range)

    # Layer control and measure tool
    folium.LayerControl(collapsed=True, position="topleft").add_to(m)
    folium.plugins.MeasureControl(
        primary_length_unit="meters",
        primary_area_unit="sqmeters",
        position="bottomleft",
    ).add_to(m)

    return m

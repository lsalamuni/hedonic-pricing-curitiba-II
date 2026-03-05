"""Merge location attributes and tier classification onto housing data.

The public function ``merge_location_attributes`` joins neighbourhood-level
census variables (population, density, amenity indicators) and the PCA-based
tier classification onto the geocoded housing DataFrame, producing one
DataFrame per market tier ready for the Rosen regression stage.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_LOCATION_VARS: list[str] = [
    "population",
    "density",
    "cicloways",
    "hospitals",
    "terminals",
    "private_schools",
    "public_schools",
    "culture_facilities",
    "green_area",
    "shoppings",
]


# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _load_info_neighborhoods(path: Path) -> pd.DataFrame:
    """Load neighbourhood attribute table and normalise column names."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def _load_classification(path: Path) -> pd.DataFrame:
    """Load PCA-based neighbourhood classification."""
    return pd.read_parquet(path)


def _normalise_neighborhood_name(series: pd.Series) -> pd.Series:
    """Upper-case and strip whitespace for join consistency."""
    return series.str.strip().str.upper()


def _add_apartment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Infer Apartment flag from the ``category`` column if present.

    The cleaned housing data dropped the category column but kept the
    lowercase column names. If a ``category`` column exists, create a
    binary ``apartment`` column (1 = Apartamento, 0 = Casa). Otherwise
    assume the column already exists or is not needed.
    """
    if "category" in df.columns:
        df = df.copy()
        df["apartment"] = (df["category"].str.lower() == "apartamento").astype(int)
    return df


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def merge_location_attributes(
    housing_df: pd.DataFrame,
    info_path: Path,
    classification_path: Path,
) -> dict[str, pd.DataFrame]:
    """Merge location attributes and tier classification onto housing data.

    Args:
        housing_df: Geocoded housing DataFrame with a ``neighborhood`` column.
        info_path: Path to ``info_neighborhoods.xlsx``.
        classification_path: Path to ``neighborhood_classification.parquet``.

    Returns:
        Dict with keys ``"low"``, ``"mid"``, ``"high"`` mapping to the
        filtered DataFrames for each market tier, and ``"all"`` for the
        full merged dataset.

    """
    info = _load_info_neighborhoods(info_path)
    classification = _load_classification(classification_path)

    # Normalise join keys
    info["neighborhood"] = _normalise_neighborhood_name(info["neighborhood"])
    classification["loc"] = _normalise_neighborhood_name(classification["loc"])

    # Normalise housing neighbourhood names for join
    df = housing_df.copy()
    df["_join_key"] = _normalise_neighborhood_name(df["neighborhood"])

    # Merge tier classification
    tier_map = classification[["loc", "tier"]].rename(columns={"loc": "_join_key"})
    df = df.merge(tier_map, on="_join_key", how="left")

    # Merge location attributes
    info_renamed = info.rename(columns={"neighborhood": "_join_key"})
    df = df.merge(info_renamed, on="_join_key", how="left")

    # Drop join key
    df = df.drop(columns=["_join_key"])

    # Drop rows with missing tier (neighbourhood not in classification)
    df = df.dropna(subset=["tier"])

    # Add apartment flag if needed
    df = _add_apartment_flag(df)

    # Split by tier
    result = {"all": df}
    for tier in ("low", "mid", "high"):
        result[tier] = df[df["tier"] == tier].copy().reset_index(drop=True)

    return result

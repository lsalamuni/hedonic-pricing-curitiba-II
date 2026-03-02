"""Geocode housing addresses using OpenStreetMap and ArcGIS.

The public function ``geocode_housing`` takes the cleaned housing DataFrame,
filters out outliers, geocodes addresses in two stages (Nominatim then ArcGIS
fallback), and returns the DataFrame with latitude/longitude columns added.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from geopy.geocoders import ArcGIS, Nominatim

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #

_CITY_SUFFIX: str = "Curitiba, Paraná, Brazil"
_USER_AGENT: str = "hedonic_analysis_geocoder"

_NOMINATIM_BATCH_SIZE: int = 50
_ARCGIS_BATCH_SIZE: int = 25
_NOMINATIM_DELAY: float = 2.0
_ARCGIS_DELAY: float = 3.0

# ------------------------------------------------------------------ #
# Private helpers
# ------------------------------------------------------------------ #


def _build_full_address(df: pd.DataFrame) -> pd.Series:
    """Concatenate address, neighborhood, and city into a full address."""
    return (
        df["address"].astype(str) + ", "
        + df["neighborhood"].astype(str) + ", "
        + _CITY_SUFFIX
    )


def _load_cache(path: Path | None) -> pd.DataFrame:
    """Load geocoding cache from parquet, or return an empty DataFrame."""
    if path is not None and path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame({
        "id": pd.Series(dtype="int64"),
        "latitude": pd.Series(dtype="float64"),
        "longitude": pd.Series(dtype="float64"),
    })


def _save_cache(cache_df: pd.DataFrame, path: Path | None) -> None:
    """Persist geocoding cache to parquet."""
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache_df.to_parquet(path, index=False)


def _geocode_with_service(
    geocoder,
    addresses: pd.Series,
    ids: pd.Series,
    batch_size: int,
    delay: float,
    cache_df: pd.DataFrame,
    cache_path: Path | None,
    stage_name: str,
) -> pd.DataFrame:
    """Geocode addresses in batches using the given geocoder.

    Updates the cache after each batch for crash recovery.
    Returns the updated cache DataFrame.
    """
    n_total = len(addresses)
    if n_total == 0:
        return cache_df

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        batch_addrs = addresses.iloc[start:end]
        batch_ids = ids.iloc[start:end]

        lats = []
        lons = []
        for addr in batch_addrs:
            try:
                location = geocoder.geocode(addr, timeout=10)
            except Exception:
                location = None

            if location is not None:
                lats.append(location.latitude)
                lons.append(location.longitude)
            else:
                lats.append(None)
                lons.append(None)
            time.sleep(delay)

        batch_result = pd.DataFrame({
            "id": batch_ids.to_numpy(),
            "latitude": lats,
            "longitude": lons,
        })

        successful = batch_result.dropna(subset=["latitude"])
        if not successful.empty:
            cache_df = pd.concat(
                [cache_df[~cache_df["id"].isin(successful["id"])], successful],
                ignore_index=True,
            )
            _save_cache(cache_df, cache_path)

        geocoded_so_far = cache_df["latitude"].notna().sum()
        print(
            f"  [{stage_name}] Rows {start + 1} to {end} of {n_total}. "
            f"Progress: {geocoded_so_far} geocoded successfully."
        )

    return cache_df


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def geocode_housing(
    df: pd.DataFrame,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Geocode housing addresses and add latitude/longitude columns.

    Args:
        df: Cleaned housing DataFrame with ``address``, ``neighborhood``,
            and ``outlier`` columns.
        cache_path: Path for the geocoding cache file. If provided, already
            geocoded addresses are skipped on re-runs.

    Returns:
        Filtered DataFrame (outliers removed) with ``latitude`` and
        ``longitude`` columns added.

    """
    # Filter outliers
    df = df[df["outlier"] == 0].copy()
    df = df.reset_index(drop=True)

    # Build full address
    full_address = _build_full_address(df)

    # Load cache
    cache_df = _load_cache(cache_path)

    # Identify uncached rows
    cached_ids = set(cache_df["id"].tolist())
    mask_uncached = ~df["id"].isin(cached_ids)
    uncached_addrs = full_address[mask_uncached]
    uncached_ids = df.loc[mask_uncached, "id"]

    # Stage 1: Nominatim (OSM)
    if not uncached_addrs.empty:
        print(
            f"Stage 1 (Nominatim): {len(uncached_addrs)} addresses to geocode."
        )
        nominatim = Nominatim(user_agent=_USER_AGENT)
        cache_df = _geocode_with_service(
            geocoder=nominatim,
            addresses=uncached_addrs,
            ids=uncached_ids,
            batch_size=_NOMINATIM_BATCH_SIZE,
            delay=_NOMINATIM_DELAY,
            cache_df=cache_df,
            cache_path=cache_path,
            stage_name="Nominatim",
        )

    # Stage 2: ArcGIS fallback for failed addresses
    cached_ids = set(cache_df["id"].tolist())
    mask_still_missing = ~df["id"].isin(cached_ids)
    missing_addrs = full_address[mask_still_missing]
    missing_ids = df.loc[mask_still_missing, "id"]

    if not missing_addrs.empty:
        print(
            f"Stage 2 (ArcGIS): {len(missing_addrs)} addresses to retry."
        )
        arcgis = ArcGIS()
        cache_df = _geocode_with_service(
            geocoder=arcgis,
            addresses=missing_addrs,
            ids=missing_ids,
            batch_size=_ARCGIS_BATCH_SIZE,
            delay=_ARCGIS_DELAY,
            cache_df=cache_df,
            cache_path=cache_path,
            stage_name="ArcGIS",
        )

    # Merge coordinates onto DataFrame
    coords = cache_df[["id", "latitude", "longitude"]]
    result = df.merge(coords, on="id", how="left")

    total = len(result)
    geocoded = result["latitude"].notna().sum()
    failed = total - geocoded
    print(f"Done! Geocoded {geocoded} of {total} addresses.")
    if failed > 0:
        print(f"Failed: {failed} addresses.")

    return result

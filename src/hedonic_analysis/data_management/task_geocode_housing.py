"""Task for geocoding cleaned housing addresses."""

import pandas as pd

from hedonic_analysis.config import BLD_DATA, SRC
from hedonic_analysis.data_management.geocode_housing import geocode_housing


def task_geocode_housing(
    script=SRC / "data_management" / "geocode_housing.py",
    data=BLD_DATA / "housing_cleaned.parquet",
    produces=BLD_DATA / "housing_geocoded.parquet",
):
    """Geocode housing addresses and save as parquet and xlsx."""
    df = pd.read_parquet(data)
    cache_path = SRC / "data" / "geocode_cache.parquet"
    geocoded = geocode_housing(df, cache_path=cache_path)
    geocoded.to_parquet(produces)
    geocoded.to_excel(produces.with_suffix(".xlsx"), index=False)

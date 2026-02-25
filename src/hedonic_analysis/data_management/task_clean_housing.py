"""Task for cleaning raw ImovelWeb housing data."""

import pandas as pd

from hedonic_analysis.config import BLD_DATA, SRC
from hedonic_analysis.data_management.clean_housing import (
    clean_housing,
)


def task_clean_housing(
    script=SRC / "data_management" / "clean_housing.py",
    data=SRC / "data" / "imovelweb_raw.csv",
    produces=BLD_DATA / "housing_cleaned.parquet",
):
    """Clean raw scraped data and save as parquet."""
    raw = pd.read_csv(data, encoding="utf-8-sig")
    cleaned = clean_housing(raw)
    cleaned.to_parquet(produces)

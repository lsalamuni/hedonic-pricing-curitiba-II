"""Task for merging location attributes onto geocoded housing data."""

import pandas as pd

from hedonic_analysis.config import BLD_DATA, SRC
from hedonic_analysis.data_management.merge_location_attributes import (
    merge_location_attributes,
)


def task_merge_location_attributes(
    script=SRC / "data_management" / "merge_location_attributes.py",
    housing=BLD_DATA / "housing_geocoded.parquet",
    info=SRC / "data" / "info_neighborhoods.xlsx",
    classification=BLD_DATA / "neighborhood_classification.parquet",
    produces=BLD_DATA / "housing_merged.parquet",
):
    """Merge location attributes and tier labels, then save per-tier files."""
    housing_df = pd.read_parquet(housing)
    tiers = merge_location_attributes(housing_df, info, classification)

    tiers["all"].to_parquet(produces)
    tiers["all"].to_excel(produces.with_suffix(".xlsx"), index=False)

    for tier_name in ("low", "mid", "high"):
        tier_path = BLD_DATA / f"{tier_name}_tier.parquet"
        tiers[tier_name].to_parquet(tier_path)
        tiers[tier_name].to_excel(tier_path.with_suffix(".xlsx"), index=False)

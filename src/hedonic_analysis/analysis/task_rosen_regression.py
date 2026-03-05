"""Task for running Rosen's two-stage hedonic regression analysis."""

import pandas as pd

from hedonic_analysis.analysis.rosen_regression import run_rosen_analysis
from hedonic_analysis.config import BLD_ANALYSIS, BLD_DATA, BLD_IMAGES, SRC


def task_rosen_regression(
    script=SRC / "analysis" / "rosen_regression.py",
    merged=BLD_DATA / "housing_merged.parquet",
    low=BLD_DATA / "low_tier.parquet",
    mid=BLD_DATA / "mid_tier.parquet",
    high=BLD_DATA / "high_tier.parquet",
    produces=BLD_DATA / "first_stage_results.parquet",
):
    """Run Rosen's two-stage hedonic regression on all market tiers."""
    tiers = {
        "low": pd.read_parquet(low),
        "mid": pd.read_parquet(mid),
        "high": pd.read_parquet(high),
    }
    run_rosen_analysis(tiers, BLD_DATA, BLD_ANALYSIS, BLD_IMAGES)

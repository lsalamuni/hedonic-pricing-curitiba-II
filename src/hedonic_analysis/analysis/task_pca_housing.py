"""Task for running PCA analysis on Curitiba's neighborhoods."""

from hedonic_analysis.analysis.pca_housing import run_pca_analysis
from hedonic_analysis.config import BLD_ANALYSIS, BLD_DATA, BLD_IMAGES, SRC


def task_pca_housing(
    script=SRC / "analysis" / "pca_housing.py",
    data=SRC / "data" / "pca.xlsx",
    produces=BLD_DATA / "neighborhood_classification.parquet",
):
    """Run PCA on census data and produce stratification outputs."""
    run_pca_analysis(data, BLD_DATA, BLD_ANALYSIS, BLD_IMAGES)

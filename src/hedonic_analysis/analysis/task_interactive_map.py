"""Task for building the interactive Curitiba housing map."""

from hedonic_analysis.analysis.interactive_map import build_interactive_map
from hedonic_analysis.config import BLD, BLD_DATA, SRC


def task_interactive_map(
    script=SRC / "analysis" / "interactive_map.py",
    geocoded=BLD_DATA / "housing_geocoded.parquet",
    classification=BLD_DATA / "neighborhood_classification.parquet",
    pca=SRC / "data" / "pca.xlsx",
    produces=BLD / "interactive_map.html",
):
    """Build interactive HTML map from geocoded housing and shapefiles."""
    m = build_interactive_map(
        housing_path=geocoded,
        classification_path=classification,
        pca_path=pca,
        shapefiles_dir=SRC / "data",
    )
    m.save(str(produces))

"""Task for building the interactive Curitiba housing map."""

from hedonic_analysis.config import BLD_DATA, BLD_FINAL, SRC
from hedonic_analysis.final.interactive_map import build_interactive_map


def task_interactive_map(
    script=SRC / "final" / "interactive_map.py",
    geocoded=BLD_DATA / "housing_geocoded.parquet",
    classification=BLD_DATA / "neighborhood_classification.parquet",
    pca=SRC / "data" / "pca.xlsx",
    produces=BLD_FINAL / "interactive_map.html",
):
    """Build interactive HTML map from geocoded housing and shapefiles."""
    m = build_interactive_map(
        housing_path=geocoded,
        classification_path=classification,
        pca_path=pca,
        shapefiles_dir=SRC / "data",
    )
    m.save(str(produces))

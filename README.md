# Pricing the Unpriced: A Hedonic Analysis of Curitiba's Segmented Real Estate Market

## Summary

This project applies Rosen's (1974) two-stage hedonic pricing model to Curitiba's
residential real estate market. Using 6,833 web-scraped property listings and
neighborhood-level socioeconomic data from IPPUC, it estimates how the implicit prices
of housing attributes vary across PCA-stratified market segments. A Wald test (F =
163.02, p < 0.001) confirms that hedonic coefficients differ significantly across low-,
middle-, and high-income tiers, validating the segmented estimation approach. Conley
heteroskedasticity and autocorrelation consistent standard errors address spatial
dependence.

## Reproducing the Analysis

The only prerequisite is [Pixi](https://pixi.sh/), which handles all dependencies
(Python, conda packages, and pip packages).

```bash
# 1. Install Pixi (https://pixi.sh/)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and enter the repository
git clone https://github.com/iame-uni-bonn/final-project-lsalamuni.git
cd final-project-lsalamuni

# 3. Run the full pipeline (data cleaning → PCA → regressions → map → paper)
pixi run pytask

# 4. Run the test suite
pixi run pytest

# 5. Preview the paper in your browser (live reload)
pixi run view-paper
```

### Useful Commands

| Command                               | Description                               |
| ------------------------------------- | ----------------------------------------- |
| `pixi run pytask`                     | Run the full computational pipeline       |
| `pixi run pytest`                     | Run all 106 tests                         |
| `pixi run pytest -m unit`             | Run only unit tests                       |
| `pixi run pytest -m "not end_to_end"` | Skip the slow pipeline test               |
| `pixi run view-paper`                 | Preview the paper (HTML with live reload) |

## Project Structure

```
final-project-lsalamuni/
├── src/hedonic_analysis/         # Source code
│   ├── config.py                 # Central path definitions and constants
│   ├── data_management/          # Stage 1: data cleaning and preparation
│   ├── analysis/                 # Stage 2: PCA and hedonic regressions
│   ├── final/                    # Stage 3: publication-ready outputs
│   └── data/                     # Raw input data (CSV, Excel, shapefiles)
│
├── tests/                        # Test suite (mirrors src/ structure)
│   ├── data_management/          # Unit tests for cleaning and geocoding
│   ├── analysis/                 # Unit tests for PCA and regressions
│   ├── final/                    # Unit and integration tests for the map
│   └── test_pipeline.py          # End-to-end pipeline test
│
├── documents/                    # Paper source files
│   ├── paper.md                  # Paper (MyST Markdown)
│   ├── refs.bib                  # Bibliography
│   └── task_documents.py         # pytask task for PDF compilation
│
├── bld/                          # Pipeline outputs (generated, not tracked)
│   ├── data/                     # Cleaned and processed datasets
│   ├── analysis/                 # LaTeX tables (.tex)
│   ├── images/                   # Figures (.png)
│   └── final/                    # Paper PDF and interactive map HTML
│
├── pyproject.toml                # Project configuration (Pixi, Ruff, pytest)
└── myst.yml                      # Jupyter Book 2.0 / MyST configuration
```

## Pipeline Stages

The pipeline is orchestrated by [pytask](https://pytask-dev.readthedocs.io/). Each stage
has a `task_*.py` file that defines the computational step and a companion module that
contains the logic.

### Stage 1: Data Management (`src/hedonic_analysis/data_management/`)

| Task                                | Module                         | Description                                                                                                   |
| ----------------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| `task_clean_housing.py`             | `clean_housing.py`             | Parse raw listings: extract prices, neighborhoods, amenities, bedroom/bathroom/parking dummies; flag outliers |
| `task_geocode_housing.py`           | `geocode_housing.py`           | Geocode addresses to latitude/longitude using Nominatim with ArcGIS fallback and persistent cache             |
| `task_merge_location_attributes.py` | `merge_location_attributes.py` | Merge neighborhood socioeconomic data and locational attributes; stratify into low/mid/high tiers             |

**Inputs:** `imovelweb_raw.csv`, `info_neighborhoods.xlsx`, `pca.xlsx`,
`neighborhood_classification.parquet`

**Outputs:** `housing_cleaned.parquet`, `housing_geocoded.parquet`,
`housing_merged.parquet`, `{low,mid,high}_tier.parquet`

### Stage 2: Analysis (`src/hedonic_analysis/analysis/`)

| Task                       | Module                | Description                                                                                                                                                                                       |
| -------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_pca_housing.py`      | `pca_housing.py`      | PCA on six neighborhood socioeconomic variables; Kaiser criterion; weighted composite score; neighborhood classification into three tiers                                                         |
| `task_rosen_regression.py` | `rosen_regression.py` | Two-stage hedonic regressions per tier: first-stage log-log price functions, Box-Cox validation, implicit price extraction, second-stage supply equations with Conley spatial HAC standard errors |

**Outputs (data):** `neighborhood_classification.parquet`, `factor_loadings.parquet`,
`pca_scores.parquet`, `first_stage_results.parquet`, `second_stage_results.parquet`,
`diagnostics_summary.parquet`

**Outputs (figures):** `scree_plot.png`, `correlation_heatmap.png`,
`loadings_heatmap.png`, `pca_biplot.png`, `boxcox_{tier}.png`,
`residuals_first_stage_{tier}.png`, `residuals_second_stage_{tier}.png`

**Outputs (tables):** `adequacy_tests.tex`, `first_stage_paper.tex`,
`second_stage_paper.tex`, `diagnostics_summary.tex`, and per-tier variants

### Stage 3: Final (`src/hedonic_analysis/final/`)

| Task                      | Module               | Description                                                                                                                                                                                             |
| ------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `task_interactive_map.py` | `interactive_map.py` | Build an interactive Folium map with geocoded listings, neighborhood boundaries, choropleth layers (population, density, tier classification), and 18 urban infrastructure layers from IPPUC shapefiles |

**Output:** `interactive_map.html`

### Documents (`documents/`)

| Task                | Description                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `task_documents.py` | Inject generated LaTeX tables into the paper, compile to PDF via Jupyter Book 2.0, copy figures to `documents/public/` |

**Output:** `paper.pdf`

## Input Data

### Property Listings

`imovelweb_raw.csv` contains 6,833 residential property listings scraped from Brazilian
real estate platforms, with fields including listing price, total area, number of
bedrooms/bathrooms/parking spots, property description, and address.

### Neighborhood Data

| File                      | Description                                                                                                           |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `pca.xlsx`                | Six socioeconomic variables per neighborhood (income, poverty rate, literacy, density, growth, population) from IPPUC |
| `info_neighborhoods.xlsx` | Locational attributes (hospitals, schools, shopping centers, green areas, transit terminals)                          |
| `green_area.xlsx`         | Green area presence per neighborhood                                                                                  |
| `private_schools.xlsx`    | Private school presence per neighborhood                                                                              |
| `shopping_centers.xlsx`   | Shopping center presence per neighborhood                                                                             |
| `geocode_cache.parquet`   | Persistent geocoding cache to avoid repeated API calls                                                                |

### IPPUC Shapefiles

The `data/` directory contains 18 shapefile subdirectories from IPPUC (Curitiba's urban
planning institute), used for the interactive map:

Neighborhoods, Parks, Squares, Cicleways, Cicleroutes, Railroads, Sport/leisure centers,
Cemiteries, Cidadania (cultural centers), Bus terminals, Dentist units, Emergency units,
Health units, Hospitals, Medical units, Irregular settlements, Lakes, Rivers, Schools.

## Testing

The test suite contains 106 tests organized in three levels:

| Marker        | Count | Description                                                                                       |
| ------------- | ----- | ------------------------------------------------------------------------------------------------- |
| `unit`        | 95    | Fast tests for individual functions (parsing, extraction, PCA helpers, regression utilities)      |
| `integration` | 10    | Tests that combine multiple components with file I/O (map building, data merging with real files) |
| `end_to_end`  | 1     | Full pipeline test via `pytask.build()` in a temporary directory                                  |

This project builds on the template by
[von Gaudecker (2019)](https://doi.org/10.5281/zenodo.2533241).

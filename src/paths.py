"""Project paths and map-review constants.

All published maps use the standard administrative boundaries carrying
审图号 GS（2024）0650号. No other national, provincial, or coastline
dataset is a geometry source.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_V10 = DATA / "raw" / "v10" / "SOCS_V10.csv"
README_V10 = DATA / "raw" / "v10" / "SOCS_V10_Readme.txt"
RAW_V9 = DATA / "raw" / "v9" / "China_profile_dataset_20250710.csv"
BOUND_DIR = DATA / "boundaries" / "gs2024_0650"
PROCESSED = DATA / "processed"
GEE_DIR = DATA / "raw" / "gee_covariates"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
METRICS = OUT / "metrics"
RASTERS = OUT / "rasters"
MANIFESTS = OUT / "manifests"

GEE_PROJECT = "hallowed-chain-458005-g9"
MAP_REVIEW_ID = "GS（2024）0650号"
MAP_REVIEW_LABEL = "审图号：GS（2024）0650号"

# China Albers equal-area on the WGS84 datum of the official shapefiles.
CHINA_ALBERS = (
    "+proj=aea +lat_1=25 +lat_2=47 +lat_0=0 +lon_0=105 "
    "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
)

PROVINCE_SHP = BOUND_DIR / "中国_省.shp"
PROVINCE_LINE_SHP = BOUND_DIR / "中国_省line.shp"
CITY_SHP = BOUND_DIR / "中国_市.shp"
COUNTY_SHP = BOUND_DIR / "中国_县.shp"

for _path in (PROCESSED, GEE_DIR, TABLES, FIGURES, METRICS, RASTERS, MANIFESTS):
    _path.mkdir(parents=True, exist_ok=True)

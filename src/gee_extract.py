"""Download real covariates from Google Earth Engine.

The computation region is the bounding rectangle of the official
GS（2024）0650 polygons and 境界线. Province polygons themselves are
applied locally; Earth Engine is not given a foreign country boundary.

Predictors
    SRTM elevation, slope, aspect, 2 km TPI, 500 m relief
    ERA5-Land monthly temperature, PET and 0–7 cm soil moisture, 2015–2022
    CHIRPS daily precipitation, 2015–2022
    MODIS MOD13Q1 NDVI percentiles, 2015–2022
    ESA WorldCover 2021

OpenLandMap SOC and bulk density are sampled only for an external
product comparison. They are not model inputs.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests

from .boundaries import official_bounds
from .paths import GEE_DIR, GEE_PROJECT, MANIFESTS, PROCESSED, RASTERS

START = "2015-01-01"
END = "2023-01-01"
N_YEARS = 8


def initialize() -> None:
    ee.Initialize(project=GEE_PROJECT)


def roi_rectangle() -> ee.Geometry:
    minx, miny, maxx, maxy = official_bounds()
    return ee.Geometry.Rectangle([minx - 0.1, miny - 0.1, maxx + 0.1, maxy + 0.1])


def terrain_image(roi: ee.Geometry) -> ee.Image:
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation").clip(roi).rename("elevation").toFloat()
    slope = ee.Terrain.slope(dem).rename("slope")
    aspect = ee.Terrain.aspect(dem).multiply(np.pi / 180.0)
    tpi = dem.subtract(dem.focal_mean(2000, "circle", "meters")).rename("tpi_2km")
    relief = dem.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.circle(2000, "meters")).rename("relief_2km")
    return ee.Image.cat(
        [
            dem,
            slope,
            aspect.sin().rename("aspect_sin"),
            aspect.cos().rename("aspect_cos"),
            tpi,
            relief,
        ]
    ).toFloat()


def climate_image(roi: ee.Geometry) -> ee.Image:
    era = (
        ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        .filterDate(START, END)
        .filterBounds(roi)
        .select(["temperature_2m", "potential_evaporation_sum", "volumetric_soil_water_layer_1"])
    )

    def scale_month(img: ee.Image) -> ee.Image:
        temp_c = img.select("temperature_2m").subtract(273.15).rename("t_c")
        pet_mm = img.select("potential_evaporation_sum").abs().multiply(1000.0).rename("pet_mm")
        soil = img.select("volumetric_soil_water_layer_1").rename("soil_vwc")
        return temp_c.addBands(pet_mm).addBands(soil).copyProperties(img, ["system:time_start"])

    monthly = era.map(scale_month)
    mat = monthly.select("t_c").mean().rename("mat_c")
    pet = monthly.select("pet_mm").mean().multiply(12.0).rename("pet_mm")
    soil = monthly.select("soil_vwc").mean().rename("soil_vwc")

    def month_mean(month) -> ee.Image:
        month = ee.Number(month)
        return (
            monthly.filter(ee.Filter.calendarRange(month, month, "month"))
            .select("t_c")
            .mean()
            .set("month", month)
        )

    climatology = ee.ImageCollection(ee.List.sequence(1, 12).map(month_mean))
    tseason = climatology.reduce(ee.Reducer.stdDev()).rename("tseason_c")
    trange = climatology.max().subtract(climatology.min()).rename("trange_c")
    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterDate(START, END)
        .filterBounds(roi)
        .select("precipitation")
        .sum()
        .divide(N_YEARS)
        .rename("map_mm")
    )
    return ee.Image.cat([mat, tseason, trange, pet, soil, chirps]).toFloat()


def optical_image(roi: ee.Geometry) -> ee.Image:
    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(START, END)
        .filterBounds(roi)
        .select("NDVI")
        .map(lambda img: img.multiply(0.0001).updateMask(img.gte(0)).updateMask(img.lte(10000)).toFloat())
    )
    mean = ndvi.mean().rename("ndvi_mean")
    std = ndvi.reduce(ee.Reducer.stdDev()).rename("ndvi_std")
    perc = ndvi.reduce(ee.Reducer.percentile([10, 50, 90])).rename(["ndvi_p10", "ndvi_p50", "ndvi_p90"])
    worldcover = (
        ee.ImageCollection("ESA/WorldCover/v200").filterBounds(roi).first().select("Map").rename("worldcover")
    )
    return ee.Image.cat([mean, std, perc, worldcover]).toFloat()


def openlandmap_image() -> ee.Image:
    soc = ee.Image("OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02").multiply(5.0)
    bd = ee.Image("OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02").multiply(0.01)
    depth = ["0", "10", "30", "60", "100", "200"]
    soc = soc.rename([f"olm_soc_{d}" for d in depth])
    bd = bd.rename([f"olm_bd_{d}" for d in depth])
    return soc.addBands(bd).toFloat()


def _profiles() -> pd.DataFrame:
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    use = profiles.loc[profiles["in_training_pool"]].copy()
    if use.empty:
        raise RuntimeError("No training profiles. Run QC first.")
    return use[["profile_id", "latitude", "longitude"]].drop_duplicates("profile_id")


def _feature_collection(frame: pd.DataFrame) -> ee.FeatureCollection:
    features = []
    for row in frame.itertuples(index=False):
        features.append(
            ee.Feature(
                ee.Geometry.Point([float(row.longitude), float(row.latitude)]),
                {"profile_id": str(row.profile_id)},
            )
        )
    return ee.FeatureCollection(features)


def _get_info(obj, attempts: int = 4):
    last = None
    for attempt in range(attempts):
        try:
            return obj.getInfo()
        except Exception as exc:  # noqa: BLE001 - network/compute retries
            last = exc
            time.sleep(8 * (attempt + 1))
    raise RuntimeError(f"Earth Engine request failed: {last}") from last


def _sample_frame(image: ee.Image, frame: pd.DataFrame, scale: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    sampled = image.sampleRegions(
        collection=_feature_collection(frame),
        properties=["profile_id"],
        scale=scale,
        geometries=False,
        tileScale=8,
    )
    payload = _get_info(sampled)
    rows = []
    for feature in payload.get("features", []):
        props = feature.get("properties", {})
        rows.append(props)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["profile_id"] = out["profile_id"].astype(str)
    return out


def _sample_resilient(image: ee.Image, frame: pd.DataFrame, scale: int, label: str) -> pd.DataFrame:
    try:
        started = time.time()
        out = _sample_frame(image, frame, scale)
        print(f"  {label}: {len(frame)} profiles in {time.time() - started:.1f}s", flush=True)
        return out
    except Exception as exc:
        if len(frame) <= 25:
            raise
        print(f"  {label} split after failure ({exc})", flush=True)
        mid = len(frame) // 2
        left = _sample_resilient(image, frame.iloc[:mid].copy(), scale, label + "a")
        right = _sample_resilient(image, frame.iloc[mid:].copy(), scale, label + "b")
        return pd.concat([left, right], ignore_index=True)


def extract_profiles(batch_size: int = 200, limit: int | None = None) -> Path:
    initialize()
    roi = roi_rectangle()
    profiles = _profiles()
    root = GEE_DIR if limit is None else GEE_DIR / "pilot"
    root.mkdir(parents=True, exist_ok=True)
    if limit is not None:
        profiles = profiles.head(limit).copy()
    stacks = {
        "terrain": (terrain_image(roi), 1000),
        "climate": (climate_image(roi), 4000),
        "optical": (optical_image(roi), 1000),
        "openlandmap": (openlandmap_image(), 250),
    }
    parts = []
    for name, (image, scale) in stacks.items():
        batch_dir = root / name
        batch_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        n_batch = int(np.ceil(len(profiles) / batch_size))
        for i in range(n_batch):
            dest = batch_dir / f"batch_{i:04d}.csv"
            if dest.exists() and limit is None:
                frames.append(pd.read_csv(dest, dtype={"profile_id": str}))
                print(f"{name} batch {i + 1}/{n_batch} cached", flush=True)
                continue
            chunk = profiles.iloc[i * batch_size : (i + 1) * batch_size].copy()
            print(f"{name} batch {i + 1}/{n_batch}", flush=True)
            sampled = _sample_resilient(image, chunk, scale, f"{name}-{i}")
            sampled.to_csv(dest, index=False)
            frames.append(sampled)
        part = pd.concat(frames, ignore_index=True).drop_duplicates("profile_id")
        parts.append(part)
        part.to_parquet(root / f"{name}.parquet", index=False)
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="profile_id", how="outer")
    if {"map_mm", "pet_mm"}.issubset(merged.columns):
        merged["ai"] = merged["map_mm"] / merged["pet_mm"].clip(lower=1.0)
    dest = root / "profile_covariates.parquet"
    merged.to_parquet(dest, index=False)
    if limit is None:
        _write_feature_manifest(merged)
    print(f"wrote {dest} rows={len(merged)} cols={len(merged.columns)}", flush=True)
    return dest


def _write_feature_manifest(frame: pd.DataFrame) -> None:
    predictor = [
        c
        for c in frame.columns
        if c != "profile_id" and not str(c).startswith("olm_")
    ]
    rows = []
    for col in frame.columns:
        if col == "profile_id":
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        rows.append(
            {
                "column": col,
                "role": "external_comparison_only" if str(col).startswith("olm_") else "predictor",
                "non_null": int(series.notna().sum()),
                "min": None if series.notna().sum() == 0 else float(series.min()),
                "median": None if series.notna().sum() == 0 else float(series.median()),
                "max": None if series.notna().sum() == 0 else float(series.max()),
            }
        )
    pd.DataFrame(rows).to_csv(MANIFESTS / "gee_feature_manifest.csv", index=False)
    (MANIFESTS / "gee_assets.json").write_text(
        json.dumps(
            {
                "gee_project": GEE_PROJECT,
                "period": [START, END],
                "predictors": predictor,
                "assets": [
                    "USGS/SRTMGL1_003",
                    "ECMWF/ERA5_LAND/MONTHLY_AGGR",
                    "UCSB-CHG/CHIRPS/DAILY",
                    "MODIS/061/MOD13Q1",
                    "ESA/WorldCover/v200",
                    "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02",
                    "OpenLandMap/SOL/SOL_BULKDENS-FINEEARTH_USDA-4A1H_M/v02",
                ],
                "openlandmap_policy": "Point comparison only. Not used as MCU-SoilNet or baseline inputs.",
                "region": "Bounding rectangle of GS（2024）0650 polygons and 境界线.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _download_url(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=900)
    response.raise_for_status()
    payload = response.content
    if payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith((".tif", ".tiff"))]
            if len(names) != 1:
                raise RuntimeError(f"Expected one GeoTIFF in {dest.name}, found {names}")
            dest.write_bytes(archive.read(names[0]))
    else:
        dest.write_bytes(payload)


def download_map_rasters(degree: float = 0.1) -> None:
    """Download covariate grids aligned to the official boundary extent."""
    initialize()
    roi = roi_rectangle()
    minx, miny, maxx, maxy = official_bounds()
    region = [minx, miny, maxx, maxy]
    stacks = {
        "terrain": terrain_image(roi),
        "climate": climate_image(roi),
        "optical": optical_image(roi),
    }
    for name, image in stacks.items():
        dest = RASTERS / f"{name}_{degree:.2f}deg.tif"
        if dest.exists():
            print(f"cached {dest.name}", flush=True)
            continue
        pieces = 2 if name == "optical" else 1
        _download_pieces(image, dest, region, degree, pieces)


def _download_pieces(image: ee.Image, dest: Path, region: list[float], degree: float, pieces: int) -> None:
    minx, miny, maxx, maxy = region
    if pieces == 1:
        _download_one(image, dest, region, degree)
        return
    paths = []
    span = (maxx - minx) / pieces
    for i in range(pieces):
        piece_region = [minx + i * span, miny, minx + (i + 1) * span, maxy]
        piece_path = dest.with_name(f"{dest.stem}_part{i}.tif")
        print(f"downloading {dest.stem} part {i + 1}/{pieces}", flush=True)
        _download_one(image, piece_path, piece_region, degree)
        paths.append(piece_path)
    import rasterio
    from rasterio.merge import merge

    datasets = [rasterio.open(path) for path in paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(width=mosaic.shape[2], height=mosaic.shape[1], transform=transform, compress="deflate")
    with rasterio.open(dest, "w", **profile) as dst:
        dst.write(mosaic)
    for src in datasets:
        src.close()
    for path in paths:
        path.unlink(missing_ok=True)
    print(f"wrote {dest}", flush=True)


def _download_one(image: ee.Image, dest: Path, region: list[float], degree: float) -> None:
    minx, miny, maxx, maxy = region
    width = max(2, int(round((maxx - minx) / degree)))
    height = max(2, int(round((maxy - miny) / degree)))
    url = image.getDownloadURL(
        {
            "name": dest.stem,
            "region": ee.Geometry.Rectangle(region),
            "dimensions": [width, height],
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    print(f"fetching {dest.name} {width}x{height}", flush=True)
    _download_url(url, dest)
    print(f"wrote {dest.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--profiles", action="store_true")
    parser.add_argument("--map", action="store_true")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--degree", type=float, default=0.1)
    args = parser.parse_args()
    if args.pilot:
        extract_profiles(batch_size=args.pilot, limit=args.pilot)
    if args.profiles:
        extract_profiles(batch_size=args.batch_size, limit=None)
    if args.map:
        download_map_rasters(degree=args.degree)


if __name__ == "__main__":
    main()

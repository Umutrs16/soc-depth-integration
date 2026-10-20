"""National maps from GEE covariate rasters and a refit MCU-SoilNet.

Grid cells are masked with the GS（2024）0650 province polygons.
The 境界线 layer is not a land mask; it is drawn only in the figures.
Water (WorldCover 80) and permanent snow/ice (70) are excluded from stock totals.
"""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.features import rasterize
from rasterio.transform import Affine

from .boundaries import load_provinces
from .experiment import _load, _profile_table, _split_fit_cal_val
from .models import (
    NUMERIC_FEATURES,
    SPECS,
    WC_CLASSES,
    design_matrix,
    ensemble_predict_layers,
    predict_stock_batch,
    train_mcu,
)
from .paths import MANIFESTS, METRICS, RASTERS, TABLES
from .qc import stock_to_depth

GEOD = Geod(ellps="WGS84")
EXCLUDE_COVER = {70, 80}


def _read_stack(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    data[~np.isfinite(data)] = np.nan
    return data, profile, transform, crs


def _align(data, src_transform, src_crs, profile, nearest_last: bool):
    from rasterio.warp import Resampling, reproject

    dest = np.full((data.shape[0], profile["height"], profile["width"]), np.nan, dtype=np.float32)
    for band in range(data.shape[0]):
        nearest = nearest_last and band == data.shape[0] - 1
        reproject(
            source=data[band],
            destination=dest[band],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.nearest if nearest else Resampling.bilinear,
        )
    return dest


def load_covariate_cube(degree: float = 0.1):
    expected = _expected_names()
    terrain, profile, transform, crs = _read_stack(RASTERS / f"terrain_{degree:.2f}deg.tif")
    profile["crs"] = crs
    arrays = [terrain]
    names = list(expected["terrain"])
    for key in ("climate", "optical"):
        data, _profile, src_transform, src_crs = _read_stack(RASTERS / f"{key}_{degree:.2f}deg.tif")
        same_grid = data.shape[1:] == terrain.shape[1:] and np.allclose(
            np.array(src_transform)[:6], np.array(transform)[:6], atol=1e-5
        )
        if not same_grid:
            data = _align(data, src_transform, src_crs, profile, nearest_last=(key == "optical"))
        arrays.append(data)
        names.extend(expected[key])
    cube = np.concatenate(arrays, axis=0)
    return cube, names, profile, transform


def _expected_names():
    return {
        "terrain": ["elevation", "slope", "aspect_sin", "aspect_cos", "tpi_2km", "relief_2km"],
        "climate": ["mat_c", "tseason_c", "trange_c", "pet_mm", "soil_vwc", "map_mm"],
        "optical": ["ndvi_mean", "ndvi_std", "ndvi_p10", "ndvi_p50", "ndvi_p90", "worldcover"],
    }


def land_masks(transform: Affine, shape):
    provinces = load_provinces()
    height, width = shape
    shapes = [(geom, i + 1) for i, geom in enumerate(provinces.geometry)]
    province = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0, dtype="int16")
    names = ["outside"] + provinces["name_en"].tolist()
    return province, names


def pixel_areas(transform: Affine, height: int, width: int, land: np.ndarray) -> np.ndarray:
    area = np.zeros((height, width), dtype=np.float64)
    rows, cols = np.where(land)
    for row, col in zip(rows, cols):
        x0, y0 = transform * (col, row)
        x1, y1 = transform * (col + 1, row + 1)
        poly_area, _perim = GEOD.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        area[row, col] = abs(poly_area)
    return area


def _fit_map_ensemble(layers, profile_meta):
    spec = SPECS["mcu"]
    all_ids = set(profile_meta["profile_id"].astype(str))
    fit_ids, val_ids, cal_ids, _tree = _split_fit_cal_val(profile_meta, np.array(sorted(all_ids)), fold=0)
    from .experiment import _profile_features

    x_lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
    models = []
    device = None
    max_layers = None
    for seed in spec.seeds:
        model, max_layers, device = train_mcu(layers, x_lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
        models.append(model)
    cal_pred = []
    meta = profile_meta.set_index("profile_id")
    for depth, ycol in ((30, "stock_0_30"), (100, "stock_0_100")):
        y, p = [], []
        for pid in cal_ids:
            if pid not in meta.index or not np.isfinite(meta.loc[pid, ycol]):
                continue
            preds = [float(predict_stock_batch(model, x_lookup[pid][None, :], depth, device)[0]) for model in models]
            y.append(float(meta.loc[pid, ycol]))
            p.append(float(np.mean(preds)))
        resid = np.asarray(y) - np.asarray(p)
        q05, q95 = np.quantile(resid, [0.05, 0.95]) if len(resid) >= 30 else (-1.0, 1.0)
        cal_pred.append({"depth": depth, "q05": float(q05), "q95": float(q95), "n_cal": int(len(resid))})
    # Feature moments for raster rows. Use the same medians and scaler as x_lookup.
    return models, device, x_lookup, fit_ids, cal_pred


def _raster_feature_matrix(cube, names, rows, cols, medians):
    frame = pd.DataFrame({name: cube[i, rows, cols] for i, name in enumerate(names)})
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["top_cm"] = 0.0
    frame["bottom_cm"] = 0.0
    frame["worldcover"] = np.round(frame["worldcover"].fillna(-1))
    for col in NUMERIC_FEATURES:
        if col not in frame:
            frame[col] = np.nan
    spec = SPECS["mcu"]
    x, _, _ = design_matrix(frame, spec, medians=medians, for_tree=False)
    return x


def _medians_from_fit(layers, fit_ids):
    spec = SPECS["mcu"]
    first = layers.groupby("profile_id", sort=False).first().reset_index()
    first = first.loc[first["profile_id"].isin(fit_ids)].copy()
    first["top_cm"] = 0.0
    first["bottom_cm"] = 0.0
    _, medians, _ = design_matrix(first, spec, for_tree=False)
    return medians


def _scale_like_training(x, layers, fit_ids, spec):
    from .experiment import _profile_features

    _lookup, _names, _med, mu, sd = _profile_features(layers, spec, fit_ids)
    return (x - mu) / sd, _lookup


def run_national_map(degree: float = 0.1) -> None:
    layers, profiles = _load()
    profile_meta = _profile_table(layers, profiles)
    spec = SPECS["mcu"]
    all_ids = np.array(sorted(profile_meta["profile_id"].astype(str)))
    fit_ids, val_ids, cal_ids, _tree = _split_fit_cal_val(profile_meta, all_ids, fold=0)
    from .experiment import _profile_features

    x_lookup, _names, medians, mu, sd = _profile_features(layers, spec, fit_ids)
    models, device = [], None
    for seed in spec.seeds:
        model, _max_layers, device = train_mcu(layers, x_lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
        models.append(model)
    quant = {}
    meta = profile_meta.set_index("profile_id")
    for depth, ycol, key in ((30, "stock_0_30", "30"), (100, "stock_0_100", "100")):
        y_list, p_list = [], []
        for pid in sorted(cal_ids):
            observed = meta.loc[pid, ycol] if pid in meta.index else np.nan
            if not np.isfinite(observed):
                continue
            preds = [float(predict_stock_batch(m, x_lookup[pid][None, :], depth, device)[0]) for m in models]
            y_list.append(float(observed))
            p_list.append(float(np.mean(preds)))
        resid = np.asarray(y_list) - np.asarray(p_list)
        q05, q95 = (np.quantile(resid, [0.05, 0.95]) if len(resid) >= 30 else np.array([-1.0, 1.0]))
        quant[key] = {"q05": float(q05), "q95": float(q95), "n_cal": int(len(resid))}

    cube, names, profile, transform = load_covariate_cube(degree)
    _bands, height, width = cube.shape
    province, province_names = land_masks(transform, (height, width))
    land = province > 0
    rows, cols = np.where(land)
    x = _raster_feature_matrix(cube, names, rows, cols, medians)
    # design_matrix already median-imputes. Apply the training scaler.
    x = (x - mu) / sd
    pred = {}
    for depth, key in ((30, "30"), (100, "100")):
        acc = np.zeros((len(rows), len(models)), dtype=np.float32)
        for i, model in enumerate(models):
            for start in range(0, len(rows), 8192):
                stop = min(len(rows), start + 8192)
                acc[start:stop, i] = predict_stock_batch(model, x[start:stop], depth, device)
        mean = acc.mean(axis=1)
        grid = np.full((height, width), np.nan, dtype=np.float32)
        lo = np.full_like(grid, np.nan)
        hi = np.full_like(grid, np.nan)
        grid[rows, cols] = mean
        lo[rows, cols] = mean + quant[key]["q05"]
        hi[rows, cols] = mean + quant[key]["q95"]
        pred[key] = {"mean": grid, "lo": lo, "hi": hi}

    worldcover = cube[names.index("worldcover")]
    excluded = np.isin(np.round(worldcover), list(EXCLUDE_COVER))
    areas = pixel_areas(transform, height, width, land)
    summary_rows = []
    province_rows = []
    for key in ("30", "100"):
        usable = land & np.isfinite(pred[key]["mean"]) & ~excluded
        total_kg = float(np.nansum(pred[key]["mean"][usable] * areas[usable]))
        summary_rows.append(
            {
                "depth": f"0-{key} cm",
                "pg_c_land_excluding_water_snow": total_kg / 1e12,
                "area_km2": float(areas[usable].sum() / 1e6),
                "excluded_water_snow_km2": float(areas[land & excluded].sum() / 1e6),
                "calibration": quant[key],
            }
        )
        for idx, name in enumerate(province_names[1:], start=1):
            sel = usable & (province == idx)
            if sel.sum() == 0:
                continue
            province_rows.append(
                {
                    "depth": f"0-{key} cm",
                    "province_en": name,
                    "pg_c": float(np.nansum(pred[key]["mean"][sel] * areas[sel]) / 1e12),
                    "area_km2": float(areas[sel].sum() / 1e6),
                    "mean_kg_m2": float(np.average(pred[key]["mean"][sel], weights=areas[sel])),
                }
            )
    profile.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")
    for key, kind in (("30", "mean"), ("30", "lo"), ("30", "hi"), ("100", "mean"), ("100", "lo"), ("100", "hi")):
        dest = RASTERS / f"socd_0_{key}_{kind}.tif"
        with rasterio.open(dest, "w", **profile) as dst:
            dst.write(pred[key][kind], 1)
            dst.update_tags(
                review="GS（2024）0650号",
                boundary="中国标准行政区划数据GS（2024）0650号 中国_省.shp",
                depth_cm=key,
                quantity=kind,
            )
    pd.DataFrame(summary_rows).to_csv(TABLES / "national_stocks.csv", index=False)
    pd.DataFrame(province_rows).to_csv(TABLES / "stocks_by_province.csv", index=False)
    (MANIFESTS / "map_calibration.json").write_text(json.dumps(quant, indent=2), encoding="utf-8")
    # AOA from covariate distance to training profiles.
    train_x = np.stack([x_lookup[pid] for pid in sorted(fit_ids | val_ids)])
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(train_x)
    train_dist, _ = nn.kneighbors(train_x)
    threshold = float(np.quantile(train_dist[:, 1], 0.95))
    pixel_nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(train_x)
    pixel_dist, _ = pixel_nn.kneighbors(x)
    aoa = np.zeros((height, width), dtype=np.uint8)
    inside = pixel_dist[:, 0] <= threshold
    aoa[rows[inside], cols[inside]] = 1
    aoa_profile = profile.copy()
    aoa_profile.update(dtype="uint8", nodata=0)
    with rasterio.open(RASTERS / "aoa_mask.tif", "w", **aoa_profile) as dst:
        dst.write(aoa, 1)
        dst.update_tags(review="GS（2024）0650号", threshold=threshold)
    aoa_rows = []
    for key in ("30", "100"):
        usable = land & np.isfinite(pred[key]["mean"]) & ~excluded & (aoa == 1)
        total_kg = float(np.nansum(pred[key]["mean"][usable] * areas[usable]))
        aoa_rows.append(
            {
                "depth": f"0-{key} cm",
                "pg_c_inside_aoa": total_kg / 1e12,
                "area_km2_inside_aoa": float(areas[usable].sum() / 1e6),
                "aoa_distance_threshold": threshold,
            }
        )
    pd.DataFrame(aoa_rows).to_csv(TABLES / "aoa_stocks.csv", index=False)
    print("national map written", flush=True)


if __name__ == "__main__":
    run_national_map()

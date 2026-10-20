"""Audit V10/V9 and build profile-safe supervision labels.

V10 is the only modeling table. V9 is used solely to count version
differences and is never mixed in as extra independent profiles.
No gravel fraction (CF) column exists, so a C–BD–CF product identity
is not treated as identified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml

from .boundaries import XINJIANG_NAME, assign_province
from .paths import MANIFESTS, PROCESSED, RAW_V10, RAW_V9, README_V10, TABLES

GAP_TOL_CM = 1.0
OVERLAP_TOL_CM = 1.0
CLOSURE_REL_TOL = 0.10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rename_mat(df: pd.DataFrame) -> pd.DataFrame:
    mat = [c for c in df.columns if str(c).startswith("MAT")]
    if not mat:
        raise RuntimeError(f"MAT column not found: {list(df.columns)}")
    return df.rename(columns={mat[0]: "mat_file_c"})


def load_v10() -> pd.DataFrame:
    df = pd.read_csv(RAW_V10, encoding="gb18030")
    df = _rename_mat(df)
    expected = [
        "Samples",
        "Profile",
        "Latitude",
        "Longitude",
        "Upper_depth_cm",
        "Lower_depth_cm",
        "BD_g_cm3",
        "SOC_g_kg",
        "SOCD_kg_m_2",
        "SOCD_methods",
        "References",
        "Time",
    ]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"V10 is missing columns: {missing}")
    return df


def load_v9() -> pd.DataFrame:
    df = pd.read_csv(RAW_V9, encoding="gb18030")
    return df.rename(
        columns={
            "profile_id": "Profile",
            "latitude": "Latitude",
            "longitude": "Longitude",
            "upper_dept": "Upper_depth_cm",
            "lower_dept": "Lower_depth_cm",
            "SOCD (kg/m2)": "SOCD_kg_m_2",
            "Year": "Time",
        }
    )


def _layer_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["profile_id"] = out["Profile"].astype(str)
    out["sample_id"] = out["Samples"].astype(str)
    out["top_cm"] = pd.to_numeric(out["Upper_depth_cm"], errors="coerce")
    out["bottom_cm"] = pd.to_numeric(out["Lower_depth_cm"], errors="coerce")
    out["thickness_cm"] = out["bottom_cm"] - out["top_cm"]
    out["soc_gkg"] = pd.to_numeric(out["SOC_g_kg"], errors="coerce")
    out["bd_gcm3"] = pd.to_numeric(out["BD_g_cm3"], errors="coerce")
    out["socd_kgm2"] = pd.to_numeric(out["SOCD_kg_m_2"], errors="coerce")
    out["year"] = pd.to_numeric(out["Time"], errors="coerce").astype("Int64")
    out["method"] = out["SOCD_methods"].astype(str).str.strip().str.lower()
    out["reference"] = out["References"].astype(str).str.strip()
    out["soc_observed"] = out["soc_gkg"].notna()
    out["bd_observed"] = out["bd_gcm3"].notna()
    out["valid_interval"] = (
        out["top_cm"].notna()
        & out["bottom_cm"].notna()
        & (out["thickness_cm"] > 0)
        & (out["top_cm"] >= 0)
        & (out["bottom_cm"] <= 800)
    )
    out["valid_socd"] = out["socd_kgm2"].notna() & (out["socd_kgm2"] >= 0) & (out["socd_kgm2"] <= 500)
    out["valid_soc"] = ~out["soc_observed"] | ((out["soc_gkg"] >= 0) & (out["soc_gkg"] <= 700))
    out["valid_bd"] = ~out["bd_observed"] | ((out["bd_gcm3"] >= 0.05) & (out["bd_gcm3"] <= 2.65))
    out["valid_coord"] = out["Latitude"].between(3, 55) & out["Longitude"].between(70, 140)
    recon = 0.01 * out["soc_gkg"] * out["bd_gcm3"] * out["thickness_cm"]
    out["recon_socd"] = recon
    denom = out["socd_kgm2"].abs().clip(lower=1e-6)
    out["recon_abs_err"] = (recon - out["socd_kgm2"]).abs()
    out["recon_rel_err"] = out["recon_abs_err"] / denom
    both = out["soc_observed"] & out["bd_observed"] & out["valid_interval"]
    out["closure_ok"] = both & out["method"].eq("direct") & (out["recon_rel_err"] <= CLOSURE_REL_TOL)

    def reliability(row) -> str:
        if not (row.valid_interval and row.valid_socd and row.valid_soc and row.valid_bd and row.valid_coord):
            return "D"
        if row.soc_observed and row.bd_observed and row.method == "direct":
            return "B_direct"
        if row.soc_observed and row.bd_observed and row.method == "indirect":
            return "B_indirect"
        if row.valid_socd and not (row.soc_observed and row.bd_observed):
            return "C_socd_only"
        return "D"

    out["reliability"] = out.apply(reliability, axis=1)
    out["train_weight"] = out["reliability"].map(
        {"B_direct": 1.0, "B_indirect": 0.5, "C_socd_only": 1.0, "D": 0.0}
    )
    return out


def _profile_geometry_flags(layers: pd.DataFrame) -> pd.DataFrame:
    records = []
    for pid, g in layers.sort_values(["top_cm", "bottom_cm"]).groupby("profile_id", sort=False):
        top = g["top_cm"].to_numpy()
        bot = g["bottom_cm"].to_numpy()
        overlap = 0
        if len(g) > 1:
            overlap = int(np.sum(top[1:] < bot[:-1] - OVERLAP_TOL_CM))
        records.append(
            {
                "profile_id": pid,
                "n_layers": int(len(g)),
                "n_overlap": overlap,
                "depth_max_cm": float(np.nanmax(bot)),
                "depth_min_cm": float(np.nanmin(top)),
                "latitude": float(g["Latitude"].median()),
                "longitude": float(g["Longitude"].median()),
                "year": int(g["year"].dropna().mode().iloc[0]) if g["year"].notna().any() else pd.NA,
                "reference": g["reference"].mode().iloc[0],
                "coord_span_m": _coord_span_m(g),
            }
        )
    profiles = pd.DataFrame.from_records(records)
    profiles["has_overlap"] = profiles["n_overlap"] > 0
    return profiles


def _coord_span_m(g: pd.DataFrame) -> float:
    lat = np.deg2rad(g["Latitude"].to_numpy())
    lon = np.deg2rad(g["Longitude"].to_numpy())
    if len(g) < 2:
        return 0.0
    dlat = (lat.max() - lat.min()) * 6371000
    dlon = (lon.max() - lon.min()) * 6371000 * np.cos(np.nanmean(lat))
    return float(np.hypot(dlat, dlon))


def stock_to_depth(g: pd.DataFrame, depth_cm: float, value_col: str = "socd_kgm2") -> float:
    """Mass-conserving sum of layer stocks that tile [0, depth].

    Returns NaN when the surface is missing, a gap exceeds 1 cm, layers
    overlap by more than 1 cm, or the profile does not reach depth_cm.
    A layer that crosses the target depth contributes only its overlapping
    fraction. Incomplete profiles are never filled with zero.
    """
    g = g.sort_values(["top_cm", "bottom_cm"])
    if g.empty or bool(g["has_overlap"].iloc[0]):
        return np.nan
    top = g["top_cm"].to_numpy(dtype=float)
    bot = g["bottom_cm"].to_numpy(dtype=float)
    val = g[value_col].to_numpy(dtype=float)
    if np.nanmin(top) > GAP_TOL_CM:
        return np.nan
    cursor = 0.0
    total = 0.0
    for a, b, y in zip(top, bot, val):
        if not np.isfinite(y) or b <= a:
            return np.nan
        if a > cursor + GAP_TOL_CM and a < depth_cm:
            return np.nan
        left = max(a, 0.0)
        right = min(b, depth_cm)
        if right > left:
            total += y * (right - left) / (b - a)
        cursor = max(cursor, b)
        if cursor >= depth_cm - 1e-6:
            return float(total)
    return np.nan


def _attach_stocks(layers: pd.DataFrame, profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    layers = layers.merge(profiles[["profile_id", "has_overlap"]], on="profile_id", how="left")
    rows = []
    for pid, g in layers.groupby("profile_id", sort=False):
        rows.append(
            {
                "profile_id": pid,
                "stock_0_30": stock_to_depth(g, 30.0),
                "stock_0_100": stock_to_depth(g, 100.0),
                "stock_100_200": _deep_stock(g),
            }
        )
    stocks = pd.DataFrame(rows)
    profiles = profiles.merge(stocks, on="profile_id", how="left")
    profiles["cover_0_30"] = profiles["stock_0_30"].notna()
    profiles["cover_0_100"] = profiles["stock_0_100"].notna()
    profiles["cover_100_200"] = profiles["stock_100_200"].notna()
    return layers, profiles


def _deep_stock(g: pd.DataFrame) -> float:
    """0–100 and 100–200 must both be supported before a deep increment is kept."""
    s100 = stock_to_depth(g, 100.0)
    s200 = stock_to_depth(g, 200.0)
    if not (np.isfinite(s100) and np.isfinite(s200)):
        return np.nan
    return float(s200 - s100)


def _spatial_blocks(profiles: pd.DataFrame, block_deg: float = 2.0) -> pd.DataFrame:
    out = profiles.copy()
    out["block_x"] = np.floor(out["longitude"] / block_deg).astype(int)
    out["block_y"] = np.floor(out["latitude"] / block_deg).astype(int)
    out["block_id"] = out["block_x"].astype(str) + "_" + out["block_y"].astype(str)
    return out


def _assign_admin(profiles: pd.DataFrame) -> pd.DataFrame:
    geo = gpd.GeoDataFrame(
        profiles[["profile_id"]].copy(),
        geometry=gpd.points_from_xy(profiles["longitude"], profiles["latitude"]),
        crs="EPSG:4326",
    )
    assigned = assign_province(geo)
    keep = assigned[["profile_id", "name", "name_en", "in_official_land"]].rename(
        columns={"name": "province_zh", "name_en": "province_en"}
    )
    out = profiles.merge(keep, on="profile_id", how="left")
    out["is_xinjiang"] = out["province_zh"].eq(XINJIANG_NAME)
    return out


def _version_compare(v10: pd.DataFrame, v9: pd.DataFrame) -> pd.DataFrame:
    def keys(df):
        return set(
            zip(
                df["Latitude"].round(4),
                df["Longitude"].round(4),
                df["Upper_depth_cm"].round(1),
                df["Lower_depth_cm"].round(1),
            )
        )

    a, b = keys(v10), keys(v9)
    row = {
        "v10_rows": int(len(v10)),
        "v9_rows": int(len(v9)),
        "v10_profiles": int(v10["Profile"].nunique()),
        "v9_profiles": int(v9["Profile"].nunique()),
        "readme_v10_profiles_claimed": 7995,
        "v10_profile_minus_readme": int(v10["Profile"].nunique()) - 7995,
        "shared_layer_keys_4dp": int(len(a & b)),
        "v10_only_layer_keys": int(len(a - b)),
        "v9_only_layer_keys": int(len(b - a)),
    }
    return pd.DataFrame([row])


def build_qc() -> dict:
    v10_raw = load_v10()
    v9_raw = load_v9()
    layers = _layer_flags(v10_raw)
    keep_cols = [
        "profile_id",
        "sample_id",
        "Latitude",
        "Longitude",
        "top_cm",
        "bottom_cm",
        "thickness_cm",
        "soc_gkg",
        "bd_gcm3",
        "socd_kgm2",
        "year",
        "method",
        "reference",
        "reliability",
        "train_weight",
        "soc_observed",
        "bd_observed",
        "valid_interval",
        "valid_socd",
        "valid_soc",
        "valid_bd",
        "valid_coord",
        "recon_socd",
        "recon_abs_err",
        "recon_rel_err",
        "closure_ok",
        "CLCD",
        "pH",
        "mat_file_c",
        "MAP_mm",
        "DEM_m",
        "NDVI",
    ]
    layers = layers[keep_cols].copy()
    profiles = _profile_geometry_flags(layers)
    layers, profiles = _attach_stocks(layers, profiles)
    profiles = _spatial_blocks(profiles)
    profiles = _assign_admin(profiles)
    trainable = layers["reliability"].ne("D") & layers["valid_socd"] & layers["valid_interval"]
    layers["in_training_pool"] = trainable & layers["profile_id"].isin(
        profiles.loc[profiles["in_official_land"].fillna(False), "profile_id"]
    )
    profiles["in_training_pool"] = profiles["profile_id"].isin(layers.loc[layers["in_training_pool"], "profile_id"])

    version = _version_compare(v10_raw, v9_raw)
    audit = {
        "v10_rows": int(len(v10_raw)),
        "v10_unique_profiles": int(v10_raw["Profile"].nunique()),
        "v10_unique_samples": int(v10_raw["Samples"].nunique()),
        "readme_claimed_profiles": 7995,
        "profile_count_discrepancy_vs_readme": int(v10_raw["Profile"].nunique()) - 7995,
        "references": int(v10_raw["References"].nunique()),
        "year_min": int(v10_raw["Time"].min()),
        "year_max": int(v10_raw["Time"].max()),
        "cf_column_present": False,
        "method_counts": v10_raw["SOCD_methods"].value_counts().to_dict(),
        "reliability_counts": layers["reliability"].value_counts().to_dict(),
        "profiles_with_layer_overlap": int(profiles["has_overlap"].sum()),
        "profiles_cover_0_30": int(profiles["cover_0_30"].sum()),
        "profiles_cover_0_100": int(profiles["cover_0_100"].sum()),
        "profiles_cover_100_200": int(profiles["cover_100_200"].sum()),
        "profiles_inside_official_land": int(profiles["in_official_land"].fillna(False).sum()),
        "profiles_outside_official_land": int((~profiles["in_official_land"].fillna(False)).sum()),
        "xinjiang_profiles": int(profiles["is_xinjiang"].fillna(False).sum()),
        "layers_in_training_pool": int(layers["in_training_pool"].sum()),
        "profiles_in_training_pool": int(profiles["in_training_pool"].sum()),
        "closure_ok_layers": int(layers["closure_ok"].sum()),
        "median_recon_rel_err_where_both": float(
            layers.loc[layers["soc_observed"] & layers["bd_observed"], "recon_rel_err"].median()
        ),
        "excluded_from_predictors": [
            "pH",
            "Sand_%",
            "Silt_%",
            "Clay_%",
            "DEM_m",
            "NDVI",
            "MAT",
            "MAP_mm",
            "PET_mm",
            "AI",
            "CLCD",
        ],
        "exclusion_reason": (
            "These V10 fields are complete even where the README says deep attributes "
            "could not be extracted, so they are treated as prejoined grids rather than "
            "independent map covariates. Main predictors come from GEE."
        ),
    }
    by_prov = (
        profiles.loc[profiles["in_official_land"].fillna(False)]
        .groupby(["province_zh", "province_en"], dropna=False)
        .agg(
            profiles=("profile_id", "size"),
            cover_0_30=("cover_0_30", "sum"),
            cover_0_100=("cover_0_100", "sum"),
            xinjiang=("is_xinjiang", "max"),
        )
        .reset_index()
        .sort_values("profiles", ascending=False)
    )
    by_depth = pd.DataFrame(
        [
            {"target": "0-30 cm stock", "profiles": audit["profiles_cover_0_30"]},
            {"target": "0-100 cm stock", "profiles": audit["profiles_cover_0_100"]},
            {"target": "100-200 cm increment", "profiles": audit["profiles_cover_100_200"]},
            {"target": "layer rows in training pool", "profiles": audit["layers_in_training_pool"]},
        ]
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    layers.to_parquet(PROCESSED / "layers_qc.parquet", index=False)
    profiles.to_parquet(PROCESSED / "profiles_qc.parquet", index=False)
    version.to_csv(TABLES / "v10_vs_v9.csv", index=False)
    by_prov.to_csv(TABLES / "profiles_by_province.csv", index=False)
    by_depth.to_csv(TABLES / "depth_support.csv", index=False)
    (TABLES / "v10_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "v10_csv": str(RAW_V10),
        "v10_sha256": sha256_file(RAW_V10),
        "v10_readme_sha256": sha256_file(README_V10),
        "v9_csv": str(RAW_V9),
        "v9_sha256": sha256_file(RAW_V9),
        "v10_doi": "10.5281/zenodo.17304024",
        "v9_doi": "10.5281/zenodo.15851730",
        "article_doi": "10.1038/s41597-025-05863-3",
        "boundary": "中国标准行政区划数据 GS（2024）0650号",
        "boundary_role": "All maps, province assignment, Xinjiang holdout, and land area",
        "modeling_table": "V10 only",
    }
    (MANIFESTS / "data_manifest.yml").write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return audit


if __name__ == "__main__":
    summary = build_qc()
    print(json.dumps(summary, ensure_ascii=False, indent=2))

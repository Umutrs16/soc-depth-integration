"""External hidden-layer check on the public WoSIS December 2023 snapshot.

Profiles are limited to the official GS (2024) 0650 land polygons. Coarse
fragments are not treated as zero when they are missing. A layer stock is
scored only where organic carbon, one bulk-density method and volumetric
coarse fragments are all present. Organic-carbon concentration is scored
separately, in g/kg, and is not a stock. The snapshot is not copied into the
code archive.
"""

from __future__ import annotations

import json
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .boundaries import load_provinces
from .paths import DATA, FIGURES, METRICS, PROCESSED, TABLES
from .round16_eaqss import (
    PRIMARY_LAMBDA,
    _linear_stock,
    _nearest_stock,
    _predict_stock,
    _profile_bootstrap,
)

SEED = 29
BD_PRIORITY = ("bdfiod", "bdwsod", "bdfi33", "bdfifm", "bdfiad")
KEEP_COLUMNS = (
    "profile_id",
    "layer_id",
    "upper_depth",
    "lower_depth",
    "value_avg",
    "longitude",
    "latitude",
    "licence",
    "country_name",
)


def _member(archive: zipfile.ZipFile, code: str) -> str:
    target = f"wosis_202312_{code}.tsv"
    for name in archive.namelist():
        if name.replace("\\", "/").lower().endswith(target):
            return name
    raise FileNotFoundError(target)


def _public_licence(series: pd.Series) -> pd.Series:
    text = series.fillna("").str.lower().str.replace("_", " ", regex=False)
    return text.str.contains("cc by") | text.str.contains("cc-by") | text.str.contains("cc0") | text.str.contains("public domain")


def _licence_family(series: pd.Series) -> pd.Series:
    text = series.fillna("").str.lower()
    family = pd.Series("other", index=series.index)
    family.loc[text.str.contains("by-nc") | text.str.contains("by nc")] = "cc_by_nc"
    family.loc[text.str.contains("cc0")] = "cc0"
    family.loc[text.str.contains("public domain") & ~text.str.contains("cc0")] = "public_domain"
    family.loc[text.str.contains("cc by") & ~text.str.contains("nc") & ~text.str.contains("cc0")] = "cc_by"
    return family


def _read_member(archive: zipfile.ZipFile, code: str, profile_ids: set[str] | None) -> pd.DataFrame:
    name = _member(archive, code)
    pieces = []
    with archive.open(name) as handle:
        for chunk in pd.read_csv(handle, sep="\t", chunksize=250_000, low_memory=False):
            chunk.columns = [str(column).lower() for column in chunk.columns]
            use = [column for column in KEEP_COLUMNS if column in chunk.columns]
            chunk = chunk[use]
            if "licence" in chunk.columns:
                chunk = chunk.loc[_public_licence(chunk["licence"])]
            if profile_ids is None and {"longitude", "latitude"}.issubset(chunk.columns):
                chunk = chunk.loc[chunk["longitude"].between(73, 136) & chunk["latitude"].between(3, 54)]
            if profile_ids is not None and "profile_id" in chunk.columns:
                chunk = chunk.loc[chunk["profile_id"].astype(str).isin(profile_ids)]
            if not chunk.empty:
                pieces.append(chunk)
    if not pieces:
        return pd.DataFrame(columns=list(KEEP_COLUMNS))
    frame = pd.concat(pieces, ignore_index=True)
    for column in ("profile_id", "layer_id"):
        if column in frame.columns:
            frame[column] = frame[column].astype(str)
    for column in ("upper_depth", "lower_depth", "value_avg", "longitude", "latitude"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _china(frame: pd.DataFrame) -> pd.DataFrame:
    box = frame.loc[
        frame["longitude"].between(73, 136) & frame["latitude"].between(3, 54)
    ].copy()
    import geopandas as gpd

    provinces = load_provinces()
    points = gpd.GeoDataFrame(
        box,
        geometry=gpd.points_from_xy(box["longitude"], box["latitude"]),
        crs="EPSG:4326",
    )
    land = provinces[["geometry"]].copy()
    joined = gpd.sjoin(points, land, predicate="within", how="inner")
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    return out.drop_duplicates(["profile_id", "layer_id"])


def _layers() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    archive_path = DATA / "raw" / "wosis" / "WoSIS_2023_December.zip"
    with zipfile.ZipFile(archive_path) as archive:
        organic = _china(_read_member(archive, "orgc", None))
        ids = set(organic["profile_id"].astype(str))
        density = {}
        for code in BD_PRIORITY:
            part = _read_member(archive, code, ids)
            if part.empty:
                density[code] = 0
                continue
            part = part.drop_duplicates(["profile_id", "layer_id"])
            density[code] = int(len(part))
            organic = organic.merge(
                part[["profile_id", "layer_id", "value_avg"]].rename(columns={"value_avg": code}),
                on=["profile_id", "layer_id"],
                how="left",
            )
        fragments = _read_member(archive, "cfvo", ids).drop_duplicates(["profile_id", "layer_id"])
        gravimetric = _read_member(archive, "cfgr", ids).drop_duplicates(["profile_id", "layer_id"])
    organic["profile_id"] = organic["profile_id"].astype(str)
    organic["layer_id"] = organic["layer_id"].astype(str)
    organic["bulk_density"] = np.nan
    organic["bulk_density_code"] = ""
    for code in BD_PRIORITY:
        if code not in organic.columns:
            continue
        missing = organic["bulk_density"].isna() & organic[code].notna()
        organic.loc[missing, "bulk_density"] = organic.loc[missing, code]
        organic.loc[missing, "bulk_density_code"] = code
    if not fragments.empty:
        fragments["profile_id"] = fragments["profile_id"].astype(str)
        fragments["layer_id"] = fragments["layer_id"].astype(str)
        organic = organic.merge(
            fragments[["profile_id", "layer_id", "value_avg"]].rename(columns={"value_avg": "cfvo"}),
            on=["profile_id", "layer_id"],
            how="left",
        )
    else:
        organic["cfvo"] = np.nan
    organic["thickness_cm"] = organic["lower_depth"] - organic["upper_depth"]
    with_density = organic["value_avg"].gt(0) & organic["bulk_density"].gt(0) & organic["thickness_cm"].gt(0)
    usable = organic.loc[with_density & organic["cfvo"].between(0, 100)].copy()
    usable["socd_kgm2"] = (
        0.01
        * usable["value_avg"]
        * usable["bulk_density"]
        * usable["thickness_cm"]
        * (1.0 - usable["cfvo"] / 100.0)
    )
    concentration = organic.loc[organic["value_avg"].gt(0) & organic["thickness_cm"].gt(0)].copy()
    families = _licence_family(organic["licence"]) if "licence" in organic.columns else pd.Series(dtype=str)
    info = {
        "china_organic_carbon_layers_public_licence": int(len(organic)),
        "china_profiles_with_organic_carbon": int(organic["profile_id"].nunique()),
        "licence_family_counts": {str(key): int(value) for key, value in families.value_counts().items()},
        "bulk_density_rows_by_code": density,
        "layers_with_carbon_and_bulk_density": int(with_density.sum()),
        "profiles_with_carbon_and_bulk_density": int(organic.loc[with_density, "profile_id"].nunique()),
        "layers_with_carbon_bulk_density_and_coarse_fragments": int(len(usable)),
        "profiles_with_complete_stock_inputs": int(usable["profile_id"].nunique()) if len(usable) else 0,
        "gravimetric_coarse_fragment_rows_not_used": int(len(gravimetric)),
        "coarse_fragments_missing_are_excluded": True,
        "bulk_density_priority": list(BD_PRIORITY),
        "bulk_density_methods_are_not_converted": True,
        "median_layer_thickness_cm": float(organic["thickness_cm"].median()) if len(organic) else None,
        "layers_at_most_5_cm": int(organic["thickness_cm"].le(5).sum()) if len(organic) else 0,
        "layers_at_most_10_cm": int(organic["thickness_cm"].le(10).sum()) if len(organic) else 0,
    }
    return usable, concentration, info


def _overlap(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["matches_v10_coordinate"] = pd.Series(dtype=bool)
        return out
    local = pd.read_parquet(PROCESSED / "profiles_qc.parquet", columns=["profile_id", "latitude", "longitude"])
    local["lat4"] = local["latitude"].round(4)
    local["lon4"] = local["longitude"].round(4)
    local_keys = set(zip(local["lat4"], local["lon4"]))
    out["lat4"] = out["latitude"].round(4)
    out["lon4"] = out["longitude"].round(4)
    out["matches_v10_coordinate"] = [key in local_keys for key in zip(out["lat4"], out["lon4"])]
    return out


def _spline(frame: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    rows = []
    failed = 0
    if frame.empty or "matches_v10_coordinate" not in frame.columns:
        return pd.DataFrame(rows), failed
    for pid, group in frame.groupby("profile_id", sort=False):
        group = group.sort_values(["upper_depth", "lower_depth"])
        top = group["upper_depth"].to_numpy(float)
        bot = group["lower_depth"].to_numpy(float)
        if mode == "stock":
            observed_layer = group["socd_kgm2"].to_numpy(float)
            density = observed_layer / (bot - top)
        elif mode == "concentration":
            observed_layer = group["value_avg"].to_numpy(float)
            density = observed_layer.copy()
        else:
            raise ValueError(mode)
        if len(observed_layer) < 3 or not np.all(np.isfinite(observed_layer)) or np.any(top[1:] < bot[:-1] - 1e-6):
            continue
        matched = bool(group["matches_v10_coordinate"].any())
        for hidden in range(len(observed_layer)):
            above = np.where(bot <= top[hidden] + 1e-6)[0]
            below = np.where(top >= bot[hidden] - 1e-6)[0]
            above = above[above != hidden]
            below = below[below != hidden]
            if len(above) == 0 or len(below) == 0:
                continue
            predicted = _predict_stock(top, bot, density, hidden, PRIMARY_LAMBDA)
            if predicted is None or not np.isfinite(predicted):
                failed += 1
                continue
            thickness = float(bot[hidden] - top[hidden])
            if mode == "stock":
                predicted_value = float(predicted)
                linear = _linear_stock(top, bot, density, hidden)
                nearest = _nearest_stock(top, bot, density, thickness, hidden)
                observed = float(observed_layer[hidden])
            else:
                predicted_value = float(predicted) / thickness
                linear = _linear_stock(top, bot, density, hidden) / thickness
                nearest = _nearest_stock(top, bot, density, thickness, hidden) / thickness
                observed = float(observed_layer[hidden])
            rows.append(
                {
                    "profile_id": pid,
                    "matches_v10_coordinate": matched,
                    "thickness_cm": thickness,
                    "abs_spline": abs(predicted_value - observed),
                    "abs_linear": abs(linear - observed),
                    "abs_nearest": abs(nearest - observed),
                }
            )
    return pd.DataFrame(rows), failed


def _macro(detail: pd.DataFrame, rng: np.random.Generator) -> dict:
    if detail.empty or "abs_spline" not in detail.columns:
        return {"n_profiles": 0, "n_layers": 0}
    means = detail.groupby("profile_id", sort=False)[["abs_spline", "abs_linear", "abs_nearest"]].mean()
    return {
        "n_profiles": int(len(means)),
        "n_layers": int(len(detail)),
        "median_layer_thickness_cm": float(detail["thickness_cm"].median()),
        "spline": _profile_bootstrap(means["abs_spline"].to_numpy(float), rng),
        "linear": _profile_bootstrap(means["abs_linear"].to_numpy(float), rng),
        "nearest": _profile_bootstrap(means["abs_nearest"].to_numpy(float), rng),
        "linear_minus_spline": _profile_bootstrap((means["abs_linear"] - means["abs_spline"]).to_numpy(float), rng),
        "nearest_minus_spline": _profile_bootstrap((means["abs_nearest"] - means["abs_spline"]).to_numpy(float), rng),
    }


def _overlap_count(frame: pd.DataFrame) -> dict:
    if frame.empty or "matches_v10_coordinate" not in frame.columns:
        return {"profiles": 0, "profiles_in_frame": 0}
    flags = frame.groupby("profile_id")["matches_v10_coordinate"].first()
    return {
        "profiles": int(flags.sum()),
        "profiles_in_frame": int(len(flags)),
        "key": "latitude and longitude rounded to 4 decimal degrees",
    }


def _target_summary(frame: pd.DataFrame, mode: str, rng: np.random.Generator) -> dict:
    frame = _overlap(frame)
    detail, failed = _spline(frame, mode)
    unmatched = detail.iloc[0:0]
    if not detail.empty:
        unmatched = detail.loc[~detail["matches_v10_coordinate"]]
    return {
        "unit": "kg m-2" if mode == "stock" else "g kg-1",
        "not_a_stock": mode != "stock",
        "failed_fits": failed,
        "coordinate_overlap_in_input_profiles": _overlap_count(frame),
        "all_scored_profiles": _macro(detail, rng),
        "profiles_without_v10_coordinate_match": _macro(unmatched, rng),
    }


def run() -> dict:
    usable, concentration, info = _layers()
    stock = _target_summary(usable, "stock", np.random.default_rng(SEED))
    concentration_summary = _target_summary(concentration, "concentration", np.random.default_rng(SEED + 2))
    stock["estimable"] = stock["all_scored_profiles"]["n_profiles"] > 0
    stock["missing_coarse_fragments_were_not_set_to_zero"] = True
    summary = {
        "source": "WoSIS snapshot December 2023",
        "doi": "10.17027/isric-wdcsoils-20231130",
        "citation": "Calisto, de Sousa and Batjes (2023), doi:10.17027/isric-wdcsoils-20231130; Batjes, Calisto and de Sousa (2024), Earth System Science Data 16, 4735-4765",
        "land_mask": "GS (2024) 0650 province polygons",
        "licences_retained": "Rows marked CC BY, CC BY-NC, CC0, or public domain",
        "not_redistributed": True,
        "not_a_fine_depth_truth": True,
        "not_a_spatial_score": True,
        "lambda": PRIMARY_LAMBDA,
        "inputs": info,
        "stock": stock,
        "concentration_g_per_kg": concentration_summary,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    (METRICS / "round17_wosis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _figure(summary)
    print(json.dumps(summary, indent=2))
    return summary


def _panel(ax, block: dict, ylabel: str, letter: str, note: str) -> None:
    keys = ("spline", "linear", "nearest")
    means = np.array([block[key]["mean"] for key in keys])
    low = np.array([block[key]["ci95_low"] for key in keys])
    high = np.array([block[key]["ci95_high"] for key in keys])
    xpos = np.arange(len(keys))
    ax.bar(
        xpos,
        means,
        color=["#2166ac", "#4d4d4d", "#b2182b"],
        width=0.72,
        yerr=np.vstack([means - low, high - means]),
        capsize=3,
        error_kw={"elinewidth": 0.8, "ecolor": "black"},
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Spline", "Linear", "Nearest"], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(0, float(np.max(high)) * 1.22)
    ax.text(0.02, 0.98, f"({letter})", transform=ax.transAxes, va="top", ha="left", fontsize=11)
    ax.text(0.98, 0.98, note, transform=ax.transAxes, va="top", ha="right", fontsize=9)


def _figure(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    _panel(
        axes[0],
        summary["stock"]["all_scored_profiles"],
        "Profile-mean absolute error (kg m$^{-2}$)",
        "a",
        "35 profiles",
    )
    _panel(
        axes[1],
        summary["concentration_g_per_kg"]["profiles_without_v10_coordinate_match"],
        "Profile-mean absolute error (g kg$^{-1}$)",
        "b",
        "1,538 profiles",
    )
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig23_wosis_horizons.png", dpi=300)
    fig.savefig(FIGURES / "fig23_wosis_horizons.pdf")
    plt.close(fig)


if __name__ == "__main__":
    run()

"""Buffer exclusion and biome holdout for a fixed-setting random forest.

The forest is fit again for these tests. Its scores are not substituted for
the archived spatial-block comparison. The buffer uses the same equirectangular
kilometres as the archived nearest-training distance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .experiment import SPECS, _folds, _load, _nearest_km, _profile_table, _split_fit_cal_val, _tree_matrix
from .metrics import metric_row
from .models import make_tree
from .paths import DATA, FIGURES, METRICS, TABLES
from .qc import stock_to_depth

BUFFERS_KM = (0.0, 25.0, 50.0, 100.0)
MIN_BIOME_TEST = 80
SEED = 23
N_BOOT = 2000


def _valid_targets() -> pd.DataFrame:
    coverage = pd.read_csv(TABLES / "round11_coverage_profiles.csv", dtype={"profile_id": str})
    part = coverage.loc[(coverage["depth_cm"] == 100.0) & coverage["status"].isin(["exact_boundary", "prorated"])]
    return part[["profile_id", "block_id", "status"]].drop_duplicates("profile_id")


def _predict_fold(layers: pd.DataFrame, meta: pd.DataFrame, train_ids: set[str], test_ids: set[str], fold: int) -> pd.DataFrame:
    spec = SPECS["rf"]
    _fit, _val, _cal, tree_ids = _split_fit_cal_val(meta, train_ids, fold)
    if len(tree_ids) < 50 or len(test_ids) == 0:
        return pd.DataFrame()
    matrix, _names, _medians = _tree_matrix(layers, spec, set(map(str, tree_ids)))
    train_mask = layers["profile_id"].isin(set(map(str, tree_ids))).to_numpy()
    test_mask = layers["profile_id"].isin(test_ids).to_numpy()
    model = make_tree(spec, seed=42)
    model.fit(matrix[train_mask], layers.loc[train_mask, "socd_kgm2"].to_numpy(float))
    predicted = np.empty(len(layers), dtype=float)
    predicted[:] = np.nan
    predicted[test_mask] = model.predict(matrix[test_mask])
    work = layers.loc[test_mask, ["profile_id", "top_cm", "bottom_cm", "has_overlap"]].copy()
    work["socd_kgm2"] = predicted[test_mask]
    rows = []
    lookup = meta.set_index("profile_id")
    for pid, group in work.groupby("profile_id", sort=False):
        if str(pid) not in lookup.index:
            continue
        rows.append(
            {
                "fold": int(fold),
                "profile_id": str(pid),
                "y100": float(lookup.loc[str(pid), "stock_0_100"]),
                "p100": stock_to_depth(group, 100.0),
                "block_id": lookup.loc[str(pid), "block_id"],
            }
        )
    return pd.DataFrame(rows)


def _block_mean(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> dict:
    work = pd.DataFrame({"value": values, "block": blocks}).dropna()
    groups = {key: grp["value"].to_numpy(float) for key, grp in work.groupby("block", sort=False)}
    keys = np.array(list(groups))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[key] for key in chosen])))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(work)), "mean": float(work["value"].mean()), "ci95_low": float(low), "ci95_high": float(high)}


def _buffer(layers: pd.DataFrame, meta: pd.DataFrame, valid: pd.DataFrame) -> dict:
    folds = _folds(meta, "block_id")
    locations = meta[["profile_id", "longitude", "latitude"]].drop_duplicates("profile_id")
    results = []
    predictions = []
    for radius in BUFFERS_KM:
        parts = []
        removed = []
        for fold, train_ids, test_ids in folds:
            train_ids = set(map(str, train_ids))
            test_ids = set(map(str, test_ids))
            train_xy = locations.loc[locations["profile_id"].isin(train_ids), ["longitude", "latitude"]].to_numpy(float)
            test_xy = locations.loc[locations["profile_id"].isin(test_ids), ["longitude", "latitude"]].to_numpy(float)
            train_order = locations.loc[locations["profile_id"].isin(train_ids), "profile_id"].astype(str).to_numpy()
            distance = _nearest_km(train_xy, test_xy)
            keep = train_order[distance >= radius]
            removed.append(int(len(train_order) - len(keep)))
            part = _predict_fold(layers, meta, set(keep), test_ids, fold)
            if not part.empty:
                parts.append(part)
            print(f"buffer {radius} fold {fold} train {len(keep)}", flush=True)
        detail = pd.concat(parts, ignore_index=True)
        detail = detail.merge(valid, on="profile_id", how="inner", suffixes=("", "_valid"))
        detail = detail.loc[np.isfinite(detail["y100"]) & np.isfinite(detail["p100"])]
        scored = metric_row(detail["y100"], detail["p100"])
        results.append(
            {
                "buffer_km": radius,
                "mean_train_profiles_removed": float(np.mean(removed)),
                "scored_on_union_valid_0_100": scored,
            }
        )
        detail["buffer_km"] = radius
        predictions.append(detail[["buffer_km", "fold", "profile_id", "block_id", "y100", "p100"]])
    saved = pd.concat(predictions, ignore_index=True)
    rng = np.random.default_rng(SEED)
    base = saved.loc[saved["buffer_km"] == 0.0, ["profile_id", "block_id", "y100", "p100"]].rename(columns={"p100": "p0"})
    paired = {}
    for radius in BUFFERS_KM[1:]:
        other = saved.loc[saved["buffer_km"] == radius, ["profile_id", "p100"]].rename(columns={"p100": "p_buffer"})
        both = base.merge(other, on="profile_id")
        delta = (both["y100"] - both["p_buffer"]).abs() - (both["y100"] - both["p0"]).abs()
        paired[str(radius)] = _block_mean(delta.to_numpy(float), both["block_id"].to_numpy(), rng)
    return {"fits": results, "paired_buffer_minus_zero_buffer": paired, "predictions": saved}


def _biomes(meta: pd.DataFrame) -> pd.DataFrame:
    eco_zip = DATA / "raw" / "ecoregions" / "Ecoregions2017.zip"
    if not eco_zip.exists():
        raise FileNotFoundError(eco_zip)
    import geopandas as gpd

    eco = gpd.read_file(f"zip://{eco_zip}")
    points = gpd.GeoDataFrame(
        meta[["profile_id", "longitude", "latitude", "block_id", "stock_0_100"]].drop_duplicates("profile_id"),
        geometry=gpd.points_from_xy(meta["longitude"], meta["latitude"]),
        crs="EPSG:4326",
    )
    # The read above uses meta order, which can repeat coordinates if the frame
    # was not deduplicated before the geometry call. Rebuild from the table.
    table = meta[["profile_id", "longitude", "latitude", "block_id", "stock_0_100"]].drop_duplicates("profile_id")
    points = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["longitude"], table["latitude"]), crs="EPSG:4326")
    columns = [c for c in ("ECO_NAME", "BIOME_NAME", "geometry") if c in eco.columns or c == "geometry"]
    joined = gpd.sjoin(points, eco[columns], predicate="intersects", how="left")
    joined = joined.drop_duplicates("profile_id")
    return pd.DataFrame(joined.drop(columns="geometry"))


def _biome_holdout(layers: pd.DataFrame, meta: pd.DataFrame, assigned: pd.DataFrame, valid: set[str]) -> dict:
    spec = SPECS["rf"]
    counts = assigned.loc[assigned["profile_id"].isin(valid), "BIOME_NAME"].value_counts(dropna=True)
    held = [name for name, n in counts.items() if n >= MIN_BIOME_TEST and isinstance(name, str)]
    rows = []
    for name in held:
        test_ids = set(assigned.loc[assigned["BIOME_NAME"] == name, "profile_id"].astype(str))
        train_ids = set(meta["profile_id"].astype(str)) - test_ids
        matrix, _names, _medians = _tree_matrix(layers, spec, train_ids)
        train_mask = layers["profile_id"].isin(train_ids).to_numpy()
        test_mask = layers["profile_id"].isin(test_ids).to_numpy()
        model = make_tree(spec, seed=42)
        model.fit(matrix[train_mask], layers.loc[train_mask, "socd_kgm2"].to_numpy(float))
        predicted = model.predict(matrix[test_mask])
        work = layers.loc[test_mask, ["profile_id", "top_cm", "bottom_cm", "has_overlap"]].copy()
        work["socd_kgm2"] = predicted
        records = []
        lookup = meta.set_index("profile_id")
        for pid, group in work.groupby("profile_id", sort=False):
            if str(pid) not in valid:
                continue
            records.append({"y100": float(lookup.loc[str(pid), "stock_0_100"]), "p100": stock_to_depth(group, 100.0)})
        scored = pd.DataFrame(records).dropna()
        row = {"biome": name, "n_profiles_assigned": int(counts[name]), **metric_row(scored["y100"], scored["p100"])}
        rows.append(row)
        print(f"biome {name} n {row['n']} mae {row['mae']:.3f}", flush=True)
    archived = pd.read_parquet(METRICS / "oof_stocks.parquet")
    archived = archived.loc[(archived["model"] == "mcu") & (archived["split"] == "spatial_block")].copy()
    archived["profile_id"] = archived["profile_id"].astype(str)
    archived = archived.merge(assigned[["profile_id", "BIOME_NAME"]], on="profile_id", how="left")
    archived = archived.loc[archived["profile_id"].isin(valid)]
    descriptive = []
    for name, group in archived.groupby("BIOME_NAME", dropna=True):
        if len(group) < 30:
            continue
        descriptive.append({"biome": name, "source": "archived_primary_spatial_blocks", **metric_row(group["y100"], group["p100"])})
    return {
        "minimum_union_valid_profiles": MIN_BIOME_TEST,
        "leave_one_biome_random_forest": rows,
        "archived_primary_by_biome": descriptive,
        "unassigned_profiles": int(assigned["BIOME_NAME"].isna().sum()),
    }


def _figure(summary: dict) -> None:
    paired = summary["buffer"]["paired_buffer_minus_zero_buffer"]
    fits = {float(row["buffer_km"]): row for row in summary["buffer"]["fits"]}
    radii = [25.0, 50.0, 100.0]
    means = np.array([paired[str(radius)]["mean"] for radius in radii])
    low = np.array([paired[str(radius)]["ci95_low"] for radius in radii])
    high = np.array([paired[str(radius)]["ci95_high"] for radius in radii])
    removed = [fits[radius]["mean_train_profiles_removed"] for radius in radii]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].errorbar(radii, means, yerr=[means - low, high - means], fmt="o", color="#2166ac", ecolor="#4d4d4d", capsize=3, ms=7)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xticks(radii)
    axes[0].tick_params(labelsize=12)
    axes[0].set_xlabel("Buffer distance (km)", fontsize=12)
    axes[0].set_ylabel("Change in absolute error (kg m$^{-2}$)", fontsize=12)
    axes[0].set_title("(a)", loc="left", fontsize=13)
    axes[1].bar([str(int(radius)) for radius in radii], removed, color="#4d4d4d", width=0.7)
    axes[1].tick_params(labelsize=12)
    axes[1].set_xlabel("Buffer distance (km)", fontsize=12)
    axes[1].set_ylabel("Mean training profiles removed", fontsize=12)
    axes[1].set_title("(b)", loc="left", fontsize=13)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig22_buffer_forest.png", dpi=300)
    fig.savefig(FIGURES / "fig22_buffer_forest.pdf")
    plt.close(fig)


def run() -> dict:
    layers, profiles = _load()
    meta = _profile_table(layers, profiles)
    meta["profile_id"] = meta["profile_id"].astype(str)
    layers["profile_id"] = layers["profile_id"].astype(str)
    valid = _valid_targets()
    valid["profile_id"] = valid["profile_id"].astype(str)
    buffer = _buffer(layers, meta, valid)
    predictions = buffer.pop("predictions")
    TABLES.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(TABLES / "round17_buffer_predictions.csv", index=False)
    assigned = _biomes(meta)
    assigned.to_csv(TABLES / "round17_biome_assignment.csv", index=False)
    biomes = _biome_holdout(layers, meta, assigned, set(valid["profile_id"]))
    summary = {
        "model": "random forest with the archived fixed settings, refit for this test",
        "replaces_archived_scores": False,
        "buffers_predeclared_km": list(BUFFERS_KM),
        "distance": "equirectangular kilometres, 111.32 km per degree, cosine of the mean latitude of the two sets",
        "scored_profiles": "union-valid 0-100 cm profiles only",
        "buffer": buffer,
        "biomes": biomes,
        "ecoregion_source": "RESOLVE Ecoregions 2017, Dinerstein et al., CC-BY 4.0",
    }
    _figure(summary)
    (METRICS / "round17_spatial.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(buffer["fits"], indent=2))
    return summary


if __name__ == "__main__":
    run()

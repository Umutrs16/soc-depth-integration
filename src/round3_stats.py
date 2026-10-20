"""Third-round checks that do not refit a new architecture.

Duplicate-key audit, WGS84 geodesic distances, and the size of the
random-forest comparison. Buffered cross-validation is not run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyproj import Geod
from sklearn.neighbors import NearestNeighbors

from .experiment import _folds, _load, _nearest_km, _profile_table
from .paths import METRICS, PROCESSED, TABLES
from .qc import load_v10

GEOD = Geod(ellps="WGS84")


def _duplicate_audit() -> dict:
    raw = load_v10()
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    block = profiles.set_index("profile_id")["block_id"].to_dict()
    raw["profile_id"] = raw["Profile"].astype(str)
    raw["key"] = list(
        zip(
            raw["Latitude"].round(4),
            raw["Longitude"].round(4),
            raw["Upper_depth_cm"].round(1),
            raw["Lower_depth_cm"].round(1),
        )
    )
    counts = raw.groupby("key", sort=False).size()
    dup_keys = counts[counts > 1].index
    same_profile = diff_profile = identical_socd = identical_all = 0
    different_year = different_reference = 0
    cross_block_keys = 0
    cross_block_profiles = set()
    for _key, g in raw.loc[raw["key"].isin(dup_keys)].groupby("key", sort=False):
        ids = g["profile_id"].astype(str).unique()
        if len(ids) == 1:
            same_profile += 1
        else:
            diff_profile += 1
            blocks = {block.get(pid) for pid in ids}
            blocks.discard(None)
            if len(blocks) > 1:
                cross_block_keys += 1
                cross_block_profiles.update(ids)
        if g["SOCD_kg_m_2"].nunique(dropna=False) == 1:
            identical_socd += 1
        soc_same = g["SOC_g_kg"].nunique(dropna=False) == 1
        bd_same = g["BD_g_cm3"].nunique(dropna=False) == 1
        if soc_same and bd_same and g["SOCD_kg_m_2"].nunique(dropna=False) == 1:
            identical_all += 1
        if "Time" in g.columns and g["Time"].nunique(dropna=False) > 1:
            different_year += 1
        if "References" in g.columns and g["References"].nunique(dropna=False) > 1:
            different_reference += 1
    extra = int((counts - 1).clip(lower=0).sum())
    return {
        "rows": int(len(raw)),
        "extra_rows": extra,
        "extra_row_fraction": float(extra / len(raw)),
        "repeat_keys": int(len(dup_keys)),
        "same_profile_id": int(same_profile),
        "different_profile_id": int(diff_profile),
        "identical_socd": int(identical_socd),
        "identical_soc_bd_socd": int(identical_all),
        "different_year": int(different_year),
        "different_reference": int(different_reference),
        "repeat_keys_in_more_than_one_block": int(cross_block_keys),
        "profiles_in_those_cross_block_keys": int(len(cross_block_profiles)),
        "note": "Rows were not removed and models were not retrained. Cross-block keys are the only repeats that can sit on both sides of a spatial fold.",
    }


def _geodesic() -> dict:
    _layers, profiles = _load()
    meta = _profile_table(_layers, profiles)
    folds = _folds(meta, "block_id")
    saved = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    saved["profile_id"] = saved["profile_id"].astype(str)
    rows = []
    for fold, train_ids, test_ids in folds:
        test = meta.loc[meta["profile_id"].isin(test_ids)].copy()
        train = meta.loc[meta["profile_id"].isin(train_ids)]
        test_xy = test[["longitude", "latitude"]].to_numpy(float)
        train_xy = train[["longitude", "latitude"]].to_numpy(float)
        equi = _nearest_km(test_xy, train_xy)
        lat0 = np.deg2rad(np.nanmean(np.concatenate([test_xy[:, 1], train_xy[:, 1]])))
        def project(xy):
            return np.column_stack([xy[:, 0] * np.cos(lat0) * 111.32, xy[:, 1] * 111.32])
        nn = NearestNeighbors(n_neighbors=min(8, len(train)), algorithm="kd_tree")
        nn.fit(project(train_xy))
        _dist, index = nn.kneighbors(project(test_xy))
        geo = np.empty(len(test), dtype=float)
        for i in range(len(test)):
            nbr = train_xy[index[i]]
            _az1, _az2, metres = GEOD.inv(
                np.full(len(nbr), test_xy[i, 0]),
                np.full(len(nbr), test_xy[i, 1]),
                nbr[:, 0],
                nbr[:, 1],
            )
            geo[i] = float(np.min(metres) / 1000.0)
        part = test[["profile_id"]].copy()
        part["profile_id"] = part["profile_id"].astype(str)
        part["fold"] = fold
        part["equirectangular_km"] = equi
        part["geodesic_km"] = geo
        rows.append(part)
    detail = pd.concat(rows, ignore_index=True)
    detail = detail.merge(saved[["profile_id", "fold", "nn_km"]], on=["profile_id", "fold"], how="left")
    bins = [0, 25, 50, 100, np.inf]
    labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    detail["bin_equi"] = pd.cut(detail["equirectangular_km"], bins=bins, labels=labels, right=False)
    detail["bin_geo"] = pd.cut(detail["geodesic_km"], bins=bins, labels=labels, right=False)
    changed = detail["bin_equi"].astype(str) != detail["bin_geo"].astype(str)
    TABLES.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "distance_geodesic.csv", index=False)
    return {
        "n": int(len(detail)),
        "median_abs_km": float((detail["geodesic_km"] - detail["equirectangular_km"]).abs().median()),
        "p95_abs_km": float((detail["geodesic_km"] - detail["equirectangular_km"]).abs().quantile(0.95)),
        "max_abs_km": float((detail["geodesic_km"] - detail["equirectangular_km"]).abs().max()),
        "profiles_changing_bin": int(changed.sum()),
        "fraction_changing_bin": float(changed.mean()),
        "note": "Geodesic distance is the WGS84 minimum among the eight nearest equirectangular neighbours. It is not a buffer cross-validation.",
    }


def _effect() -> dict:
    folds = pd.read_csv(METRICS / "cv_fold_metrics.csv")
    part = folds.loc[(folds["split"] == "spatial_block") & (folds["target"] == "stock_0_100")]
    mcu = part.loc[part["model"] == "mcu"].set_index("fold")
    rf = part.loc[part["model"] == "rf"].set_index("fold")
    delta = (rf["mae"] - mcu["mae"]).sort_index()
    pooled = json.loads((METRICS / "round2_stats.json").read_text(encoding="utf-8"))
    paired = next(r for r in json.loads((METRICS / "revision_stats.json").read_text(encoding="utf-8"))["paired_block_bootstrap"] if r["target"] == "stock_0_100" and r["baseline"] == "rf")
    return {
        "fold_mae_rf_minus_mcu": {str(int(i)): float(v) for i, v in delta.items()},
        "fold_mean_reduction": float(delta.mean()),
        "pooled_mae_note": "The archived eight-point 0-100 cm error pooled over profiles is 4.361 kg m-2. Table 2 averages the five fold MAEs and is 4.357.",
        "paired_mean": paired["mean_baseline_minus_mcu"],
        "paired_ci": [paired["ci95_low"], paired["ci95_high"]],
        "reduction_as_fraction_of_rf_fold_mean_mae": float(delta.mean() / rf["mae"].mean()),
        "ci_low_as_fraction_of_rf_error": float(paired["ci95_low"] / rf["mae"].mean()),
        "distance_bins_unchanged_source": "round2_stats.json distance_bin_paired",
        "comparisons_in_table_3": 8,
        "multiplicity": "Intervals are marginal block-bootstrap intervals. They are not adjusted for the number of comparisons.",
    }


def run() -> dict:
    payload = {"duplicates": _duplicate_audit(), "geodesic": _geodesic(), "effect": _effect()}
    (METRICS / "round3_stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()

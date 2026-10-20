"""Second-round diagnostics that do not refit models.

Distance bins use the saved 2° folds. They are not a buffer cross-validation:
nearby profiles in neighbouring blocks can still train the test prediction.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .experiment import _folds, _load, _nearest_km, _profile_features, _profile_table, _split_fit_cal_val
from .models import SPECS
from .paths import METRICS, PROCESSED, TABLES
from .qc import load_v10

WC = {
    10: "tree",
    20: "shrub",
    30: "grass",
    40: "cropland",
    50: "built",
    60: "bare",
    70: "snow",
    80: "water",
    90: "wetland",
    95: "mangrove",
    100: "moss",
}


def _mae_bias(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() == 0:
        return {"n": 0, "mae": None, "bias": None, "r2": None}
    err = p[ok] - y[ok]
    ss = np.sum((y[ok] - y[ok].mean()) ** 2)
    r2 = float(1.0 - np.sum(err**2) / ss) if ss > 0 else None
    return {"n": int(ok.sum()), "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err)), "r2": r2}


def _picp(y, lo, hi) -> dict:
    y = np.asarray(y, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if ok.sum() == 0:
        return {"n": 0}
    cover = (y[ok] >= lo[ok]) & (y[ok] <= hi[ok])
    return {
        "n": int(ok.sum()),
        "picp": float(cover.mean()),
        "fraction_lo_negative": float(np.mean(lo[ok] < 0)),
        "mpiw": float(np.mean(hi[ok] - lo[ok])),
    }


def _distance_frame(layers, profile_meta) -> pd.DataFrame:
    folds = _folds(profile_meta, "block_id")
    spec = SPECS["mcu"]
    parts = []
    for fold, train_ids, test_ids in folds:
        test = profile_meta.loc[profile_meta["profile_id"].isin(test_ids)].copy()
        train = profile_meta.loc[profile_meta["profile_id"].isin(train_ids)]
        dist = _nearest_km(
            test[["longitude", "latitude"]].to_numpy(float),
            train[["longitude", "latitude"]].to_numpy(float),
        )
        test = test.copy()
        test["fold"] = fold
        test["nn_km"] = dist
        fit_ids, _val, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
        train_x = np.stack([lookup[str(pid)] for pid in train["profile_id"] if str(pid) in lookup])
        test_ids_ok = [str(pid) for pid in test["profile_id"] if str(pid) in lookup]
        test_x = np.stack([lookup[pid] for pid in test_ids_ok])
        nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
        nn.fit(train_x)
        cov_dist, _ = nn.kneighbors(test_x)
        cov_map = dict(zip(test_ids_ok, cov_dist[:, 0]))
        test["covariate_nn"] = test["profile_id"].astype(str).map(cov_map)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def _bin_metrics(frame: pd.DataFrame, ycol: str, pcol: str) -> list[dict]:
    bins = [0, 25, 50, 100, np.inf]
    labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    work = frame.loc[np.isfinite(frame[ycol]) & np.isfinite(frame[pcol])].copy()
    work["bin"] = pd.cut(work["nn_km"], bins=bins, labels=labels, right=False)
    rows = []
    for label, g in work.groupby("bin", observed=False):
        stat = _mae_bias(g[ycol], g[pcol])
        stat.update({"bin": str(label), "target": ycol, "median_nn_km": float(g["nn_km"].median()) if len(g) else None})
        rows.append(stat)
    return rows


def _interval_detail(stocks: pd.DataFrame, layers: pd.DataFrame) -> dict:
    spatial = stocks.loc[(stocks["model"] == "mcu") & (stocks["split"] == "spatial_block")].copy()
    out = {}
    for target, y, lo, hi in (("0–30 cm", "y30", "lo30", "hi30"), ("0–100 cm", "y100", "lo100", "hi100")):
        raw = _picp(spatial[y], spatial[lo], spatial[hi])
        clipped = _picp(spatial[y], np.maximum(spatial[lo], 0), spatial[hi])
        finite = spatial.loc[np.isfinite(spatial[y]) & np.isfinite(spatial[lo]) & np.isfinite(spatial[hi])]
        if len(finite):
            threshold = float(np.quantile(finite[y], 0.9))
            tail = finite.loc[finite[y] >= threshold]
            tail_stat = _picp(tail[y], tail[lo], tail[hi])
            tail_stat["threshold"] = threshold
        else:
            tail_stat = {"n": 0}
        out[target] = {"raw": raw, "lo_clipped_at_zero": clipped, "upper_decile": tail_stat}
    layer = layers.loc[(layers["model"] == "mcu") & (layers["split"] == "spatial_block")].copy()
    depth_rows = []
    for depth, g in layer.groupby("depth_bin"):
        stat = _picp(g["socd_kgm2"], g["lo"], g["hi"])
        finite = g.loc[np.isfinite(g["socd_kgm2"])]
        if len(finite):
            thr = float(np.quantile(finite["socd_kgm2"], 0.9))
            tail = finite.loc[finite["socd_kgm2"] >= thr]
            tail_stat = _picp(tail["socd_kgm2"], tail["lo"], tail["hi"])
        else:
            thr, tail_stat = None, {"n": 0}
        stat.update({"depth_bin": str(depth), "upper_decile_threshold": thr, "upper_decile": tail_stat})
        depth_rows.append(stat)
    out["layers_by_depth"] = depth_rows
    return out


def _duplicate_keys() -> dict:
    raw = load_v10()
    raw["key"] = list(
        zip(
            raw["Latitude"].round(4),
            raw["Longitude"].round(4),
            raw["Upper_depth_cm"].round(1),
            raw["Lower_depth_cm"].round(1),
        )
    )
    counts = raw.groupby("key", sort=False).size()
    extra = int((counts - 1).clip(lower=0).sum())
    dup_keys = set(counts[counts > 1].index)
    dup = raw.loc[raw["key"].isin(dup_keys)].copy()
    same_profile = 0
    same_socd = 0
    same_soc_bd = 0
    different_profile = 0
    for _key, g in dup.groupby("key", sort=False):
        profiles = g["Profile"].nunique()
        if profiles == 1:
            same_profile += 1
        else:
            different_profile += 1
        if g["SOCD_kg_m_2"].nunique(dropna=False) == 1:
            same_socd += 1
        soc = g["SOC_g_kg"]
        bd = g["BD_g_cm3"]
        if soc.nunique(dropna=False) == 1 and bd.nunique(dropna=False) == 1:
            same_soc_bd += 1
    return {
        "rows": int(len(raw)),
        "unique_keys": int(counts.size),
        "extra_rows_beyond_unique_keys": extra,
        "keys_with_repeats": int(len(dup_keys)),
        "repeat_keys_same_profile_id": int(same_profile),
        "repeat_keys_different_profile_id": int(different_profile),
        "repeat_keys_identical_socd": int(same_socd),
        "repeat_keys_identical_soc_and_bd": int(same_soc_bd),
        "note": "A repeated key is latitude/longitude to 4 decimal degrees and layer bounds to 0.1 cm. Identical fields are not proof of an independent remeasurement.",
    }


def run() -> dict:
    layers, profiles = _load()
    profile_meta = _profile_table(layers, profiles)
    dist = _distance_frame(layers, profile_meta)
    stocks = pd.read_parquet(METRICS / "oof_stocks.parquet")
    oof_layers = pd.read_parquet(METRICS / "oof_layers.parquet")
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    dist["profile_id"] = dist["profile_id"].astype(str)
    spatial = stocks.loc[stocks["split"] == "spatial_block"].copy()
    mcu = spatial.loc[spatial["model"] == "mcu"].merge(
        dist[["profile_id", "fold", "nn_km", "covariate_nn", "reference", "block_id"]],
        on=["profile_id", "fold"],
        how="left",
    )
    rf = spatial.loc[spatial["model"] == "rf", ["profile_id", "fold", "p30", "p100"]].rename(
        columns={"p30": "p30_rf", "p100": "p100_rf"}
    )
    mcu = mcu.merge(rf, on=["profile_id", "fold"], how="left")
    distance_gap = []
    bins = [0, 25, 50, 100, np.inf]
    labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    gap_frame = mcu.loc[np.isfinite(mcu["y100"]) & np.isfinite(mcu["p100"]) & np.isfinite(mcu["p100_rf"])].copy()
    gap_frame["delta"] = (gap_frame["y100"] - gap_frame["p100_rf"]).abs() - (gap_frame["y100"] - gap_frame["p100"]).abs()
    gap_frame["bin"] = pd.cut(gap_frame["nn_km"], bins=bins, labels=labels, right=False)
    rng = np.random.default_rng(42)
    for label, g in gap_frame.groupby("bin", observed=False):
        groups = {key: grp["delta"].to_numpy() for key, grp in g.groupby("block_id")}
        keys = np.array(list(groups))
        draws = np.empty(2000)
        for i in range(2000):
            chosen = rng.choice(keys, size=len(keys), replace=True)
            draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        distance_gap.append(
            {
                "bin": str(label),
                "n": int(len(g)),
                "n_blocks": int(len(keys)),
                "mean_rf_minus_mcu": float(g["delta"].mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
            }
        )
    distance_rows = []
    for model, p30, p100 in (("mcu", "p30", "p100"), ("rf", "p30_rf", "p100_rf")):
        for row in _bin_metrics(mcu, "y30", p30):
            row["model"] = model
            distance_rows.append(row)
        for row in _bin_metrics(mcu, "y100", p100):
            row["model"] = model
            distance_rows.append(row)
    cov_rows = []
    finite = mcu.loc[np.isfinite(mcu["y100"]) & np.isfinite(mcu["covariate_nn"])].copy()
    if len(finite):
        finite["cov_bin"] = pd.qcut(finite["covariate_nn"], 4, duplicates="drop")
        for label, g in finite.groupby("cov_bin", observed=False):
            stat = _mae_bias(g["y100"], g["p100"])
            stat.update({"covariate_bin": str(label), "model": "mcu"})
            cov_rows.append(stat)
            stat_rf = _mae_bias(g["y100"], g["p100_rf"])
            stat_rf.update({"covariate_bin": str(label), "model": "rf"})
            cov_rows.append(stat_rf)
    qc = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    qc["sample_id"] = qc["sample_id"].astype(str)
    layer = oof_layers.loc[(oof_layers["model"] == "mcu") & (oof_layers["split"] == "spatial_block")].copy()
    layer["sample_id"] = layer["sample_id"].astype(str)
    layer = layer.merge(qc[["sample_id", "reliability", "method"]], on="sample_id", how="left")
    reliability_rows = []
    for col in ("reliability", "method", "depth_bin"):
        for key, g in layer.groupby(col):
            stat = _mae_bias(g["socd_kgm2"], g["pred"])
            stat.update({"group": col, "level": str(key)})
            reliability_rows.append(stat)
    first = layers.groupby("profile_id", sort=False)["worldcover"].first()
    support = profiles.copy()
    support["profile_id"] = support["profile_id"].astype(str)
    support["worldcover"] = support["profile_id"].map(first)
    support["landcover"] = support["worldcover"].round().map(WC)
    cover_rows = []
    for flag, name in (("cover_0_30", "0–30 cm"), ("cover_0_100", "0–100 cm")):
        sub = support.loc[support[flag].astype(bool)]
        counts = sub["landcover"].fillna("unknown").value_counts()
        refs = sub["reference"].fillna("unknown").value_counts().head(8)
        cover_rows.append(
            {
                "target": name,
                "n": int(len(sub)),
                "landcover": {str(k): int(v) for k, v in counts.items()},
                "top_references": {str(k): int(v) for k, v in refs.items()},
            }
        )
    fold_n = (
        mcu.groupby("fold")
        .agg(profiles=("profile_id", "size"), n30=("y30", lambda s: int(np.isfinite(s).sum())), n100=("y100", lambda s: int(np.isfinite(s).sum())))
        .reset_index()
    )
    point = mcu[
        [
            "profile_id",
            "fold",
            "block_id",
            "province_en",
            "reference",
            "nn_km",
            "covariate_nn",
            "y30",
            "p30",
            "y100",
            "p100",
            "lo100",
            "hi100",
            "p30_rf",
            "p100_rf",
        ]
    ].copy()
    TABLES.mkdir(parents=True, exist_ok=True)
    point.to_csv(TABLES / "oof_spatial_stocks.csv", index=False)
    payload = {
        "distance_bins": distance_rows,
        "distance_bin_paired": distance_gap,
        "covariate_distance_quartiles": cov_rows,
        "intervals": _interval_detail(stocks, oof_layers),
        "layer_groups": reliability_rows,
        "stock_composition": cover_rows,
        "fold_counts": fold_n.to_dict(orient="records"),
        "duplicate_keys": _duplicate_keys(),
        "not_run": [
            "50/100/200 km buffer cross-validation",
            "leave-one-ecoregion and additional geographic-transfer refits",
            "nested hyperparameter search, CatBoost, GAM, and mass-preserving splines",
            "soil-thickness, rock-outcrop, and coarse-fragment scenarios",
            "spatially correlated simulation of the national petagram sum",
            "conditional conformal or CQR",
        ],
    }
    (METRICS / "round2_stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


if __name__ == "__main__":
    run()

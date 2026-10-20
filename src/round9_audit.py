"""Source-label audit, tail sensitivity, and an RF split-conformal interval.

The random-forest interval is a new fit. It does not replace archived scores.
The finite-sample residual quantile is the split-conformal formula. Spatial
blocks are not an exchangeable sample, so the nominal 90% is not a guarantee.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .experiment import _folds, _load, _profile_table, _split_fit_cal_val, _tree_matrix
from .models import ModelSpec, make_tree
from .paths import METRICS, TABLES
from .qc import stock_to_depth


def _modal_reference(series: pd.Series) -> str:
    modes = series.astype(str).mode()
    return str(modes.iloc[0]) if len(modes) else ""


def _finite_quantile(residuals: np.ndarray) -> tuple[float, float]:
    ordered = np.sort(residuals[np.isfinite(residuals)])
    n = len(ordered)
    hi = min(n, max(1, int(np.ceil((n + 1) * 0.95))))
    lo = min(n, max(1, int(np.floor((n + 1) * 0.05))))
    return float(ordered[lo - 1]), float(ordered[hi - 1])


def _stocks_from_prediction(layers, pred_by_pid, meta, ids) -> pd.DataFrame:
    rows = []
    for pid in ids:
        key = str(pid)
        if key not in pred_by_pid or key not in meta.index:
            continue
        group = layers.loc[layers["profile_id"] == key].copy()
        pred = np.asarray(pred_by_pid[key], dtype=float)
        if len(pred) != len(group):
            continue
        observed = meta.loc[key, "stock_0_100"]
        if not np.isfinite(observed):
            continue
        group["socd_kgm2"] = pred
        rows.append(
            {
                "profile_id": key,
                "block_id": str(meta.loc[key, "block_id"]),
                "y": float(observed),
                "p": float(stock_to_depth(group, 100)),
            }
        )
    return pd.DataFrame(rows)


def _fit_rf_stocks(layers, profile_meta, train_ids, test_ids, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_ids, _val_ids, cal_ids, tree_ids = _split_fit_cal_val(profile_meta, np.asarray(list(train_ids)), fold)
    spec = ModelSpec("rf", "tree")
    matrix, _names, _med = _tree_matrix(layers, spec, tree_ids)
    train_mask = layers["profile_id"].isin(tree_ids).to_numpy()
    model = make_tree(spec, seed=42)
    model.fit(matrix[train_mask], layers.loc[train_mask, "socd_kgm2"].to_numpy(float))
    meta = profile_meta.set_index(profile_meta["profile_id"].astype(str))

    def predict(ids):
        mask = layers["profile_id"].isin(set(map(str, ids))).to_numpy()
        pred = model.predict(matrix[mask])
        subset = layers.loc[mask].copy()
        subset["pred"] = pred
        return {str(pid): g["pred"].to_numpy(float) for pid, g in subset.groupby("profile_id", sort=False)}

    cal = _stocks_from_prediction(layers, predict(cal_ids), meta, cal_ids)
    test = _stocks_from_prediction(layers, predict(test_ids), meta, test_ids)
    return cal, test


def _coverage(cal: pd.DataFrame, test: pd.DataFrame) -> dict:
    resid = (cal["y"] - cal["p"]).to_numpy(float)
    q_lo, q_hi = _finite_quantile(resid)
    ordinary = np.quantile(resid, [0.05, 0.95])
    out = test.copy()
    out["hit_finite"] = (out["y"] >= out["p"] + q_lo) & (out["y"] <= out["p"] + q_hi)
    out["hit_ordinary"] = (out["y"] >= out["p"] + ordinary[0]) & (out["y"] <= out["p"] + ordinary[1])
    out["width_finite"] = q_hi - q_lo
    out["decile"] = pd.qcut(out["p"], 10, labels=False, duplicates="drop") + 1
    deciles = []
    for dec, group in out.groupby("decile"):
        deciles.append(
            {
                "predicted_decile": int(dec),
                "n": int(len(group)),
                "picp_finite_sample": float(group["hit_finite"].mean()),
                "picp_ordinary_quantile": float(group["hit_ordinary"].mean()),
                "width_finite_sample": float(q_hi - q_lo),
            }
        )
    return {
        "n_calibration": int(len(cal)),
        "n_test": int(len(out)),
        "finite_sample_residual_quantile_low": q_lo,
        "finite_sample_residual_quantile_high": q_hi,
        "ordinary_quantile_low": float(ordinary[0]),
        "ordinary_quantile_high": float(ordinary[1]),
        "picp_finite_sample": float(out["hit_finite"].mean()),
        "picp_ordinary_quantile": float(out["hit_ordinary"].mean()),
        "width_finite_sample": float(q_hi - q_lo),
        "width_ordinary": float(ordinary[1] - ordinary[0]),
        "mae": float((out["y"] - out["p"]).abs().mean()),
        "deciles": deciles,
    }


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def run() -> None:
    layers, profiles = _load()
    layers["profile_id"] = layers["profile_id"].astype(str)
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    conflict = layers.loc[layers["profile_id"].isin(["2848", "2986", "3719"])].sort_values(["profile_id", "top_cm", "bottom_cm"])
    keep_cols = [c for c in ("profile_id", "top_cm", "bottom_cm", "reference", "year", "soc_gkg", "bd_gcm3", "socd_kgm2", "socd_method", "SOCD_methods", "worldcover") if c in conflict.columns]
    conflict[keep_cols].to_csv(TABLES / "round9_conflict_layers.csv", index=False)
    nunique = layers.groupby("profile_id")["reference"].nunique()
    mixed_ids = set(nunique[nunique > 1].index.astype(str))
    shallow = layers.groupby("profile_id", sort=False)["reference"].first().astype(str)
    modal = layers.groupby("profile_id")["reference"].agg(_modal_reference)
    liu_shallow = shallow.str.contains("Liu et al., 2022", regex=False)
    liu_modal = modal.str.contains("Liu et al., 2022", regex=False)
    pure = ~pd.Series(modal.index.astype(str), index=modal.index).isin(mixed_ids)
    counts = {
        "n_profiles": int(modal.size),
        "n_mixed_reference_profiles": int(len(mixed_ids)),
        "liu_shallowest_layer": int(liu_shallow.sum()),
        "liu_modal": int(liu_modal.sum()),
        "liu_modal_excluding_mixed": int((liu_modal & pure).sum()),
        "profiles_whose_2022_label_changes_between_shallowest_and_modal": sorted(
            set(liu_shallow.index.astype(str)[liu_shallow.to_numpy()]) ^ set(liu_modal.index.astype(str)[liu_modal.to_numpy()])
        ),
        "shallowest_means": "after _load sorts by top, bottom and sample id, groupby first is the shallowest layer",
    }
    (METRICS / "round9_source_counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")

    stocks = pd.read_parquet(METRICS / "oof_stocks.parquet")
    stocks["profile_id"] = stocks["profile_id"].astype(str)

    def tail(frame: pd.DataFrame, name: str) -> dict:
        base = frame.dropna(subset=["y100", "p100"]).copy()
        drop_one = base.loc[base["profile_id"] != "3637"]
        threshold = float(np.quantile(base["y100"], 0.99))
        drop_tail = base.loc[base["y100"] < threshold]
        return {
            "split_model": name,
            "n": int(len(base)),
            "mae": _mae(base["y100"], base["p100"]),
            "mae_without_3637": _mae(drop_one["y100"], drop_one["p100"]) if len(drop_one) else None,
            "n_without_3637": int(len(drop_one)),
            "mae_below_99th_percentile_of_this_set": _mae(drop_tail["y100"], drop_tail["p100"]),
            "n_below_99th": int(len(drop_tail)),
            "threshold_99th": threshold,
            "model_was_not_retrained": True,
        }

    tail_rows = []
    for split, model in (("xinjiang_holdout", "mcu"), ("xinjiang_holdout", "rf"), ("spatial_block", "mcu"), ("spatial_block", "rf")):
        part = stocks.loc[(stocks["split"] == split) & (stocks["model"] == model)]
        tail_rows.append(tail(part, f"{split}:{model}"))
    profile_3637 = layers.loc[layers["profile_id"] == "3637"].sort_values(["top_cm", "bottom_cm"])
    profile_3637.to_csv(TABLES / "round9_profile_3637_layers.csv", index=False)
    closure = []
    for row in profile_3637.itertuples(index=False):
        soc = getattr(row, "soc_gkg", np.nan)
        bd = getattr(row, "bd_gcm3", np.nan)
        thick = float(row.bottom_cm - row.top_cm)
        product = 0.01 * float(soc) * float(bd) * thick if np.isfinite(soc) and np.isfinite(bd) else np.nan
        closure.append(
            {
                "top_cm": float(row.top_cm),
                "bottom_cm": float(row.bottom_cm),
                "soc_gkg": None if not np.isfinite(soc) else float(soc),
                "bd_gcm3": None if not np.isfinite(bd) else float(bd),
                "socd_kgm2": float(row.socd_kgm2),
                "product_0.01_soc_bd_thickness": None if not np.isfinite(product) else product,
            }
        )
    (METRICS / "round9_tail_sensitivity.json").write_text(
        json.dumps({"scores": tail_rows, "profile_3637_internal_product": closure, "source_table_not_checked": True}, indent=2),
        encoding="utf-8",
    )

    spatial = stocks.loc[(stocks["split"] == "spatial_block") & (stocks["model"] == "mcu")].dropna(subset=["y100", "p100"]).copy()
    spatial = spatial.merge(modal.rename("modal_reference"), left_on="profile_id", right_index=True, how="left")
    spatial["liu"] = spatial["modal_reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
    if "block_id" not in spatial.columns:
        spatial = spatial.merge(profiles[["profile_id", "block_id"]], on="profile_id", how="left")
    rng = np.random.default_rng(9)
    blocks = spatial["block_id"].astype(str).unique()

    def boot_mae(frame: pd.DataFrame) -> dict:
        draws = []
        for _ in range(2000):
            chosen = rng.choice(blocks, size=len(blocks), replace=True)
            parts = [frame.loc[frame["block_id"].astype(str) == b] for b in chosen]
            sample = pd.concat(parts, ignore_index=True) if parts else frame.iloc[0:0]
            if len(sample) == 0:
                continue
            draws.append(_mae(sample["y100"], sample["p100"]))
        return {
            "n": int(len(frame)),
            "mae": _mae(frame["y100"], frame["p100"]),
            "block_bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
            "block_bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
        }

    source_mae = {
        "rule": "modal reference, archived spatial-block predictions, model not retrained",
        "liu_2022": boot_mae(spatial.loc[spatial["liu"]]),
        "other_sources": boot_mae(spatial.loc[~spatial["liu"]]),
    }
    (METRICS / "round9_source_spatial_mae.json").write_text(json.dumps(source_mae, indent=2), encoding="utf-8")

    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    print("round9 rf split conformal", flush=True)
    fold_rows = []
    test_parts = []
    cal_parts = []
    for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
        print(f"rf conformal fold {fold}", flush=True)
        cal, test = _fit_rf_stocks(layers, profile_meta, train_ids, test_ids, fold)
        cal["fold"] = fold
        test["fold"] = fold
        cal_parts.append(cal)
        test_parts.append(test)
        fold_rows.append({"fold": fold, **_coverage(cal, test)})
    all_cal = pd.concat(cal_parts, ignore_index=True)
    all_test = pd.concat(test_parts, ignore_index=True)
    # Recompute one pooled interval only for the decile plot of fold-wise intervals.
    pooled_hits = []
    for fold, group in all_test.groupby("fold"):
        cal = all_cal.loc[all_cal["fold"] == fold]
        q_lo, q_hi = _finite_quantile((cal["y"] - cal["p"]).to_numpy(float))
        hit = (group["y"] >= group["p"] + q_lo) & (group["y"] <= group["p"] + q_hi)
        item = group.copy()
        item["hit"] = hit.to_numpy()
        item["width"] = q_hi - q_lo
        pooled_hits.append(item)
    pooled = pd.concat(pooled_hits, ignore_index=True)
    pooled["decile"] = pd.qcut(pooled["p"], 10, labels=False, duplicates="drop") + 1
    deciles = []
    for dec, group in pooled.groupby("decile"):
        deciles.append(
            {
                "predicted_decile": int(dec),
                "n": int(len(group)),
                "picp_finite_sample": float(group["hit"].mean()),
                "mean_width": float(group["width"].mean()),
            }
        )
    xj_train = profile_meta.loc[~profile_meta["is_xinjiang"].fillna(False), "profile_id"]
    xj_test = profile_meta.loc[profile_meta["is_xinjiang"].fillna(False), "profile_id"]
    print("rf conformal xinjiang", flush=True)
    xj_cal, xj_test_pred = _fit_rf_stocks(layers, profile_meta, xj_train, xj_test, 21)
    modal_flag = modal.str.contains("Liu et al., 2022", regex=False)
    liu_ids = set(modal_flag.index.astype(str)[modal_flag.to_numpy()])
    print("rf conformal source", flush=True)
    src_cal, src_test = _fit_rf_stocks(
        layers,
        profile_meta,
        profile_meta.loc[~profile_meta["profile_id"].isin(liu_ids), "profile_id"],
        profile_meta.loc[profile_meta["profile_id"].isin(liu_ids), "profile_id"],
        31,
    )
    payload = {
        "label": "random-forest split residual interval; calibration blocks were not used to fit the forest; finite-sample order statistics use ceil((n+1)*0.95) and floor((n+1)*0.05); spatial blocks are not exchangeable, so this is not a coverage guarantee",
        "not_the_archived_neural_interval": True,
        "spatial_folds": [{k: v for k, v in row.items() if k != "deciles"} for row in fold_rows],
        "spatial_pooled": {
            "n": int(len(pooled)),
            "picp_finite_sample": float(pooled["hit"].mean()),
            "mean_width": float(pooled["width"].mean()),
            "mae": float((pooled["y"] - pooled["p"]).abs().mean()),
            "deciles": deciles,
        },
        "xinjiang": _coverage(xj_cal, xj_test_pred),
        "train_other_test_liu_modal": _coverage(src_cal, src_test),
    }
    (METRICS / "round9_conformal_rf.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"spatial_picp": payload["spatial_pooled"]["picp_finite_sample"], "xinjiang_picp": payload["xinjiang"]["picp_finite_sample"]}, indent=2), flush=True)


if __name__ == "__main__":
    run()

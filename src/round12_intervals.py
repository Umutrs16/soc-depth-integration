"""Rebuild stock-interval quantiles on union-valid calibration profiles.

The archived calibration predictions and network weights were not saved.
This script refits the primary integral and the fixed random forest on the
same 2° folds and seeds, drops profiles that fail the union coverage test,
and recomputes residual quantiles. It also refits the map calibration split.
New files only: it does not replace oof_stocks.parquet or the national raster.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import (
    _folds,
    _load,
    _profile_features,
    _profile_table,
    _split_fit_cal_val,
    _stock_predictions,
    _tree_matrix,
)
from .metrics import interval_metrics, metric_row
from .models import SPECS, make_tree, predict_stock_batch, train_mcu
from .paths import METRICS, TABLES
from .round11_coverage import _classify

SEED_NOTE = "mcu seeds 42, 43, 44; random forest seed 42; fold splitter is GroupKFold; inner blocks use Generator seeds 1000+fold and 2000+fold"


def _valid_sets(classified: pd.DataFrame) -> dict[int, set[str]]:
    out = {}
    for depth in (30, 100, 200):
        part = classified.loc[classified["depth_cm"] == depth]
        out[depth] = set(part.loc[part["status"].isin(["exact_boundary", "prorated"]), "profile_id"].astype(str))
    return out


def _status_map(classified: pd.DataFrame, depth: float) -> dict[str, str]:
    part = classified.loc[classified["depth_cm"] == depth, ["profile_id", "status"]]
    return dict(zip(part["profile_id"].astype(str), part["status"].astype(str)))


def _quantiles(frame: pd.DataFrame, ycol: str, pcol: str) -> dict:
    use = frame[[ycol, pcol]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(use) < 20:
        q05, q95 = -1.0, 1.0
    else:
        q05, q95 = np.quantile((use[ycol] - use[pcol]).to_numpy(float), [0.05, 0.95])
    return {"n": int(len(use)), "q05": float(q05), "q95": float(q95), "width": float(q95 - q05)}


def _mask_invalid(frame: pd.DataFrame, valid30: set[str], valid100: set[str]) -> pd.DataFrame:
    out = frame.copy()
    out.loc[~out["profile_id"].isin(valid30), "y30"] = np.nan
    out.loc[~out["profile_id"].isin(valid100), "y100"] = np.nan
    return out


def _attach(cal: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = test.copy()
    quant = {}
    for pred, y, lo_name, hi_name, key in (
        ("p30", "y30", "lo30", "hi30", "30"),
        ("p100", "y100", "lo100", "hi100", "100"),
    ):
        stats = _quantiles(cal, y, pred)
        quant[key] = stats
        out[lo_name] = out[pred] + stats["q05"]
        out[hi_name] = out[pred] + stats["q95"]
    return out, quant


def _mcu_predict(layers, profile_meta, fit_ids, val_ids, cal_ids, test_ids, fold: int):
    spec = SPECS["mcu"]
    lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in spec.seeds:
        print(f"mcu fold {fold} seed {seed}", flush=True)
        model, _max_layers, device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
        models.append(model)
    cal, _ = _stock_predictions(layers, profile_meta, spec, cal_ids, None, models, lookup, device, "round12_refit", fold)
    test, _ = _stock_predictions(layers, profile_meta, spec, test_ids, None, models, lookup, device, "round12_refit", fold)
    del models
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cal, test


def _rf_predict(layers, profile_meta, tree_ids, cal_ids, test_ids, fold: int):
    print(f"rf fold {fold}", flush=True)
    spec = SPECS["rf"]
    x_all, _names, _med = _tree_matrix(layers, spec, tree_ids)
    train_mask = layers["profile_id"].astype(str).isin(tree_ids).to_numpy()
    model = make_tree(spec, seed=42)
    model.fit(x_all[train_mask], layers.loc[train_mask, "socd_kgm2"].to_numpy(float))
    pred = model.predict(x_all)
    ids = layers["profile_id"].astype(str).to_numpy()

    def layer_pred(wanted: set[str]) -> dict:
        out = {}
        for pid in wanted:
            sel = ids == str(pid)
            if sel.any():
                out[str(pid)] = pred[sel]
        return out

    cal, _ = _stock_predictions(layers, profile_meta, spec, cal_ids, layer_pred(set(map(str, cal_ids))), None, None, None, "round12_refit", fold)
    test, _ = _stock_predictions(layers, profile_meta, spec, test_ids, layer_pred(set(map(str, test_ids))), None, None, None, "round12_refit", fold)
    return cal, test


def _deciles(frame: pd.DataFrame, by: str) -> list[dict]:
    work = frame.dropna(subset=["y100", "p100", "lo100", "hi100"]).copy()
    if work.empty:
        return []
    work["decile"] = pd.qcut(work[by], 10, labels=False, duplicates="drop") + 1
    rows = []
    for decile, group in work.groupby("decile"):
        hit = interval_metrics(group["y100"], group["lo100"], group["hi100"])
        rows.append(
            {
                "grouped_by": by,
                "decile": int(decile),
                "n": int(len(group)),
                "picp": hit["picp"],
                "mpiw": hit["mpiw"],
                "mae": float((group["y100"] - group["p100"]).abs().mean()),
            }
        )
    return rows


def _score_model(test: pd.DataFrame, status100: dict[str, str]) -> dict:
    out = {}
    for depth, y, p, lo, hi in (("30", "y30", "p30", "lo30", "hi30"), ("100", "y100", "p100", "lo100", "hi100")):
        use = test.dropna(subset=[y, p, lo, hi])
        folds = []
        for fold, group in use.groupby("fold"):
            folds.append({"fold": int(fold), **metric_row(group[y], group[p], group[lo], group[hi])})
        detail = pd.DataFrame(folds)
        means = {col: float(detail[col].mean()) for col in ("n", "mae", "r2", "bias", "picp", "mpiw")}
        out[depth] = {"pooled": metric_row(use[y], use[p], use[lo], use[hi]), "fold_mean": means, "folds": folds}
    use100 = test.dropna(subset=["y100", "p100", "lo100", "hi100"]).copy()
    use100["status"] = use100["profile_id"].map(status100)
    by_status = {}
    for status, group in use100.groupby("status"):
        by_status[str(status)] = {"n": int(len(group)), **interval_metrics(group["y100"], group["lo100"], group["hi100"])}
    out["by_status_0_100"] = by_status
    out["predicted_decile"] = _deciles(use100, "p100")
    out["observed_decile"] = _deciles(use100, "y100")
    return out


def _coverage_payload(layers, profiles, classified) -> dict:
    valid = _valid_sets(classified)
    archived = int(profiles["cover_100_200"].sum()) if "cover_100_200" in profiles.columns else None
    part = classified.loc[classified["depth_cm"] == 200]
    return {
        "union_cover_0_200": int(len(valid[200])),
        "archived_cover_100_200": archived,
        "status_counts_200": part["status"].value_counts().to_dict(),
        "note": "A 100–200 cm increment is supported only when the union of recorded layers reaches 200 cm. The archived count used stock_to_depth and can keep a profile that jumps across a gap.",
    }


def run() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, profiles = _load()
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    classified = pd.concat([_classify(layers, 30.0), _classify(layers, 100.0), _classify(layers, 200.0)], ignore_index=True)
    classified["profile_id"] = classified["profile_id"].astype(str)
    coverage = _coverage_payload(layers, profiles, classified)
    (METRICS / "round12_depth200.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print("COVER200", json.dumps(coverage), flush=True)

    valid = _valid_sets(classified)
    status100 = _status_map(classified, 100.0)
    folds = _folds(profile_meta, "block_id")
    archived = pd.read_parquet(METRICS / "oof_stocks.parquet")
    archived["profile_id"] = archived["profile_id"].astype(str)
    archived = archived.loc[archived["split"] == "spatial_block"]

    rows = []
    summaries = {}
    match = {}
    for model_name, predictor in (("mcu", "mcu"), ("rf", "rf")):
        cal_parts, test_parts, fold_quant = [], [], []
        for fold, train_ids, test_ids in folds:
            train_ids = set(map(str, train_ids))
            test_ids = set(map(str, test_ids))
            fit_ids, val_ids, cal_ids, tree_ids = _split_fit_cal_val(profile_meta, np.array(sorted(train_ids)), fold)
            if predictor == "mcu":
                cal, test = _mcu_predict(layers, profile_meta, fit_ids, val_ids, cal_ids, test_ids, fold)
            else:
                cal, test = _rf_predict(layers, profile_meta, tree_ids, cal_ids, test_ids, fold)
            cal["profile_id"] = cal["profile_id"].astype(str)
            test["profile_id"] = test["profile_id"].astype(str)
            cal["role"] = "calibration"
            test["role"] = "test"
            raw_q = {
                "30": _quantiles(cal, "y30", "p30"),
                "100": _quantiles(cal, "y100", "p100"),
            }
            cal_c = _mask_invalid(cal, valid[30], valid[100])
            test_c = _mask_invalid(test, valid[30], valid[100])
            test_out, quant = _attach(cal_c, test_c)
            cal_out, _ = _attach(cal_c, cal_c)
            fold_quant.append({"fold": int(fold), "including_invalid_archived_targets": raw_q, "union_valid_only": quant})
            cal_parts.append(cal_out)
            test_parts.append(test_out)
        test_all = pd.concat(test_parts, ignore_index=True)
        cal_all = pd.concat(cal_parts, ignore_index=True)
        both = pd.concat([cal_all, test_all], ignore_index=True)
        both["model"] = model_name
        rows.append(both)
        summaries[model_name] = {"folds": fold_quant, "scores": _score_model(test_all, status100)}
        arch = archived.loc[archived["model"] == model_name, ["profile_id", "p30", "p100"]].rename(columns={"p30": "p30_arch", "p100": "p100_arch"})
        merged = test_all.merge(arch, on="profile_id", how="inner")
        match[model_name] = {
            "n": int(len(merged)),
            "median_abs_p30": float((merged["p30"] - merged["p30_arch"]).abs().median()),
            "median_abs_p100": float((merged["p100"] - merged["p100_arch"]).abs().median()),
            "max_abs_p100": float((merged["p100"] - merged["p100_arch"]).abs().max()),
        }
        print(model_name, json.dumps(match[model_name]), flush=True)

    print("map calibration refit", flush=True)
    all_ids = np.array(sorted(profile_meta["profile_id"].astype(str)))
    fit_ids, val_ids, cal_ids, _tree = _split_fit_cal_val(profile_meta, all_ids, fold=0)
    map_cal, _map_test = _mcu_predict(layers, profile_meta, fit_ids, val_ids, cal_ids, set(), -1)
    map_cal["profile_id"] = map_cal["profile_id"].astype(str)
    map_raw = {"30": _quantiles(map_cal, "y30", "p30"), "100": _quantiles(map_cal, "y100", "p100")}
    map_valid = _mask_invalid(map_cal, valid[30], valid[100])
    map_new = {"30": _quantiles(map_valid, "y30", "p30"), "100": _quantiles(map_valid, "y100", "p100")}
    archived_map = json.loads((METRICS.parent / "manifests" / "map_calibration.json").read_text(encoding="utf-8"))

    detail = pd.concat(rows, ignore_index=True)
    keep = [c for c in ("model", "role", "fold", "profile_id", "y30", "y100", "p30", "p100", "lo30", "hi30", "lo100", "hi100") if c in detail.columns]
    detail[keep].to_csv(TABLES / "round12_interval_profiles.csv", index=False)
    payload = {
        "label": "same-code refit of calibration residual quantiles on union-valid targets; not a replacement of archived point scores",
        "seeds": SEED_NOTE,
        "weights_of_archived_run": "not saved",
        "raster_not_rebuilt": True,
        "coverage_200": coverage,
        "refit_versus_archived_test_predictions": match,
        "models": summaries,
        "map_calibration_refit": {
            "archived_quantiles": archived_map,
            "refit_including_invalid_targets": map_raw,
            "refit_union_valid_only": map_new,
            "raster_still_uses_archived_widths": True,
        },
    }
    (METRICS / "round12_intervals.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("round12 done", flush=True)


if __name__ == "__main__":
    run()

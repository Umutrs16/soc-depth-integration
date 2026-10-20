"""Inner-block hyperparameter search for the tree baselines.

The search scores validation blocks only. The chosen settings are refit on
the fit and validation blocks and then applied to the untouched test blocks.
CatBoost and pyGAM are recorded if they are not installed. Archived scores
are not replaced.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .experiment import _folds, _load, _profile_table, _split_fit_cal_val, _tree_matrix
from .models import SPECS
from .paths import METRICS, TABLES
from .qc import stock_to_depth

RF_GRID = [
    {"max_depth": 12, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"max_depth": 22, "min_samples_leaf": 3, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    {"max_depth": 22, "min_samples_leaf": 8, "max_features": 0.5},
]
XGB_GRID = [
    {"max_depth": 4, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.05},
    {"max_depth": 8, "learning_rate": 0.05},
    {"max_depth": 6, "learning_rate": 0.1},
]


def _optional_imports() -> dict:
    out = {}
    try:
        import catboost  # noqa: F401

        out["catboost"] = "installed but not used: no search was added beyond RF and XGBoost"
    except ImportError:
        out["catboost"] = "not installed"
    try:
        import pygam  # noqa: F401

        out["pygam"] = "installed but not used"
    except ImportError:
        out["pygam"] = "not installed"
    return out


def _stock_mae(layers: pd.DataFrame, pred: np.ndarray, meta: pd.DataFrame) -> float:
    work = layers.copy()
    work["socd_kgm2"] = pred
    rows = []
    for pid, g in work.groupby("profile_id", sort=False):
        observed = meta.loc[str(pid), "stock_0_100"] if str(pid) in meta.index else np.nan
        if not np.isfinite(observed):
            continue
        rows.append(abs(float(stock_to_depth(g, 100)) - float(observed)))
    return float(np.mean(rows)) if rows else float("inf")


def _fit_predict(kind: str, params: dict, x_train, y_train, x_pred, final: bool):
    if kind == "rf":
        model = RandomForestRegressor(
            n_estimators=300 if final else 160,
            n_jobs=4,
            random_state=42,
            **params,
        )
    else:
        model = XGBRegressor(
            n_estimators=400 if final else 200,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=4,
            **params,
        )
    model.fit(x_train, y_train)
    return model.predict(x_pred)


def run() -> None:
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    meta = profile_meta.set_index("profile_id")
    spec = SPECS["rf"]
    records = []
    chosen = []
    started = time.perf_counter()
    for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
        print(f"round4 trees fold {fold}", flush=True)
        fit_ids, val_ids, _cal, tree_ids = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        x_all, _names, _med = _tree_matrix(layers, spec, fit_ids)
        y = layers["socd_kgm2"].to_numpy(float)
        fit_mask = layers["profile_id"].isin(fit_ids).to_numpy()
        val_mask = layers["profile_id"].isin(val_ids).to_numpy()
        test_mask = layers["profile_id"].isin({str(pid) for pid in test_ids}).to_numpy()
        val_layers = layers.loc[val_mask].copy()
        for kind, grid in (("rf", RF_GRID), ("xgb", XGB_GRID)):
            best_mae = float("inf")
            best = None
            for params in grid:
                pred = _fit_predict(kind, params, x_all[fit_mask], y[fit_mask], x_all[val_mask], final=False)
                mae = _stock_mae(val_layers, pred, meta)
                if mae < best_mae:
                    best_mae = mae
                    best = params
            final_mask = layers["profile_id"].isin(tree_ids).to_numpy()
            pred_test = _fit_predict(kind, best, x_all[final_mask], y[final_mask], x_all[test_mask], final=True)
            test_layers = layers.loc[test_mask].copy()
            test_layers["pred"] = pred_test
            for pid, g in test_layers.groupby("profile_id", sort=False):
                records.append(
                    {
                        "model": f"tuned_{kind}",
                        "fold": fold,
                        "profile_id": str(pid),
                        "y30": meta.loc[str(pid), "stock_0_30"],
                        "y100": meta.loc[str(pid), "stock_0_100"],
                        "p30": stock_to_depth(g, 30, value_col="pred"),
                        "p100": stock_to_depth(g, 100, value_col="pred"),
                        "block_id": meta.loc[str(pid), "block_id"],
                    }
                )
            chosen.append({"fold": int(fold), "model": kind, "params": best, "validation_mae_0_100": best_mae})
            print(kind, fold, best, best_mae, flush=True)
    detail = pd.DataFrame(records)
    detail.to_csv(TABLES / "round4_tree_stocks.csv", index=False)
    summary = []
    for model, g in detail.groupby("model"):
        ok = np.isfinite(g["y100"]) & np.isfinite(g["p100"])
        summary.append(
            {
                "model": model,
                "n_0_100": int(ok.sum()),
                "mae_0_100": float(np.mean(np.abs(g.loc[ok, "y100"] - g.loc[ok, "p100"]))),
                "mae_0_30": float(np.mean(np.abs(g.loc[np.isfinite(g["y30"]) & np.isfinite(g["p30"]), "y30"] - g.loc[np.isfinite(g["y30"]) & np.isfinite(g["p30"]), "p30"]))),
            }
        )
    payload = {"imports": _optional_imports(), "chosen": chosen, "summary": summary, "seconds": time.perf_counter() - started}
    (METRICS / "round4_trees.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    run()

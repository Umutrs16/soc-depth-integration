"""Spatial, source, and Xinjiang experiments.

Outer folds are profile-disjoint. Spatial folds use 2-degree blocks.
A 20% block holdout inside each training split is used only for
conformal residual quantiles and is not used to update parameters.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from .metrics import metric_row, summarize_folds
from .models import SPECS, design_matrix, ensemble_predict_layers, make_tree, predict_stock_batch, train_mcu
from .paths import GEE_DIR, METRICS, PROCESSED
from .qc import stock_to_depth

PRIMARY = "mcu"
BLOCK_DEG = 2.0


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    cov = pd.read_parquet(GEE_DIR / "profile_covariates.parquet")
    for frame in (layers, profiles, cov):
        frame["profile_id"] = frame["profile_id"].astype(str)
    cov = cov.drop_duplicates("profile_id")
    profiles = profiles.drop_duplicates("profile_id")
    layers = layers.merge(cov, on="profile_id", how="inner")
    layers = layers.merge(
        profiles[
            [
                "profile_id",
                "has_overlap",
                "block_id",
                "province_zh",
                "province_en",
                "is_xinjiang",
                "stock_0_30",
                "stock_0_100",
                "cover_0_30",
                "cover_0_100",
                "latitude",
                "longitude",
                "reference",
            ]
        ],
        on="profile_id",
        how="left",
        suffixes=("", "_profile"),
    )
    if "reference_profile" in layers.columns:
        layers["reference"] = layers["reference"].fillna(layers["reference_profile"])
    layers = layers.loc[layers["in_training_pool"]].copy()
    layers = layers.sort_values(["profile_id", "top_cm", "bottom_cm", "sample_id"]).reset_index(drop=True)
    profiles = profiles.loc[profiles["profile_id"].isin(layers["profile_id"])].copy()
    return layers, profiles


def _profile_table(layers: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    base = profiles.loc[profiles["profile_id"].isin(layers["profile_id"]), ["profile_id", "block_id", "reference", "is_xinjiang", "latitude", "longitude", "province_en", "stock_0_30", "stock_0_100", "cover_0_30", "cover_0_100"]].copy()
    cov_cols = [c for c in ("map_mm", "pet_mm", "ai", "mat_c") if c in layers.columns]
    extra = layers.groupby("profile_id", sort=False)[cov_cols].first().reset_index()
    return base.merge(extra, on="profile_id", how="left")


def _ids_for_blocks(profile_meta: pd.DataFrame, blocks) -> np.ndarray:
    return profile_meta.loc[profile_meta["block_id"].isin(blocks), "profile_id"].astype(str).unique()


def _split_fit_cal_val(profile_meta: pd.DataFrame, train_ids: np.ndarray, fold: int):
    train = profile_meta.loc[profile_meta["profile_id"].isin(train_ids)]
    blocks = train["block_id"].unique().copy()
    rng = np.random.default_rng(1000 + fold)
    rng.shuffle(blocks)
    n_cal = max(1, int(round(0.20 * len(blocks))))
    cal_blocks = set(blocks[:n_cal])
    fit_blocks = blocks[n_cal:]
    rng_val = np.random.default_rng(2000 + fold)
    rng_val.shuffle(fit_blocks)
    n_val = max(1, int(round(0.15 * len(fit_blocks))))
    val_blocks = set(fit_blocks[:n_val])
    pure_blocks = set(fit_blocks[n_val:])
    cal_ids = set(_ids_for_blocks(train, cal_blocks))
    val_ids = set(_ids_for_blocks(train, val_blocks))
    fit_ids = set(_ids_for_blocks(train, pure_blocks))
    tree_ids = fit_ids | val_ids
    return fit_ids, val_ids, cal_ids, tree_ids


def _scale(x_fit: np.ndarray, x_all: np.ndarray):
    mu = x_fit.mean(axis=0)
    sd = x_fit.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return (x_all - mu) / sd, mu, sd


def _profile_features(frame: pd.DataFrame, spec, fit_ids: set[str]):
    first = frame.groupby("profile_id", sort=False).first().reset_index()
    dummy = first.copy()
    dummy["top_cm"] = 0.0
    dummy["bottom_cm"] = 0.0
    fit_rows = dummy.loc[dummy["profile_id"].isin(fit_ids)]
    _, medians, names = design_matrix(fit_rows, spec, for_tree=False)
    x_all, _, _ = design_matrix(dummy, spec, medians=medians, for_tree=False)
    fit_x, _, _ = design_matrix(fit_rows, spec, medians=medians, for_tree=False)
    x_scaled, mu, sd = _scale(fit_x, x_all)
    lookup = {str(pid): x_scaled[i] for i, pid in enumerate(dummy["profile_id"].astype(str))}
    return lookup, names, medians, mu, sd


def _tree_matrix(frame: pd.DataFrame, spec, fit_ids: set[str]):
    fit_rows = frame.loc[frame["profile_id"].isin(fit_ids)]
    _, medians, names = design_matrix(fit_rows, spec, for_tree=True)
    x_all, _, _ = design_matrix(frame, spec, medians=medians, for_tree=True)
    return x_all, names, medians


def _depth_bin(mid: float) -> str:
    if mid < 30:
        return "0-30"
    if mid < 100:
        return "30-100"
    return "100-200"


def _conformal_quantiles(y, pred, bins=None) -> dict:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    global_q = np.quantile(y[ok] - pred[ok], [0.05, 0.95]) if ok.sum() >= 20 else np.array([-1.0, 1.0])
    out = {"_global": (float(global_q[0]), float(global_q[1]))}
    if bins is None:
        return out
    bins = np.asarray(bins)
    for key in np.unique(bins[ok]):
        sel = ok & (bins == key)
        if sel.sum() < 30:
            out[str(key)] = out["_global"]
        else:
            q = np.quantile(y[sel] - pred[sel], [0.05, 0.95])
            out[str(key)] = (float(q[0]), float(q[1]))
    return out


def _apply_interval(pred, bins, quantiles: dict):
    lo = np.empty_like(pred, dtype=float)
    hi = np.empty_like(pred, dtype=float)
    for i, (value, binv) in enumerate(zip(pred, bins)):
        q05, q95 = quantiles.get(str(binv), quantiles["_global"])
        lo[i] = value + q05
        hi[i] = value + q95
    return lo, hi


def _nearest_km(test_xy: np.ndarray, train_xy: np.ndarray) -> np.ndarray:
    lat0 = np.deg2rad(np.nanmean(np.concatenate([test_xy[:, 1], train_xy[:, 1]])))
    def project(xy):
        return np.column_stack([xy[:, 0] * np.cos(lat0) * 111.32, xy[:, 1] * 111.32])
    nn = NearestNeighbors(n_neighbors=1, algorithm="kd_tree")
    nn.fit(project(train_xy))
    dist, _ = nn.kneighbors(project(test_xy))
    return dist[:, 0]


def _predict_trees(spec, x_train, y_train, w_train, x_pred):
    model = make_tree(spec, seed=42)
    model.fit(x_train, y_train, sample_weight=w_train)
    return model.predict(x_pred)


def _run_fold(layers, profile_meta, spec, train_ids, test_ids, fold: int, split: str):
    fit_ids, val_ids, cal_ids, tree_ids = _split_fit_cal_val(profile_meta, train_ids, fold)
    test_layers = layers.loc[layers["profile_id"].isin(test_ids)].copy()
    cal_layers = layers.loc[layers["profile_id"].isin(cal_ids)].copy()
    if spec.kind == "tree":
        x_all, _names, _med = _tree_matrix(layers, spec, tree_ids)
        train_mask = layers["profile_id"].isin(tree_ids).to_numpy()
        cal_mask = layers["profile_id"].isin(cal_ids).to_numpy()
        test_mask = layers["profile_id"].isin(test_ids).to_numpy()
        weight = np.ones(train_mask.sum(), dtype=float)
        model = make_tree(spec, seed=42)
        model.fit(x_all[train_mask], layers.loc[train_mask, "socd_kgm2"].to_numpy(float), sample_weight=weight)
        pred_cal = model.predict(x_all[cal_mask])
        pred_test = model.predict(x_all[test_mask])
        std_test = np.zeros_like(pred_test)
        layer_pred = {
            str(pid): pred_test[test_layers["profile_id"].astype(str).to_numpy() == str(pid)]
            for pid in test_layers["profile_id"].unique()
        }
        layer_pred_cal = {
            str(pid): pred_cal[cal_layers["profile_id"].astype(str).to_numpy() == str(pid)]
            for pid in cal_layers["profile_id"].unique()
        }
        models = None
        x_lookup = None
        device = None
        max_layers = None
    else:
        x_lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
        models = []
        max_layers = int(layers.groupby("profile_id").size().max())
        for seed in spec.seeds:
            model, max_layers, device = train_mcu(layers, x_lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
            models.append(model)
        cal_mean, _cal_std = ensemble_predict_layers(models, layers, x_lookup, sorted(cal_ids), max_layers, device)
        test_mean, test_std = ensemble_predict_layers(models, layers, x_lookup, sorted(test_ids), max_layers, device)
        pred_cal = np.concatenate([cal_mean[str(pid)] for pid, _g in cal_layers.groupby("profile_id", sort=False)])
        layer_pred_cal = cal_mean
        pred_test_parts, std_parts, test_order = [], [], []
        for pid, g in test_layers.groupby("profile_id", sort=False):
            pred_test_parts.append(test_mean[str(pid)])
            std_parts.append(test_std[str(pid)])
            test_order.append(g)
        pred_test = np.concatenate(pred_test_parts)
        std_test = np.concatenate(std_parts)
        test_layers = pd.concat(test_order, ignore_index=False)
        cal_order = [g for _pid, g in cal_layers.groupby("profile_id", sort=False)]
        cal_layers = pd.concat(cal_order, ignore_index=False)
        layer_pred = test_mean
    if spec.kind == "tree":
        pass
    bins_cal = cal_layers["top_cm"].add(cal_layers["bottom_cm"]).div(2).map(_depth_bin).to_numpy()
    quant = _conformal_quantiles(cal_layers["socd_kgm2"].to_numpy(float), pred_cal, bins_cal)
    bins_test = test_layers["top_cm"].add(test_layers["bottom_cm"]).div(2).map(_depth_bin).to_numpy()
    lo, hi = _apply_interval(pred_test, bins_test, quant)
    oof = test_layers[["profile_id", "sample_id", "top_cm", "bottom_cm", "socd_kgm2", "province_en", "reference"]].copy()
    oof["model"] = spec.name
    oof["split"] = split
    oof["fold"] = fold
    oof["pred"] = pred_test
    oof["pred_std"] = std_test
    oof["lo"] = lo
    oof["hi"] = hi
    oof["depth_bin"] = bins_test
    rows = []
    rows.append({"model": spec.name, "split": split, "fold": fold, "target": "layer", **metric_row(oof["socd_kgm2"], oof["pred"], oof["lo"], oof["hi"])})
    for bin_name, g in oof.groupby("depth_bin"):
        rows.append(
            {
                "model": spec.name,
                "split": split,
                "fold": fold,
                "target": f"layer_{bin_name}",
                **metric_row(g["socd_kgm2"], g["pred"], g["lo"], g["hi"]),
            }
        )
    stock_test, _unused = _stock_predictions(
        layers, profile_meta, spec, test_ids, layer_pred, models, x_lookup, device, split, fold
    )
    stock_cal, _unused_cal = _stock_predictions(
        layers, profile_meta, spec, cal_ids, layer_pred_cal, models, x_lookup, device, split, fold
    )
    stock_test = _attach_stock_intervals(stock_cal, stock_test)
    return oof, rows, stock_test


def _stock_predictions(layers, profile_meta, spec, test_ids, layer_pred, models, x_lookup, device, split, fold):
    """Build stock predictions. Tree models sum layer predictions; MCU integrates q."""
    records = []
    metric_rows = []
    meta = profile_meta.set_index("profile_id")
    # Conformal for stocks is computed after collecting calibration elsewhere.
    # Here, stock intervals are filled later if quantiles are absent.
    for pid in test_ids:
        key = str(pid)
        if key not in meta.index:
            continue
        g = layers.loc[layers["profile_id"] == key].copy()
        if spec.kind == "tree":
            pred = np.asarray(layer_pred[key], dtype=float)
            if len(pred) != len(g):
                continue
            g = g.copy()
            g["socd_kgm2"] = pred
            p30 = stock_to_depth(g, 30)
            p100 = stock_to_depth(g, 100)
            info = meta.loc[key]
            records.append(
                {
                    "model": spec.name,
                    "split": split,
                    "fold": fold,
                    "profile_id": key,
                    "province_en": info["province_en"],
                    "y30": info["stock_0_30"],
                    "y100": info["stock_0_100"],
                    "p30": p30,
                    "p100": p100,
                }
            )
    if spec.kind != "tree" and models is not None:
        ids = [str(pid) for pid in test_ids if str(pid) in meta.index and str(pid) in x_lookup]
        if ids:
            matrix = np.stack([x_lookup[pid] for pid in ids])
            mean30 = np.mean([predict_stock_batch(model, matrix, 30, device) for model in models], axis=0)
            mean100 = np.mean([predict_stock_batch(model, matrix, 100, device) for model in models], axis=0)
            for pid, p30, p100 in zip(ids, mean30, mean100):
                info = meta.loc[pid]
                records.append(
                    {
                        "model": spec.name,
                        "split": split,
                        "fold": fold,
                        "profile_id": pid,
                        "province_en": info["province_en"],
                        "y30": info["stock_0_30"],
                        "y100": info["stock_0_100"],
                        "p30": float(p30),
                        "p100": float(p100),
                    }
                )
    stocks = pd.DataFrame.from_records(records)
    return stocks, metric_rows


def _attach_stock_intervals(stock_cal: pd.DataFrame, stock_test: pd.DataFrame) -> pd.DataFrame:
    out = stock_test.copy()
    for src, pred, y, lo_name, hi_name in (
        ("p30", "p30", "y30", "lo30", "hi30"),
        ("p100", "p100", "y100", "lo100", "hi100"),
    ):
        cal_ok = stock_cal[[y, pred]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(cal_ok) < 20:
            q05, q95 = -1.0, 1.0
        else:
            q05, q95 = np.quantile(cal_ok[y] - cal_ok[pred], [0.05, 0.95])
        out[lo_name] = out[pred] + q05
        out[hi_name] = out[pred] + q95
    return out


def _metrics_from_stocks(stocks: pd.DataFrame) -> list[dict]:
    rows = []
    if stocks.empty:
        return rows
    for (model, split, fold), g in stocks.groupby(["model", "split", "fold"]):
        for target, y, p, lo, hi in (
            ("stock_0_30", "y30", "p30", "lo30", "hi30"),
            ("stock_0_100", "y100", "p100", "lo100", "hi100"),
        ):
            if lo not in g:
                rows.append({"model": model, "split": split, "fold": fold, "target": target, **metric_row(g[y], g[p])})
            else:
                rows.append({"model": model, "split": split, "fold": fold, "target": target, **metric_row(g[y], g[p], g[lo], g[hi])})
    return rows


def _distance_table(profile_meta, folds) -> pd.DataFrame:
    rows = []
    for fold, train_ids, test_ids in folds:
        test = profile_meta.loc[profile_meta["profile_id"].isin(test_ids), ["longitude", "latitude"]].to_numpy(float)
        train = profile_meta.loc[profile_meta["profile_id"].isin(train_ids), ["longitude", "latitude"]].to_numpy(float)
        dist = _nearest_km(test, train)
        rows.append(
            {
                "fold": fold,
                "n_test": int(len(test)),
                "n_train": int(len(train)),
                "nn_km_p05": float(np.quantile(dist, 0.05)),
                "nn_km_median": float(np.median(dist)),
                "nn_km_p95": float(np.quantile(dist, 0.95)),
            }
        )
    return pd.DataFrame(rows)


def _folds(profile_meta: pd.DataFrame, group_col: str):
    splitter = GroupKFold(n_splits=5)
    x = np.zeros(len(profile_meta))
    groups = profile_meta[group_col].astype(str).to_numpy()
    folds = []
    for fold, (tr, te) in enumerate(splitter.split(x, groups=groups)):
        train_ids = profile_meta.iloc[tr]["profile_id"].astype(str).to_numpy()
        test_ids = profile_meta.iloc[te]["profile_id"].astype(str).to_numpy()
        folds.append((fold, train_ids, test_ids))
    return folds


def _evaluate_split(layers, profile_meta, spec, folds, split: str):
    oof_parts, metric_parts, stock_parts = [], [], []
    for fold, train_ids, test_ids in folds:
        print(f"{spec.name} {split} fold {fold} train={len(train_ids)} test={len(test_ids)}", flush=True)
        oof, rows, stock_frame = _run_fold(layers, profile_meta, spec, set(train_ids), set(test_ids), fold, split)
        oof_parts.append(oof)
        metric_parts.extend(rows)
        stock_parts.append(stock_frame)
    oof = pd.concat(oof_parts, ignore_index=True)
    stocks = pd.concat(stock_parts, ignore_index=True) if stock_parts else pd.DataFrame()
    return oof, metric_parts, stocks


def _olm_stock(row, depth: str) -> float:
    slabs = {
        "30": [("0", 5.0), ("10", 10.0), ("30", 15.0)],
        "100": [("0", 5.0), ("10", 10.0), ("30", 15.0), ("60", 30.0), ("100", 40.0)],
    }
    total = 0.0
    for node, thick in slabs[depth]:
        soc = row.get(f"olm_soc_{node}")
        bd = row.get(f"olm_bd_{node}")
        if pd.isna(soc) or pd.isna(bd):
            return np.nan
        total += 0.01 * float(soc) * float(bd) * thick
    return total


def evaluate_openlandmap(layers, profile_meta) -> dict:
    needed = [c for c in layers.columns if str(c).startswith("olm_")]
    if "olm_soc_0" not in needed:
        return {}
    base_cols = ["profile_id"]
    for col in ("stock_0_30", "stock_0_100"):
        if col in layers.columns:
            base_cols.append(col)
    first = layers.groupby("profile_id", sort=False)[base_cols[1:] + needed].first().reset_index()
    if "stock_0_30" not in first.columns:
        first = first.merge(profile_meta[["profile_id", "stock_0_30", "stock_0_100"]], on="profile_id", how="left")
    first["olm30"] = first.apply(lambda r: _olm_stock(r, "30"), axis=1)
    first["olm100"] = first.apply(lambda r: _olm_stock(r, "100"), axis=1)
    out = {
        "stock_0_30": metric_row(first["stock_0_30"], first["olm30"]),
        "stock_0_100": metric_row(first["stock_0_100"], first["olm100"]),
        "note": "OpenLandMap is an external product comparison and may share source profiles with V10. It is not an independent validation and is not a model input.",
    }
    return out


def _paired(summary_source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sub = summary_source.loc[summary_source["split"] == "spatial_block"]
    for target in ("layer", "stock_0_30", "stock_0_100"):
        part = sub.loc[sub["target"] == target]
        if part.empty or "mcu" not in set(part["model"]) or "rf" not in set(part["model"]):
            continue
        mcu = part.loc[part["model"] == "mcu"].set_index("fold")["mae"]
        for baseline in ("rf", "xgb", "hgb", "mlp_integral"):
            if baseline not in set(part["model"]):
                continue
            base = part.loc[part["model"] == baseline].set_index("fold")["mae"]
            common = mcu.index.intersection(base.index)
            delta = base.loc[common] - mcu.loc[common]
            rows.append(
                {
                    "target": target,
                    "baseline": baseline,
                    "n_folds": int(len(common)),
                    "mae_baseline_mean": float(base.loc[common].mean()),
                    "mae_mcu_mean": float(mcu.loc[common].mean()),
                    "mae_reduction": float(delta.mean()),
                    "folds_mcu_better": int((delta > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def run(models: list[str] | None = None) -> None:
    layers, profiles = _load()
    profile_meta = _profile_table(layers, profiles)
    selected = [SPECS[name] for name in (models or ["rf", "xgb", "hgb", "mlp_integral", "mcu", "mcu_equal_weight", "mcu_closure", "mcu_nors"])]
    spatial = _folds(profile_meta, "block_id")
    distances = _distance_table(profile_meta, spatial)
    distances.to_csv(METRICS / "spatial_nn_distance.csv", index=False)
    all_metrics = []
    oof_keep = []
    stock_keep = []
    for spec in selected:
        oof, rows, stocks = _evaluate_split(layers, profile_meta, spec, spatial, "spatial_block")
        all_metrics.extend(rows)
        # Stock conformal needs calibration predictions. Recompute intervals inside _run_fold
        # by a second pass: predict calibration stocks with the same models.
        # The current helper does not return calibration stocks, so stock PICP is added below
        # only when lo/hi exist. Fill a global residual interval from OOF is invalid.
        # Instead, split each fold's stocks is test-only. Leave stock intervals to the
        # dedicated calibration pass implemented in _stock_interval_pass.
        oof_keep.append(oof)
        stock_keep.append(stocks)
        all_metrics.extend(_metrics_from_stocks(stocks))
    # Source-out and Xinjiang for the discrete and primary models only.
    transfer_names = [name for name in ("rf", "mcu") if name in [s.name for s in selected]]
    source_folds = _folds(profile_meta, "reference")
    xj_test = profile_meta.loc[profile_meta["is_xinjiang"].fillna(False), "profile_id"].astype(str).to_numpy()
    xj_train = profile_meta.loc[~profile_meta["is_xinjiang"].fillna(False), "profile_id"].astype(str).to_numpy()
    for name in transfer_names:
        spec = SPECS[name]
        if name == "mcu":
            spec = replace(spec, seeds=(42,))
        oof, rows, stocks = _evaluate_split(layers, profile_meta, spec, source_folds, "source_out")
        all_metrics.extend(rows)
        all_metrics.extend(_metrics_from_stocks(stocks))
        oof_keep.append(oof)
        stock_keep.append(stocks)
        if len(xj_test) >= 30:
            oof, rows, stock_frame = _run_fold(layers, profile_meta, spec, set(xj_train), set(xj_test), 0, "xinjiang_holdout")
            all_metrics.extend(rows)
            all_metrics.extend(_metrics_from_stocks(stock_frame))
            oof_keep.append(oof)
            stock_keep.append(stock_frame)
    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(METRICS / "cv_fold_metrics.csv", index=False)
    summary = summarize_folds(metrics, ["n", "r2", "rmse", "mae", "bias", "ccc", "picp", "mpiw"])
    summary.to_csv(METRICS / "cv_summary.csv", index=False)
    paired = _paired(metrics)
    paired.to_csv(METRICS / "paired_mae.csv", index=False)
    oof = pd.concat(oof_keep, ignore_index=True)
    oof.to_parquet(METRICS / "oof_layers.parquet", index=False)
    stocks = pd.concat(stock_keep, ignore_index=True)
    stocks.to_parquet(METRICS / "oof_stocks.parquet", index=False)
    olm = evaluate_openlandmap(layers, profile_meta)
    (METRICS / "openlandmap_comparison.json").write_text(json.dumps(olm, indent=2), encoding="utf-8")
    digest = {
        "n_layers": int(len(layers)),
        "n_profiles": int(profile_meta["profile_id"].nunique()),
        "block_deg": BLOCK_DEG,
        "spatial_nn_km": distances.to_dict(orient="records"),
        "paired_mae": paired.to_dict(orient="records"),
        "openlandmap": olm,
    }
    (METRICS / "results_digest.json").write_text(json.dumps(digest, indent=2, default=float), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()

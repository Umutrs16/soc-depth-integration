"""Correct the standard-depth coverage test and rescore saved predictions.

The archived stock_to_depth accepts a profile when a later layer starts at or
below the target, even if a gap larger than 1 cm remains above the target.
This script does not refit any model. Training loss is the layer stock.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .experiment import _folds, _load, _profile_table, _split_fit_cal_val
from .metrics import interval_metrics, metric_row
from .paths import METRICS, TABLES
from .qc import GAP_TOL_CM

TOL = float(GAP_TOL_CM)
N_BOOT = 2000
SEED = 11


def _coverage_end(top: np.ndarray, bot: np.ndarray, depth: float) -> float:
    cursor = 0.0
    order = np.argsort(top, kind="mergesort")
    for a, b in zip(top[order], bot[order]):
        if b <= a:
            return cursor
        if a > cursor + TOL:
            return cursor
        cursor = max(cursor, float(b))
        if cursor >= depth - 1e-6:
            return cursor
    return cursor


def _class_for(g: pd.DataFrame, depth: float) -> dict:
    g = g.sort_values(["top_cm", "bottom_cm"])
    top = g["top_cm"].to_numpy(float)
    bot = g["bottom_cm"].to_numpy(float)
    if g.empty or bool(g["has_overlap"].iloc[0]):
        return {"status": "overlap_or_empty", "coverage_end_cm": np.nan, "crosses": False, "exact_edge": False}
    end = _coverage_end(top, bot, depth)
    covered = bool(end >= depth - 1e-6)
    crosses = bool(np.any((top < depth - 1e-3) & (bot > depth + 1e-3)))
    exact_edge = bool(np.any(np.abs(bot - depth) <= 1e-3) or np.any(np.abs(top - depth) <= 1e-3))
    if not covered:
        status = "invalid_gap" if end + TOL < depth else "invalid_short"
        if np.nanmin(top) > TOL:
            status = "invalid_surface"
    elif crosses:
        status = "prorated"
    else:
        status = "exact_boundary"
    return {"status": status, "coverage_end_cm": float(end), "crosses": crosses, "exact_edge": exact_edge}


def _classify(layers: pd.DataFrame, depth: float) -> pd.DataFrame:
    rows = []
    for pid, g in layers.groupby("profile_id", sort=False):
        row = _class_for(g, depth)
        row["profile_id"] = str(pid)
        row["depth_cm"] = depth
        rows.append(row)
    return pd.DataFrame(rows)


def _boot_mean(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> dict:
    work = pd.DataFrame({"value": values, "block": blocks})
    work = work.loc[np.isfinite(work["value"])]
    groups = {key: grp["value"].to_numpy() for key, grp in work.groupby("block", sort=False)}
    keys = np.array(list(groups))
    observed = float(work["value"].mean())
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(work)), "n_blocks": int(len(keys)), "mean": observed, "ci95_low": float(lo), "ci95_high": float(hi)}


def _paired(frame: pd.DataFrame, ycol: str, left: str, right: str, rng: np.random.Generator) -> dict:
    use = frame[[ycol, left, right, "block_id"]].dropna()
    delta = (use[ycol] - use[right]).abs() - (use[ycol] - use[left]).abs()
    stat = _boot_mean(delta.to_numpy(float), use["block_id"].to_numpy(), rng)
    stat["mae_left"] = float((use[ycol] - use[left]).abs().mean())
    stat["mae_right"] = float((use[ycol] - use[right]).abs().mean())
    return stat


def _fold_summary(frame: pd.DataFrame, ycol: str, pcol: str, lo=None, hi=None) -> dict:
    rows = []
    for fold, g in frame.groupby("fold"):
        if lo and lo in g:
            rows.append({"fold": int(fold), **metric_row(g[ycol], g[pcol], g[lo], g[hi])})
        else:
            rows.append({"fold": int(fold), **metric_row(g[ycol], g[pcol])})
    detail = pd.DataFrame(rows)
    pooled = metric_row(frame[ycol], frame[pcol], frame[lo], frame[hi]) if lo and lo in frame else metric_row(frame[ycol], frame[pcol])
    means = {col: float(detail[col].mean()) for col in ("n", "mae", "r2", "bias", "picp", "mpiw") if col in detail}
    return {"pooled": pooled, "fold_mean": means, "folds": detail.to_dict(orient="records")}


def _cluster(detail: pd.DataFrame, column: str, rng: np.random.Generator) -> dict:
    med = float(np.nanmedian(detail[column]))
    means = detail.groupby("profile_id")[column].mean()
    macro = float(means.mean())
    ids = means.index.to_numpy()
    groups = {pid: part[column].to_numpy(float) for pid, part in detail.groupby("profile_id", sort=False)}
    med_draws, mac_draws = [], []
    for _ in range(1000):
        chosen = rng.choice(ids, size=ids.size, replace=True)
        med_draws.append(float(np.nanmedian(np.concatenate([groups[pid] for pid in chosen]))))
        mac_draws.append(float(np.mean([groups[pid].mean() for pid in chosen])))
    return {
        "n_layers": int(len(detail)),
        "n_profiles": int(ids.size),
        "layer_median": med,
        "layer_median_ci": [float(np.quantile(med_draws, 0.025)), float(np.quantile(med_draws, 0.975))],
        "macro_mean": macro,
        "macro_ci": [float(np.quantile(mac_draws, 0.025)), float(np.quantile(mac_draws, 0.975))],
    }


def main() -> None:
    layers, profiles = _load()
    meta = _profile_table(layers, profiles)
    classified = pd.concat([_classify(layers, 30.0), _classify(layers, 100.0)], ignore_index=True)
    classified = classified.merge(
        meta[["profile_id", "block_id", "reference", "is_xinjiang", "province_en", "stock_0_30", "stock_0_100", "cover_0_30", "cover_0_100"]],
        on="profile_id",
        how="left",
    )
    classified.to_csv(TABLES / "round11_coverage_profiles.csv", index=False)

    summary = {}
    for depth, stock_col, cover_col in ((30.0, "stock_0_30", "cover_0_30"), (100.0, "stock_0_100", "cover_0_100")):
        part = classified.loc[classified["depth_cm"] == depth]
        archived = part.loc[part[cover_col].astype(bool)]
        summary[str(int(depth))] = {
            "archived_n": int(archived.shape[0]),
            "status_counts": part["status"].value_counts().to_dict(),
            "archived_status_counts": archived["status"].value_counts().to_dict(),
            "invalid_in_archived": archived.loc[~archived["status"].isin(["exact_boundary", "prorated"]), ["profile_id", "status", "coverage_end_cm", "province_en", "reference"]].to_dict(orient="records"),
        }

    stocks_all = pd.read_parquet(METRICS / "oof_stocks.parquet")
    stocks_all["profile_id"] = stocks_all["profile_id"].astype(str)
    stocks = stocks_all.loc[stocks_all["split"] == "spatial_block"].copy()
    spatial = stocks.loc[stocks["model"] == "mcu", ["profile_id", "fold"]].drop_duplicates("profile_id")
    valid = {
        30: set(classified.loc[(classified["depth_cm"] == 30) & classified["status"].isin(["exact_boundary", "prorated"]), "profile_id"]),
        100: set(classified.loc[(classified["depth_cm"] == 100) & classified["status"].isin(["exact_boundary", "prorated"]), "profile_id"]),
    }
    classes = {
        depth: {
            status: set(classified.loc[(classified["depth_cm"] == depth) & (classified["status"] == status), "profile_id"])
            for status in ("exact_boundary", "prorated")
        }
        for depth in (30, 100)
    }

    models = ["mcu", "rf", "mlp_integral", "hgb", "xgb"]
    score = {}
    rng = np.random.default_rng(SEED)
    for depth, ycol, pcol, lo, hi in (
        (30, "y30", "p30", "lo30", "hi30"),
        (100, "y100", "p100", "lo100", "hi100"),
    ):
        depth_score = {}
        for model in models:
            one = stocks.loc[stocks["model"] == model, ["profile_id", "fold", ycol, pcol] + [c for c in (lo, hi) if c in stocks.columns]].copy()
            one = one.loc[one["profile_id"].isin(valid[depth])].dropna(subset=[ycol, pcol])
            depth_score[model] = _fold_summary(one, ycol, pcol, lo if lo in one else None, hi if hi in one else None)
        mcu = stocks.loc[(stocks["model"] == "mcu") & stocks["profile_id"].isin(valid[depth]), ["profile_id", "fold", ycol, pcol]].rename(columns={pcol: "p_mcu"})
        mcu = mcu.merge(meta[["profile_id", "block_id"]], on="profile_id", how="left")
        paired = {}
        for model in ("rf", "mlp_integral", "hgb", "xgb"):
            other = stocks.loc[(stocks["model"] == model) & stocks["profile_id"].isin(valid[depth]), ["profile_id", pcol]].rename(columns={pcol: "p_other"})
            merged = mcu.merge(other, on="profile_id")
            paired[model] = _paired(merged, ycol, "p_mcu", "p_other", rng)
        by_class = {}
        for status, ids in classes[depth].items():
            sub_m = mcu.loc[mcu["profile_id"].isin(ids)]
            sub_r = stocks.loc[(stocks["model"] == "rf") & stocks["profile_id"].isin(ids), ["profile_id", pcol]].rename(columns={pcol: "p_rf"})
            merged = sub_m.merge(sub_r, on="profile_id")
            by_class[status] = {
                "primary": metric_row(merged[ycol], merged["p_mcu"]),
                "random_forest": metric_row(merged[ycol], merged["p_rf"]),
                "paired_rf_minus_primary": _paired(merged.rename(columns={"p_rf": "p_other"}), ycol, "p_mcu", "p_other", rng),
                "primary_mae_interval": _boot_mean((merged[ycol] - merged["p_mcu"]).abs().to_numpy(float), merged["block_id"].to_numpy(), rng),
            }
        use = mcu.dropna(subset=[ycol, "p_mcu"])
        order = np.argsort(use["p_mcu"].to_numpy())
        top = use.iloc[order[int(0.9 * len(order)) :]]
        interval = stocks.loc[(stocks["model"] == "mcu") & stocks["profile_id"].isin(top["profile_id"]), [ycol, lo, hi]]
        depth_score["paired_other_minus_primary"] = paired
        depth_score["by_class"] = by_class
        depth_score["top_predicted_decile_picp"] = interval_metrics(interval[ycol], interval[lo], interval[hi])
        score[str(depth)] = depth_score

    folds = _folds(meta, "block_id")
    role = {}
    invalid_ids = set(classified.loc[~classified["status"].isin(["exact_boundary", "prorated", "overlap_or_empty"]), "profile_id"])
    archived_invalid = set()
    for depth, cover_col in ((30.0, "cover_0_30"), (100.0, "cover_0_100")):
        bad = classified.loc[(classified["depth_cm"] == depth) & classified[cover_col].astype(bool) & ~classified["status"].isin(["exact_boundary", "prorated"]), "profile_id"]
        archived_invalid.update(bad.astype(str))
    reproduced = True
    oof_fold = spatial.set_index("profile_id")["fold"]
    for fold, _train, test_ids in folds:
        for pid in test_ids:
            if pid in oof_fold.index and int(oof_fold.loc[pid]) != int(fold):
                reproduced = False
                break
    roles = []
    if reproduced:
        for pid in sorted(archived_invalid):
            for fold, train_ids, test_ids in folds:
                train_ids = set(map(str, train_ids))
                if pid in set(map(str, test_ids)):
                    roles.append({"profile_id": pid, "fold": int(fold), "role": "test"})
                elif pid in train_ids:
                    fit_ids, val_ids, cal_ids, _tree = _split_fit_cal_val(meta, np.array(sorted(train_ids)), fold)
                    if pid in cal_ids:
                        roles.append({"profile_id": pid, "fold": int(fold), "role": "calibration"})
                    elif pid in val_ids:
                        roles.append({"profile_id": pid, "fold": int(fold), "role": "validation"})
                    elif pid in fit_ids:
                        roles.append({"profile_id": pid, "fold": int(fold), "role": "fit"})
        all_ids = np.array(sorted(meta["profile_id"].astype(str)))
        _fit, _val, map_cal, _tree = _split_fit_cal_val(meta, all_ids, fold=0)
        map_hit = sorted(archived_invalid.intersection(map_cal))
    else:
        map_hit = None

    hold = pd.read_csv(TABLES / "round9_holdout_layers.csv")
    hold["profile_id"] = hold["profile_id"].astype(str)
    hold["endpoint"] = (hold["top_cm"] <= 1e-6) | (np.abs(hold["bottom_cm"] - 100) <= 1e-3)
    hold_rng = np.random.default_rng(10)
    hidden = {}
    for name, column in (
        ("selected", "grid_selected_stock_abs"),
        ("lambda_10", "grid_10_stock_abs"),
        ("lambda_1e-4", "grid_1e-4_stock_abs"),
        ("fourier", "fourier_stock_abs"),
    ):
        hidden[name] = {
            "endpoint": _cluster(hold.loc[hold["endpoint"]], column, hold_rng),
            "interior": _cluster(hold.loc[~hold["endpoint"]], column, hold_rng),
        }

    g = layers.loc[layers["profile_id"] == "3637"].sort_values(["top_cm", "bottom_cm"])
    soc = layers["soc_gkg"].to_numpy(float)
    soc_cap = float(np.nanquantile(soc[np.isfinite(soc)], 0.99))
    layer_rows = g[["top_cm", "bottom_cm", "soc_gkg", "bd_gcm3", "socd_kgm2", "reference", "method"]].copy()
    layer_rows["thickness_cm"] = layer_rows["bottom_cm"] - layer_rows["top_cm"]
    layer_rows["product_0p01_soc_bd_thick"] = 0.01 * layer_rows["soc_gkg"] * layer_rows["bd_gcm3"] * layer_rows["thickness_cm"]
    y100 = float(meta.loc[meta["profile_id"] == "3637", "stock_0_100"].iloc[0])
    cross = layer_rows.loc[(layer_rows["top_cm"] < 100) & (layer_rows["bottom_cm"] > 100)].iloc[0]
    frac = (100 - float(cross["top_cm"])) / float(cross["thickness_cm"])
    capped_layer = float(cross["socd_kgm2"]) * min(1.0, soc_cap / float(cross["soc_gkg"]))
    capped_piece = capped_layer * frac
    capped_y = y100 - float(cross["socd_kgm2"]) * frac + capped_piece
    xj_all = stocks_all.loc[stocks_all["split"] == "xinjiang_holdout"].copy()
    xj_m = xj_all.loc[xj_all["model"] == "mcu", ["profile_id", "y100", "p100"]]
    xj_r = xj_all.loc[xj_all["model"] == "rf", ["profile_id", "p100"]].rename(columns={"p100": "p100_rf"})
    xj = xj_m.merge(xj_r, on="profile_id").dropna(subset=["y100", "p100", "p100_rf"])
    xj = xj.loc[xj["profile_id"].isin(valid[100])]
    spatial_csv = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    spatial_csv["profile_id"] = spatial_csv["profile_id"].astype(str)
    row = spatial_csv.loc[spatial_csv["profile_id"] == "3637"].iloc[0]

    def _mae(frame, y, p):
        return float((frame[y] - frame[p]).abs().mean()), int(len(frame))

    xj_in = _mae(xj, "y100", "p100"), _mae(xj, "y100", "p100_rf")
    xj_out = xj.loc[xj["profile_id"] != "3637"]
    xj_drop = _mae(xj_out, "y100", "p100"), _mae(xj_out, "y100", "p100_rf")
    xj_cap = xj.copy()
    xj_cap.loc[xj_cap["profile_id"] == "3637", "y100"] = capped_y
    xj_capped = _mae(xj_cap, "y100", "p100"), _mae(xj_cap, "y100", "p100_rf")
    spatial_pred = spatial_csv.dropna(subset=["y100", "p100", "p100_rf"])
    spatial_pred = spatial_pred.loc[spatial_pred["profile_id"].isin(valid[100])]
    spatial_cap = spatial_pred.copy()
    spatial_cap.loc[spatial_cap["profile_id"] == "3637", "y100"] = capped_y

    valid100 = classified.loc[(classified["depth_cm"] == 100) & classified["status"].isin(["exact_boundary", "prorated"])]
    liu = valid100["reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
    valid30 = classified.loc[(classified["depth_cm"] == 30) & classified["status"].isin(["exact_boundary", "prorated"])]
    xj_profiles = meta.loc[meta["is_xinjiang"].fillna(False), "profile_id"]

    payload = {
        "tolerance_cm": TOL,
        "training_loss": "Huber on layer stock socd_kgm2. Standard-depth labels are evaluation targets and stock-interval residuals, not the training loss.",
        "weights_not_retrained": True,
        "bootstrap_seed": SEED,
        "bootstrap_draws": N_BOOT,
        "fold_assignment_reproduced": reproduced,
        "coverage": summary,
        "valid_n": {"0_30": int(len(valid30)), "0_100": int(len(valid100))},
        "liu_complete_0_100": {"n": int(liu.sum()), "denominator": int(len(valid100)), "share": float(liu.mean())},
        "xinjiang_valid": {
            "profiles": int(xj_profiles.nunique()),
            "valid_0_30": int(valid30["profile_id"].isin(set(xj_profiles)).sum()),
            "valid_0_100": int(valid100["profile_id"].isin(set(xj_profiles)).sum()),
        },
        "scores": score,
        "invalid_roles": roles,
        "invalid_in_map_calibration": map_hit,
        "hidden_endpoint": hidden,
        "profile_3637": {
            "soc_cap_g_kg_p99": soc_cap,
            "y100_archived": y100,
            "y100_soc_capped_at_p99": capped_y,
            "crossing_layer": cross.to_dict(),
            "layers": layer_rows.to_dict(orient="records"),
            "prediction_p100": float(row["p100"]),
            "prediction_p100_rf": float(row["p100_rf"]),
            "xinjiang_include_mae_primary_rf": xj_in,
            "xinjiang_exclude_mae_primary_rf": xj_drop,
            "xinjiang_capped_target_mae_primary_rf": xj_capped,
            "spatial_capped_target_mae_primary_rf": (_mae(spatial_cap, "y100", "p100"), _mae(spatial_cap, "y100", "p100_rf")),
            "robust_loss_on_standard_depth_target": "not run. The archived primary loss is already Huber on the layer stock.",
            "source_note": "V10 reference string and the Science Bulletin grid paper do not publish this layer table. No gravel fraction is in the file.",
        },
    }
    (METRICS / "round11_coverage.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"valid": payload["valid_n"], "liu": payload["liu_complete_0_100"], "invalid": summary, "map_cal": map_hit, "reproduced": reproduced}, indent=2, default=float))


if __name__ == "__main__":
    main()

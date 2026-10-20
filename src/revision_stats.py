"""Extra statistics requested by the review, computed only from saved predictions.

The block bootstrap resamples the existing out-of-fold errors. It does not refit
the models. Buffered cross-validation, soil-thickness scenarios, and a spatially
correlated national carbon interval were not run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import METRICS, PROCESSED, TABLES

RNG = np.random.default_rng(42)


def _block_bootstrap(frame: pd.DataFrame, value: str, block: str, n_boot: int = 2000) -> dict:
    work = frame.loc[np.isfinite(frame[value])].copy()
    groups = {key: grp[value].to_numpy() for key, grp in work.groupby(block, sort=False)}
    keys = np.array(list(groups))
    if len(keys) < 2:
        return {"n_blocks": int(len(keys)), "n": int(len(work)), "mean": None}
    observed = float(work[value].mean())
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = RNG.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(len(work)),
        "n_blocks": int(len(keys)),
        "mean_baseline_minus_mcu": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "fraction_boot_positive": float(np.mean(draws > 0)),
    }


def _paired(layers: pd.DataFrame, stocks: pd.DataFrame, profiles: pd.DataFrame) -> list[dict]:
    blocks = profiles[["profile_id", "block_id"]].copy()
    blocks["profile_id"] = blocks["profile_id"].astype(str)
    rows = []
    spatial_layers = layers.loc[layers["split"] == "spatial_block"].copy()
    spatial_layers["profile_id"] = spatial_layers["profile_id"].astype(str)
    mcu_l = spatial_layers.loc[spatial_layers["model"] == "mcu", ["profile_id", "sample_id", "socd_kgm2", "pred"]]
    for baseline in ["rf", "xgb", "hgb", "mlp_integral", "mcu_equal_weight", "mcu_closure", "mcu_nors"]:
        other = spatial_layers.loc[spatial_layers["model"] == baseline, ["sample_id", "pred"]]
        merged = mcu_l.merge(other, on="sample_id", suffixes=("_mcu", "_base"))
        merged = merged.merge(blocks, on="profile_id", how="left")
        merged["delta"] = (merged["socd_kgm2"] - merged["pred_base"]).abs() - (merged["socd_kgm2"] - merged["pred_mcu"]).abs()
        stat = _block_bootstrap(merged, "delta", "block_id")
        stat.update({"target": "layer", "baseline": baseline})
        rows.append(stat)
    spatial_stocks = stocks.loc[stocks["split"] == "spatial_block"].copy()
    spatial_stocks["profile_id"] = spatial_stocks["profile_id"].astype(str)
    for target, ycol, pcol in (("stock_0_30", "y30", "p30"), ("stock_0_100", "y100", "p100")):
        mcu_s = spatial_stocks.loc[spatial_stocks["model"] == "mcu", ["profile_id", ycol, pcol]].dropna()
        for baseline in ["rf", "xgb", "hgb", "mlp_integral", "mcu_equal_weight", "mcu_closure", "mcu_nors"]:
            other = spatial_stocks.loc[spatial_stocks["model"] == baseline, ["profile_id", pcol]].dropna()
            merged = mcu_s.merge(other, on="profile_id", suffixes=("_mcu", "_base"))
            merged = merged.merge(blocks, on="profile_id", how="left")
            merged["delta"] = (merged[ycol] - merged[f"{pcol}_base"]).abs() - (merged[ycol] - merged[f"{pcol}_mcu"]).abs()
            stat = _block_bootstrap(merged, "delta", "block_id")
            stat.update({"target": target, "baseline": baseline})
            rows.append(stat)
    return rows


def _winkler(y, lo, hi, alpha: float = 0.1) -> float:
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return float(np.mean(width + (2.0 / alpha) * below + (2.0 / alpha) * above))


def _interval_scores(layers: pd.DataFrame, stocks: pd.DataFrame) -> list[dict]:
    rows = []
    for split in ["spatial_block", "source_out", "xinjiang_holdout"]:
        part = layers.loc[(layers["model"] == "mcu") & (layers["split"] == split)]
        finite = part.loc[np.isfinite(part["lo"]) & np.isfinite(part["hi"]) & np.isfinite(part["socd_kgm2"])]
        if len(finite) == 0:
            continue
        cover = (finite["socd_kgm2"] >= finite["lo"]) & (finite["socd_kgm2"] <= finite["hi"])
        rows.append(
            {
                "target": "layer",
                "split": split,
                "n": int(len(finite)),
                "picp": float(cover.mean()),
                "mpiw": float((finite["hi"] - finite["lo"]).mean()),
                "winkler90": _winkler(finite["socd_kgm2"].to_numpy(), finite["lo"].to_numpy(), finite["hi"].to_numpy()),
            }
        )
        for depth, sub in finite.groupby("depth_bin"):
            cover_d = (sub["socd_kgm2"] >= sub["lo"]) & (sub["socd_kgm2"] <= sub["hi"])
            rows.append(
                {
                    "target": f"layer_{depth}",
                    "split": split,
                    "n": int(len(sub)),
                    "picp": float(cover_d.mean()),
                    "mpiw": float((sub["hi"] - sub["lo"]).mean()),
                    "winkler90": _winkler(sub["socd_kgm2"].to_numpy(), sub["lo"].to_numpy(), sub["hi"].to_numpy()),
                }
            )
    for split in ["spatial_block", "source_out", "xinjiang_holdout"]:
        part = stocks.loc[(stocks["model"] == "mcu") & (stocks["split"] == split)]
        for target, ycol, locol, hicol in (("stock_0_30", "y30", "lo30", "hi30"), ("stock_0_100", "y100", "lo100", "hi100")):
            finite = part.loc[np.isfinite(part[ycol]) & np.isfinite(part[locol]) & np.isfinite(part[hicol])]
            if len(finite) == 0:
                continue
            cover = (finite[ycol] >= finite[locol]) & (finite[ycol] <= finite[hicol])
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "n": int(len(finite)),
                    "picp": float(cover.mean()),
                    "mpiw": float((finite[hicol] - finite[locol]).mean()),
                    "winkler90": _winkler(finite[ycol].to_numpy(), finite[locol].to_numpy(), finite[hicol].to_numpy()),
                }
            )
    return rows


def _source_summary(stocks: pd.DataFrame, profiles: pd.DataFrame) -> list[dict]:
    ref = profiles[["profile_id", "reference"]].copy()
    ref["profile_id"] = ref["profile_id"].astype(str)
    rows = []
    for model in ["mcu", "rf"]:
        part = stocks.loc[(stocks["model"] == model) & (stocks["split"] == "source_out")].copy()
        part["profile_id"] = part["profile_id"].astype(str)
        part = part.merge(ref, on="profile_id", how="left")
        for target, ycol, pcol in (("stock_0_30", "y30", "p30"), ("stock_0_100", "y100", "p100")):
            finite = part.loc[np.isfinite(part[ycol]) & np.isfinite(part[pcol])].copy()
            finite["ae"] = (finite[ycol] - finite[pcol]).abs()
            finite["bias"] = finite[pcol] - finite[ycol]
            fold_mae = finite.groupby("fold")["ae"].mean()
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "n": int(len(finite)),
                    "micro_mae": float(finite["ae"].mean()) if len(finite) else None,
                    "micro_bias": float(finite["bias"].mean()) if len(finite) else None,
                    "macro_mae_nonempty_folds": float(fold_mae.mean()) if len(fold_mae) else None,
                    "n_nonempty_folds": int(fold_mae.shape[0]),
                    "fold_counts": {str(k): int(v) for k, v in finite.groupby("fold").size().items()},
                }
            )
            for name, grp in finite.groupby("reference"):
                if len(grp) < 30 and name not in ("Liu et al., 2022", "Xu et al., 2019"):
                    continue
                if len(grp) < 15:
                    continue
                rows.append(
                    {
                        "model": model,
                        "target": target,
                        "reference": name,
                        "n": int(len(grp)),
                        "micro_mae": float(grp["ae"].mean()),
                        "micro_bias": float(grp["bias"].mean()),
                    }
                )
    return rows


def _xinjiang_tail(stocks: pd.DataFrame, layers: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
    part = stocks.loc[(stocks["model"] == "mcu") & (stocks["split"] == "xinjiang_holdout")].copy()
    finite = part.loc[np.isfinite(part["y100"]) & np.isfinite(part["p100"])].copy()
    view = float(np.nanpercentile(finite[["y100", "p100"]].to_numpy(), 97))
    finite["beyond_display"] = (finite["y100"] > view) | (finite["p100"] > view)
    rf = stocks.loc[(stocks["model"] == "rf") & (stocks["split"] == "xinjiang_holdout"), ["profile_id", "p100", "p30"]]
    finite = finite.merge(rf, on="profile_id", how="left", suffixes=("", "_rf"))
    ids = finite.loc[finite["beyond_display"], "profile_id"].astype(str)
    layer_part = layers.loc[layers["profile_id"].astype(str).isin(ids)].copy()
    layer_part["profile_id"] = layer_part["profile_id"].astype(str)
    profiles = profiles.copy()
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    meta = profiles.set_index("profile_id")
    audit_rows = []
    for pid, grp in layer_part.groupby("profile_id"):
        stock_row = finite.loc[finite["profile_id"].astype(str) == pid].iloc[0]
        audit_rows.append(
            {
                "profile_id": pid,
                "reference": grp["reference"].iloc[0],
                "year": int(grp["year"].iloc[0]) if np.isfinite(grp["year"].iloc[0]) else None,
                "latitude": float(grp["Latitude"].iloc[0]),
                "longitude": float(grp["Longitude"].iloc[0]),
                "n_layers": int(len(grp)),
                "depth_max_cm": float(grp["bottom_cm"].max()),
                "methods": ",".join(sorted(grp["method"].dropna().unique())),
                "reliability": ",".join(sorted(grp["reliability"].dropna().unique())),
                "soc_min": float(grp["soc_gkg"].min()) if grp["soc_gkg"].notna().any() else None,
                "soc_max": float(grp["soc_gkg"].max()) if grp["soc_gkg"].notna().any() else None,
                "bd_min": float(grp["bd_gcm3"].min()) if grp["bd_gcm3"].notna().any() else None,
                "bd_max": float(grp["bd_gcm3"].max()) if grp["bd_gcm3"].notna().any() else None,
                "max_layer_socd": float(grp["socd_kgm2"].max()),
                "max_layer_thickness_cm": float(grp.loc[grp["socd_kgm2"].idxmax(), "thickness_cm"]),
                "y100": float(stock_row["y100"]),
                "mcu_p100": float(stock_row["p100"]),
                "rf_p100": float(stock_row["p100_rf"]) if np.isfinite(stock_row["p100_rf"]) else None,
                "clcd": grp["CLCD"].iloc[0],
                "overlap": bool(grp["has_overlap"].iloc[0]) if "has_overlap" in grp else None,
            }
        )
    audit = pd.DataFrame(audit_rows).sort_values("y100", ascending=False)
    def _mae(frame, pred):
        return float((frame["y100"] - frame[pred]).abs().mean())
    sensitivity = {
        "display_cap_kg_m2": view,
        "n_beyond_display": int(finite["beyond_display"].sum()),
        "n_all": int(len(finite)),
        "mcu_mae_all": _mae(finite, "p100"),
        "rf_mae_all": _mae(finite, "p100_rf"),
        "mcu_mae_without_beyond": _mae(finite.loc[~finite["beyond_display"]], "p100"),
        "rf_mae_without_beyond": _mae(finite.loc[~finite["beyond_display"]], "p100_rf"),
        "mcu_bias_all": float((finite["p100"] - finite["y100"]).mean()),
        "mcu_bias_without_beyond": float((finite.loc[~finite["beyond_display"], "p100"] - finite.loc[~finite["beyond_display"], "y100"]).mean()),
        "note": "Removing the tail is a sensitivity check. Those profiles are not treated as false and are retained in the primary score.",
    }
    _ = meta
    return audit, sensitivity


def _key_duplicates(layers: pd.DataFrame) -> dict:
    keys = list(zip(layers["Latitude"].round(4), layers["Longitude"].round(4), layers["top_cm"].round(1), layers["bottom_cm"].round(1)))
    n = len(keys)
    n_unique = len(set(keys))
    return {
        "v10_rows": n,
        "unique_keys_latlon4_depth1": n_unique,
        "rows_sharing_a_key": n - n_unique,
        "key_definition": "latitude and longitude rounded to 4 decimal degrees, top and bottom rounded to 0.1 cm; a set, so repeated keys count once",
    }


def main() -> None:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    oof_layers = pd.read_parquet(METRICS / "oof_layers.parquet")
    oof_stocks = pd.read_parquet(METRICS / "oof_stocks.parquet")
    audit, sensitivity = _xinjiang_tail(oof_stocks, layers, profiles)
    audit.to_csv(TABLES / "xinjiang_tail_audit.csv", index=False)
    payload = {
        "paired_block_bootstrap": _paired(oof_layers, oof_stocks, profiles),
        "interval_scores": _interval_scores(oof_layers, oof_stocks),
        "source_out": _source_summary(oof_stocks, profiles),
        "xinjiang_tail_sensitivity": sensitivity,
        "v10_key_rule": _key_duplicates(layers),
        "not_run": [
            "50/100/200 km buffered spatial cross-validation",
            "refitted hyperparameter search",
            "soil-thickness and coarse-fragment scenarios",
            "spatially correlated simulation of the national petagram interval",
            "conditional conformal or CQR refits",
        ],
    }
    (METRICS / "revision_stats.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote revision stats", len(audit), "tail profiles")


if __name__ == "__main__":
    main()

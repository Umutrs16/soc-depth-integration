"""Analyses that use saved predictions only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, TABLES


def _boot(values, blocks, n_boot=2000, seed=42):
    frame = pd.DataFrame({"v": values, "b": blocks})
    frame = frame.loc[np.isfinite(frame["v"])]
    groups = {k: g["v"].to_numpy() for k, g in frame.groupby("b")}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    observed = float(frame["v"].mean())
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(frame)), "mean": observed, "ci95_low": float(lo), "ci95_high": float(hi)}


def main() -> None:
    stocks = pd.read_csv(TABLES / "round4_stocks.csv")
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    base = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    base["profile_id"] = base["profile_id"].astype(str)
    add = pd.read_csv(TABLES / "integral_additivity.csv")
    add["profile_id"] = add["profile_id"].astype(str)
    base = base.merge(add[["profile_id", "fold", "s30_8", "s30_100_8", "s100_8"]], on=["profile_id", "fold"], how="left")
    out = {"stock_agreement": {}, "by_source": {}, "predicted_decile_picp": []}
    low = stocks.loc[stocks["model"] == "lowfreq"].merge(base, on=["profile_id", "fold"], suffixes=("_low", "_pri"))
    cum = stocks.loc[stocks["model"] == "cumulative", ["profile_id", "fold", "p30", "p100", "gap"]].merge(
        low, on=["profile_id", "fold"], suffixes=("_cum", "")
    )
    # after the second merge, cumulative columns may be p30_cum if both had p30
    def paired(frame, a, b):
        ok = np.isfinite(frame[a]) & np.isfinite(frame[b])
        delta = (frame.loc[ok, a] - frame.loc[ok, b]).abs()
        return {
            "n": int(ok.sum()),
            "median_abs": float(delta.median()),
            "p95_abs": float(delta.quantile(0.95)),
            "corr": float(np.corrcoef(frame.loc[ok, a], frame.loc[ok, b])[0, 1]),
        }

    # column names after merges
    out["columns_low"] = list(low.columns)
    out["columns_cum"] = list(cum.columns)
    (METRICS / "round5_existing_columns.json").write_text(json.dumps({"low": list(low.columns), "cum": list(cum.columns)}, indent=2), encoding="utf-8")

    work = base.loc[np.isfinite(base["y100"]) & np.isfinite(base["p100"]) & np.isfinite(base["lo100"])].copy()
    work["hit"] = (work["y100"] >= work["lo100"]) & (work["y100"] <= work["hi100"])
    work["decile"] = pd.qcut(work["p100"], 10, labels=False, duplicates="drop")
    rows = []
    for decile, g in work.groupby("decile"):
        rows.append({"predicted_decile": int(decile) + 1, "n": int(len(g)), "picp": float(g["hit"].mean()), "mean_width": float((g["hi100"] - g["lo100"]).mean()), "mae": float(np.mean(np.abs(g["y100"] - g["p100"])))})
    out["predicted_decile_picp"] = rows
    obs = base.loc[np.isfinite(base["y100"])].copy()
    obs["liu"] = obs["reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
    for label, g in (("liu_2022", obs.loc[obs["liu"]]), ("other_sources", obs.loc[~obs["liu"]])):
        out["by_source"][label] = {
            "n_0_100": int(len(g)),
            "mae_primary": float(np.mean(np.abs(g["y100"] - g["p100"]))),
            "mae_rf": float(np.mean(np.abs(g["y100"] - g["p100_rf"]))),
        }
    obs30 = base.loc[np.isfinite(base["y30"])].copy()
    obs30["liu"] = obs30["reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
    for label, g in (("liu_2022", obs30.loc[obs30["liu"]]), ("other_sources", obs30.loc[~obs30["liu"]])):
        out["by_source"][label + "_0_30"] = {
            "n": int(len(g)),
            "mae_primary": float(np.mean(np.abs(g["y30"] - g["p30"]))),
            "mae_rf": float(np.mean(np.abs(g["y30"] - g["p30_rf"]))),
        }
    (METRICS / "round5_existing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["by_source"], indent=2))
    print(json.dumps(out["predicted_decile_picp"], indent=2))


if __name__ == "__main__":
    main()

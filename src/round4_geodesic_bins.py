"""Bin the exact WGS84 nearest-neighbour distances and bootstrap the paired gap."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, TABLES


def _boot(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict:
    groups = {key: grp["delta"].to_numpy() for key, grp in frame.groupby("block_id", sort=False)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    observed = float(frame["delta"].mean())
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(frame)), "mean": observed, "ci95_low": float(lo), "ci95_high": float(hi), "n_blocks": int(len(keys))}


def main() -> None:
    detail = pd.read_csv(TABLES / "distance_exact_geodesic.csv")
    old = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    old["profile_id"] = old["profile_id"].astype(str)
    detail["profile_id"] = detail["profile_id"].astype(str)
    bins = [0, 25, 50, 100, np.inf]
    labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    work = detail.loc[np.isfinite(detail["y100"])].copy()
    work["bin_exact"] = pd.cut(work["geodesic_exact_km"], bins, labels=labels, right=False).astype(str)
    work["bin_equi"] = pd.cut(work["nn_km"], bins, labels=labels, right=False).astype(str)
    changed = int((work["bin_exact"] != work["bin_equi"]).sum())
    rows = []
    for label in labels:
        part = work.loc[work["bin_exact"] == label].copy()
        part["delta"] = (part["y100"] - part["p100_rf"]).abs() - (part["y100"] - part["p100"]).abs()
        stat = _boot(part)
        stat.update(
            {
                "bin": label,
                "mae_primary": float(np.mean(np.abs(part["y100"] - part["p100"]))),
                "mae_rf": float(np.mean(np.abs(part["y100"] - part["p100_rf"]))),
            }
        )
        rows.append(stat)
    all_profiles = detail.copy()
    all_profiles["bin_exact"] = pd.cut(all_profiles["geodesic_exact_km"], bins, labels=labels, right=False)
    all_profiles["bin_equi"] = pd.cut(all_profiles["nn_km"], bins, labels=labels, right=False)
    payload = {
        "profiles": int(len(detail)),
        "complete_0_100": int(len(work)),
        "complete_profiles_changing_bin_versus_equirectangular": changed,
        "all_profiles_changing_bin_versus_equirectangular": int((all_profiles["bin_exact"] != all_profiles["bin_equi"]).sum()),
        "median_abs_km_versus_equirectangular": float(np.median(np.abs(detail["geodesic_exact_km"] - detail["nn_km"]))),
        "p95_abs_km_versus_equirectangular": float(np.quantile(np.abs(detail["geodesic_exact_km"] - detail["nn_km"]), 0.95)),
        "max_abs_km_versus_equirectangular": float(np.max(np.abs(detail["geodesic_exact_km"] - detail["nn_km"]))),
        "bins": rows,
    }
    (METRICS / "round4_geodesic_bins.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

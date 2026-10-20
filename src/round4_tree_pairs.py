"""Paired comparison of tuned trees against the archived primary scores."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, TABLES


def _boot(frame: pd.DataFrame, value: str, block: str, n_boot: int = 2000, seed: int = 42) -> dict:
    work = frame.loc[np.isfinite(frame[value])].copy()
    groups = {key: grp[value].to_numpy() for key, grp in work.groupby(block, sort=False)}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    observed = float(work[value].mean())
    draws = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[k] for k in chosen])))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(len(work)),
        "n_blocks": int(len(keys)),
        "mean": observed,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
    }


def main() -> None:
    trees = pd.read_csv(TABLES / "round4_tree_stocks.csv")
    trees["profile_id"] = trees["profile_id"].astype(str)
    base = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    base["profile_id"] = base["profile_id"].astype(str)
    rows = []
    for model, g in trees.groupby("model"):
        merged = g.merge(
            base[["profile_id", "fold", "y100", "p100", "p100_rf", "block_id"]],
            on=["profile_id", "fold"],
            how="inner",
            suffixes=("_tuned", "_primary"),
        )
        ok = np.isfinite(merged["y100_primary"]) & np.isfinite(merged["p100_tuned"]) & np.isfinite(merged["p100_primary"])
        part = merged.loc[ok].copy()
        part["delta_vs_primary"] = (part["y100_primary"] - part["p100_primary"]).abs() - (part["y100_primary"] - part["p100_tuned"]).abs()
        part["delta_vs_archived_rf"] = (part["y100_primary"] - part["p100_rf"]).abs() - (part["y100_primary"] - part["p100_tuned"]).abs()
        fold_mae = part.groupby("fold").apply(lambda d: np.mean(np.abs(d["y100_primary"] - d["p100_tuned"])), include_groups=False)
        stat = _boot(part, "delta_vs_primary", "block_id_primary")
        stat_rf = _boot(part, "delta_vs_archived_rf", "block_id_primary")
        rows.append(
            {
                "model": model,
                "pooled_mae": float(np.mean(np.abs(part["y100_primary"] - part["p100_tuned"]))),
                "fold_mae": {str(k): float(v) for k, v in fold_mae.items()},
                "fold_mean_mae": float(fold_mae.mean()),
                "versus_primary_positive_means_tuned_worse": stat,
                "versus_archived_rf_positive_means_tuned_better": stat_rf,
            }
        )
    (METRICS / "round4_tree_pairs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

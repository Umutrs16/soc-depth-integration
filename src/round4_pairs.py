"""Block bootstrap of fourth-round models against the archived primary scores."""

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
    return {"n": int(len(frame)), "n_blocks": int(len(keys)), "mean": observed, "ci95_low": float(lo), "ci95_high": float(hi)}


def main() -> None:
    stocks = pd.read_csv(TABLES / "round4_stocks.csv")
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    base = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    base["profile_id"] = base["profile_id"].astype(str)
    rows = []
    for model, g in stocks.groupby("model"):
        merged = g.merge(base[["profile_id", "fold", "y100", "p100", "block_id"]], on=["profile_id", "fold"], how="inner", suffixes=("_new", "_primary"))
        ok = np.isfinite(merged["y100_primary"]) & np.isfinite(merged["p100_new"]) & np.isfinite(merged["p100_primary"])
        part = merged.loc[ok].copy()
        part["delta"] = (part["y100_primary"] - part["p100_primary"]).abs() - (part["y100_primary"] - part["p100_new"]).abs()
        part = part.rename(columns={"block_id_primary": "block_id"})
        stat = _boot(part)
        stat["model"] = model
        stat["pooled_mae"] = float(np.mean(np.abs(part["y100_primary"] - part["p100_new"])))
        rows.append(stat)
    (METRICS / "round4_pairs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

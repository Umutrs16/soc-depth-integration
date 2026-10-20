"""Replay the seed-11 block bootstrap from the archived tables.

The generator is advanced in the same order as the coverage rescoring:
0–30 cm first, then 0–100 cm; within each depth the paired contrasts are
random forest, integral MLP, histogram boosting and XGBoost, followed by
the exact-boundary and prorated subsets. It does not refit a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
N_BOOT = 2000
SEED = 11


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


def _paired(frame: pd.DataFrame, ycol: str, rng: np.random.Generator) -> dict:
    use = frame[[ycol, "p_mcu", "p_other", "block_id"]].dropna()
    delta = (use[ycol] - use["p_other"]).abs() - (use[ycol] - use["p_mcu"]).abs()
    stat = _boot_mean(delta.to_numpy(float), use["block_id"].to_numpy(), rng)
    stat["mae_primary"] = float((use[ycol] - use["p_mcu"]).abs().mean())
    stat["mae_other"] = float((use[ycol] - use["p_other"]).abs().mean())
    return stat


def replay() -> dict:
    stocks = pd.read_parquet(ROOT / "outputs" / "metrics" / "oof_stocks.parquet")
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    stocks = stocks.loc[stocks["split"] == "spatial_block"].copy()
    cover = pd.read_csv(ROOT / "outputs" / "tables" / "round11_coverage_profiles.csv")
    cover["profile_id"] = cover["profile_id"].astype(str)
    blocks = cover.drop_duplicates("profile_id").set_index("profile_id")["block_id"]
    rng = np.random.default_rng(SEED)
    out = {}
    for depth, ycol, pcol in ((30, "y30", "p30"), (100, "y100", "p100")):
        part = cover.loc[cover["depth_cm"] == depth]
        valid = set(part.loc[part["status"].isin(["exact_boundary", "prorated"]), "profile_id"])
        classes = {
            status: set(part.loc[part["status"] == status, "profile_id"])
            for status in ("exact_boundary", "prorated")
        }
        mcu = stocks.loc[(stocks["model"] == "mcu") & stocks["profile_id"].isin(valid), ["profile_id", ycol, pcol]].rename(columns={pcol: "p_mcu"})
        mcu = mcu.merge(blocks.rename("block_id"), left_on="profile_id", right_index=True, how="left")
        paired = {}
        for model in ("rf", "mlp_integral", "hgb", "xgb"):
            other = stocks.loc[(stocks["model"] == model) & stocks["profile_id"].isin(valid), ["profile_id", pcol]].rename(columns={pcol: "p_other"})
            paired[model] = _paired(mcu.merge(other, on="profile_id"), ycol, rng)
        for status, ids in classes.items():
            sub = mcu.loc[mcu["profile_id"].isin(ids)]
            other = stocks.loc[(stocks["model"] == "rf") & stocks["profile_id"].isin(ids), ["profile_id", pcol]].rename(columns={pcol: "p_other"})
            merged = sub.merge(other, on="profile_id")
            _paired(merged, ycol, rng)
            _boot_mean((merged[ycol] - merged["p_mcu"]).abs().to_numpy(float), merged["block_id"].to_numpy(), rng)
        out[str(depth)] = paired
    return out


def main() -> None:
    fresh = replay()
    archived = json.loads((ROOT / "outputs" / "metrics" / "round11_coverage.json").read_text(encoding="utf-8"))
    report = {}
    for depth, paired in fresh.items():
        report[depth] = {}
        stored = archived["scores"][depth]["paired_other_minus_primary"]
        for model, row in paired.items():
            ref = stored[model]
            report[depth][model] = {
                "replayed_mean": row["mean"],
                "replayed_ci95": [row["ci95_low"], row["ci95_high"]],
                "archived_mean": ref["mean"],
                "archived_ci95": [ref["ci95_low"], ref["ci95_high"]],
                "matches_archived_interval": bool(np.isclose(row["ci95_low"], ref["ci95_low"]) and np.isclose(row["ci95_high"], ref["ci95_high"])),
            }
    target = report["100"]["rf"]
    print(json.dumps({"seed": SEED, "draws": N_BOOT, "sampling_unit": "spatial block", "contrasts": report}, indent=2))
    if not target["matches_archived_interval"]:
        raise SystemExit("The replayed 0–100 cm random-forest interval does not match the archived interval.")


if __name__ == "__main__":
    main()

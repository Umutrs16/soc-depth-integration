"""Recompute the published 0-100 cm scores from archived predictions.

This script does not train a model. It reads the archived out-of-fold stocks
and the union-coverage table, keeps profiles whose layers cover 0-100 cm,
and reports the prespecified comparison. The random-forest column in
round12_interval_profiles.csv is a later calibration refit and is not used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "outputs" / "metrics" / "oof_stocks.parquet"
COVER = ROOT / "outputs" / "tables" / "round11_coverage_profiles.csv"
VALID = {"exact_boundary", "prorated"}


def mae(frame: pd.DataFrame, prediction: str) -> float:
    return float((frame[prediction] - frame["y100"]).abs().mean())


def main() -> None:
    cover = pd.read_csv(COVER)
    cover["profile_id"] = cover["profile_id"].astype(str)
    valid = cover.loc[
        (cover["depth_cm"] == 100) & cover["status"].isin(VALID),
        ["profile_id"],
    ].drop_duplicates()
    stocks = pd.read_parquet(OOF)
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    spatial = stocks.loc[stocks["split"] == "spatial_block"].copy()
    primary = spatial.loc[spatial["model"] == "mcu", ["profile_id", "fold", "y100", "p100"]]
    forest = spatial.loc[spatial["model"] == "rf", ["profile_id", "p100"]].rename(columns={"p100": "p100_rf"})
    scored = primary.merge(forest, on="profile_id", how="inner").merge(valid, on="profile_id", how="inner")
    scored = scored.dropna(subset=["y100", "p100", "p100_rf"])
    primary_mae = mae(scored, "p100")
    forest_mae = mae(scored, "p100_rf")
    report = {
        "analysis_version": "union-valid-0-100",
        "n_profiles": int(len(scored)),
        "primary_mae_kg_m2": round(primary_mae, 5),
        "archived_random_forest_mae_kg_m2": round(forest_mae, 5),
        "planned_difference_forest_minus_primary": round(forest_mae - primary_mae, 5),
        "prediction_source": "archived out-of-fold predictions",
        "not_used": "round12 random-forest refit",
    }
    print(json.dumps(report, indent=2))
    if report["n_profiles"] != 3210:
        raise SystemExit(f"expected 3,210 covered profiles, found {report['n_profiles']}")
    if abs(primary_mae - 4.36234) > 0.002:
        raise SystemExit(f"primary MAE {primary_mae} does not match the published 4.362")


if __name__ == "__main__":
    main()

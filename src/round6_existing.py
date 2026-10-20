"""Comparisons that use saved predictions only. No retraining."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, PROCESSED, TABLES
from .round4_tree_pairs import _boot

ZONES = {
    "Xizang": "plateau",
    "Qinghai": "plateau",
    "Xinjiang": "northwest_arid",
    "Gansu": "northwest_arid",
    "Ningxia": "northwest_arid",
    "Inner Mongolia": "northwest_arid",
    "Heilongjiang": "northeast",
    "Jilin": "northeast",
    "Liaoning": "northeast",
}


def _finite_pair(frame: pd.DataFrame) -> dict:
    work = frame.loc[np.isfinite(frame["y100"]) & np.isfinite(frame["a"]) & np.isfinite(frame["b"])].copy()
    work["delta"] = (work["y100"] - work["a"]).abs() - (work["y100"] - work["b"]).abs()
    out = _boot(work, "delta", "block_id")
    out["mae_a"] = float((work["y100"] - work["a"]).abs().mean())
    out["mae_b"] = float((work["y100"] - work["b"]).abs().mean())
    out["sign"] = "positive means column a has the larger absolute error"
    return out


def main() -> None:
    stocks = pd.read_csv(TABLES / "round4_stocks.csv")
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    trees = pd.read_csv(TABLES / "round4_tree_stocks.csv")
    trees["profile_id"] = trees["profile_id"].astype(str)
    base = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    base["profile_id"] = base["profile_id"].astype(str)
    tuned = trees.loc[trees["model"] == "tuned_rf", ["profile_id", "fold", "y100", "p100", "block_id"]].rename(
        columns={"p100": "p_rf"}
    )
    pairs = {}
    for model in ("lowfreq", "cumulative"):
        part = stocks.loc[stocks["model"] == model, ["profile_id", "fold", "y100", "p100"]].rename(columns={"p100": "p_new"})
        merged = tuned.merge(part, on=["profile_id", "fold", "y100"])
        merged = merged.rename(columns={"p_rf": "a", "p_new": "b"})
        pairs[f"tuned_rf_vs_{model}"] = _finite_pair(merged)
    (METRICS / "round6_tree_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    cover = profiles.loc[profiles["cover_0_100"].astype(bool), ["profile_id", "province_en", "reference", "stock_0_100"]].copy()
    cover["zone"] = cover["province_en"].map(ZONES).fillna("humid_monsoon_provinces")
    cover["liu"] = cover["reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
    matrix = (
        cover.groupby(["zone", "liu"], dropna=False)
        .size()
        .rename("n_complete_0_100")
        .reset_index()
    )
    matrix.to_csv(TABLES / "round6_zone_source_counts.csv", index=False)

    pred = base.merge(cover[["profile_id", "zone", "liu", "province_en"]], on="profile_id", how="inner")
    pred = pred.loc[np.isfinite(pred["y100"]) & np.isfinite(pred["p100"])]
    rows = []
    for keys, g in pred.groupby(["zone", "liu"]):
        err = (g["y100"] - g["p100"]).abs()
        rows.append(
            {
                "zone": keys[0],
                "liu_2022": bool(keys[1]),
                "n": int(len(g)),
                "mae_primary": float(err.mean()),
                "mae_rf": float((g["y100"] - g["p100_rf"]).abs().mean()) if "p100_rf" in g else None,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "round6_zone_source_mae.csv", index=False)

    # Post-hoc scaled residual intervals. Other blocks' saved out-of-fold residuals
    # set the quantiles. This is not split conformal and not a refit.
    pred["resid"] = pred["y100"] - pred["p100"]
    pred["scale"] = np.maximum(pred["p100"].to_numpy(float), 1.0)
    scaled_hit = []
    plain_hit = []
    records = []
    for block, g in pred.groupby("block_id"):
        others = pred.loc[pred["block_id"] != block]
        plain_lo, plain_hi = np.quantile(others["resid"], [0.05, 0.95])
        ratio = others["resid"] / others["scale"]
        slo, shi = np.quantile(ratio, [0.05, 0.95])
        lo = g["p100"] + plain_lo
        hi = g["p100"] + plain_hi
        slo_b = g["p100"] + slo * g["scale"]
        shi_b = g["p100"] + shi * g["scale"]
        y = g["y100"]
        plain_hit.append((y >= lo) & (y <= hi))
        scaled_hit.append((y >= slo_b) & (y <= shi_b))
        g = g.copy()
        g["plain_hit"] = ((y >= lo) & (y <= hi)).to_numpy()
        g["scaled_hit"] = ((y >= slo_b) & (y <= shi_b)).to_numpy()
        g["scaled_width"] = (shi_b - slo_b).to_numpy()
        g["plain_width"] = (hi - lo).to_numpy()
        records.append(g[["profile_id", "p100", "y100", "plain_hit", "scaled_hit", "scaled_width", "plain_width"]])
    cal = pd.concat(records, ignore_index=True)
    cal["decile"] = pd.qcut(cal["p100"], 10, labels=False, duplicates="drop") + 1
    decile_rows = []
    for dec, g in cal.groupby("decile"):
        decile_rows.append(
            {
                "predicted_decile": int(dec),
                "n": int(len(g)),
                "picp_constant_residual": float(g["plain_hit"].mean()),
                "picp_scaled_residual": float(g["scaled_hit"].mean()),
                "mean_width_constant": float(g["plain_width"].mean()),
                "mean_width_scaled": float(g["scaled_width"].mean()),
                "mae": float((g["y100"] - g["p100"]).abs().mean()),
            }
        )
    payload = {
        "label": "post hoc rescaling of saved spatial out-of-fold residuals; not CQR and not a refit",
        "overall_picp_constant": float(cal["plain_hit"].mean()),
        "overall_picp_scaled": float(cal["scaled_hit"].mean()),
        "overall_width_constant": float(cal["plain_width"].mean()),
        "overall_width_scaled": float(cal["scaled_width"].mean()),
        "deciles": decile_rows,
        "zone_rule": "Province groups, not a published ecoregion map and not a leave-one-zone retrain.",
    }
    (METRICS / "round6_scaled_intervals.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"pairs": pairs, "intervals": payload, "zones": rows}, indent=2))


if __name__ == "__main__":
    main()

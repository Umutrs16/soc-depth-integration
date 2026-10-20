"""Boundary-label audit and profile-clustered hidden-layer summaries.

Neither refits a spatial model. The boundary alternatives change only the
evaluation target. The hidden-layer file is the round-9 within-profile test.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, PROCESSED, TABLES

TOL = 1e-3


def _piece(a: float, b: float, y: float, depth: float) -> dict | None:
    if not (a < depth - TOL and b > depth + TOL):
        return None
    width = b - a
    frac = (depth - a) / width
    return {
        "top_cm": a,
        "bottom_cm": b,
        "layer_stock": y,
        "overlap_fraction": frac,
        "proportional": y * frac,
        "decreasing_to_zero_at_base": y * (1.0 - ((b - depth) / width) ** 2),
        "increasing_from_zero_at_top": y * (frac**2),
    }


def _boundary(layers: pd.DataFrame, profiles: pd.DataFrame, depth: float) -> pd.DataFrame:
    stock_col = "stock_0_30" if depth == 30 else "stock_0_100"
    cover = profiles.loc[profiles[stock_col].notna(), ["profile_id", stock_col]].copy()
    rows = []
    grouped = {pid: g for pid, g in layers.groupby("profile_id", sort=False)}
    for pid, stock in zip(cover["profile_id"], cover[stock_col]):
        g = grouped[str(pid)].sort_values(["top_cm", "bottom_cm"])
        top = g["top_cm"].to_numpy(float)
        bot = g["bottom_cm"].to_numpy(float)
        val = g["socd_kgm2"].to_numpy(float)
        pieces = [p for p in (_piece(a, b, y, depth) for a, b, y in zip(top, bot, val)) if p]
        exact = bool(np.any(np.abs(bot - depth) <= TOL) or np.any(np.abs(top - depth) <= TOL))
        row = {
            "profile_id": str(pid),
            "depth_cm": depth,
            "recorded_stock": float(stock),
            "exact_boundary": exact and not pieces,
            "crosses_boundary": bool(pieces),
            "n_crossing_layers": len(pieces),
        }
        if len(pieces) == 1:
            piece = pieces[0]
            row.update(piece)
            row["stock_proportional"] = float(stock)
            row["stock_decreasing"] = float(stock) - piece["proportional"] + piece["decreasing_to_zero_at_base"]
            row["stock_increasing"] = float(stock) - piece["proportional"] + piece["increasing_from_zero_at_top"]
            row["prorated_share_of_stock"] = piece["proportional"] / float(stock) if float(stock) else np.nan
        else:
            row.update(
                {
                    "top_cm": np.nan,
                    "bottom_cm": np.nan,
                    "layer_stock": np.nan,
                    "overlap_fraction": np.nan,
                    "proportional": np.nan,
                    "decreasing_to_zero_at_base": np.nan,
                    "increasing_from_zero_at_top": np.nan,
                    "stock_proportional": float(stock),
                    "stock_decreasing": np.nan,
                    "stock_increasing": np.nan,
                    "prorated_share_of_stock": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _scores(frame: pd.DataFrame, pred: pd.Series, target: pd.Series) -> dict:
    use = pred.notna() & target.notna()
    err = pred.loc[use] - target.loc[use]
    return {
        "n": int(use.sum()),
        "mae": float(err.abs().mean()) if use.any() else np.nan,
        "bias_pred_minus_target": float(err.mean()) if use.any() else np.nan,
    }


def _cluster_median(detail: pd.DataFrame, column: str, rng: np.random.Generator, n_draw: int = 1000) -> dict:
    groups = {pid: part[column].to_numpy(float) for pid, part in detail.groupby("profile_id", sort=False)}
    ids = np.array(list(groups))
    point = float(np.nanmedian(detail[column].to_numpy(float)))
    draws = []
    for _ in range(n_draw):
        chosen = rng.choice(ids, size=ids.size, replace=True)
        sample = np.concatenate([groups[pid] for pid in chosen])
        draws.append(float(np.nanmedian(sample)))
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"median": point, "profile_cluster_ci95_low": float(lo), "profile_cluster_ci95_high": float(hi), "n_layers": int(len(detail)), "n_profiles": int(ids.size)}


def _macro(detail: pd.DataFrame, column: str, rng: np.random.Generator, n_draw: int = 1000) -> dict:
    means = detail.groupby("profile_id", sort=False)[column].mean()
    ids = means.index.to_numpy()
    values = means.to_numpy(float)
    point = float(np.nanmean(values))
    draws = [float(np.nanmean(rng.choice(values, size=values.size, replace=True))) for _ in range(n_draw)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "macro_mean_of_profile_means": point,
        "profile_cluster_ci95_low": float(lo),
        "profile_cluster_ci95_high": float(hi),
        "n_profiles": int(len(ids)),
    }


def _paired(detail: pd.DataFrame, left: str, right: str, rng: np.random.Generator) -> dict:
    groups = {}
    for pid, part in detail.groupby("profile_id", sort=False):
        groups[pid] = (part[left] - part[right]).to_numpy(float)
    ids = np.array(list(groups))
    all_diff = np.concatenate(list(groups.values()))
    profile_means = np.array([np.nanmean(groups[pid]) for pid in ids])
    med_draws = []
    macro_draws = []
    for _ in range(1000):
        chosen = rng.choice(ids, size=ids.size, replace=True)
        med_draws.append(float(np.nanmedian(np.concatenate([groups[pid] for pid in chosen]))))
        macro_draws.append(float(np.nanmean([np.nanmean(groups[pid]) for pid in chosen])))
    med_lo, med_hi = np.quantile(med_draws, [0.025, 0.975])
    mac_lo, mac_hi = np.quantile(macro_draws, [0.025, 0.975])
    return {
        "contrast": f"{left} minus {right}",
        "median_layer_difference": float(np.nanmedian(all_diff)),
        "median_ci95_low": float(med_lo),
        "median_ci95_high": float(med_hi),
        "macro_mean_difference": float(np.nanmean(profile_means)),
        "macro_ci95_low": float(mac_lo),
        "macro_ci95_high": float(mac_hi),
    }


def run() -> None:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    oof = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    oof["profile_id"] = oof["profile_id"].astype(str)
    frames = []
    summaries = []
    for depth in (30.0, 100.0):
        frame = _boundary(layers, profiles, depth)
        frames.append(frame)
        pred = oof.set_index("profile_id")
        joined = frame.merge(pred[["y30", "p30", "y100", "p100", "p30_rf", "p100_rf"]], on="profile_id", how="left")
        if depth == 100:
            ycol, pcol, rfcol = "y100", "p100", "p100_rf"
            joined = joined.loc[joined["y100"].notna()].copy()
        else:
            ycol, pcol, rfcol = "y30", "p30", "p30_rf"
            joined = joined.loc[joined["y30"].notna()].copy()
        exact = joined.loc[joined["exact_boundary"]]
        cross = joined.loc[joined["crosses_boundary"] & joined["stock_decreasing"].notna()]
        item = {
            "depth_cm": depth,
            "n_targets": int(len(joined)),
            "n_exact_boundary": int(joined["exact_boundary"].sum()),
            "n_crossing": int(joined["crosses_boundary"].sum()),
            "n_unclassified": int((~joined["exact_boundary"] & ~joined["crosses_boundary"]).sum()),
            "crossing_prorated_kgm2_median": float(cross["proportional"].median()) if len(cross) else None,
            "crossing_prorated_kgm2_mean": float(cross["proportional"].mean()) if len(cross) else None,
            "crossing_share_of_stock_median": float(cross["prorated_share_of_stock"].median()) if len(cross) else None,
            "abs_shift_decreasing_median": float((cross["stock_decreasing"] - cross["stock_proportional"]).abs().median()) if len(cross) else None,
            "abs_shift_increasing_median": float((cross["stock_increasing"] - cross["stock_proportional"]).abs().median()) if len(cross) else None,
            "model_not_retrained": True,
            "assumption": "proportional split assumes constant stock per unit depth inside the crossing layer; the two alternatives are linear densities that are zero at one end of that layer",
            "all_targets_primary": _scores(joined, joined[pcol], joined[ycol]),
            "all_targets_random_forest": _scores(joined, joined[rfcol], joined[ycol]),
            "exact_boundary_primary": _scores(exact, exact[pcol], exact[ycol]),
            "exact_boundary_random_forest": _scores(exact, exact[rfcol], exact[ycol]),
            "decreasing_target_primary": _scores(cross, cross[pcol], cross["stock_decreasing"]) if len(cross) else None,
            "increasing_target_primary": _scores(cross, cross[pcol], cross["stock_increasing"]) if len(cross) else None,
            "decreasing_target_random_forest": _scores(cross, cross[rfcol], cross["stock_decreasing"]) if len(cross) else None,
            "increasing_target_random_forest": _scores(cross, cross[rfcol], cross["stock_increasing"]) if len(cross) else None,
            "proportional_on_crossing_primary": _scores(cross, cross[pcol], cross[ycol]) if len(cross) else None,
            "proportional_on_crossing_random_forest": _scores(cross, cross[rfcol], cross[ycol]) if len(cross) else None,
        }
        summaries.append(item)
        if depth == 100:
            one = joined.loc[joined["profile_id"] == "3637"]
            item["profile_3637"] = one.to_dict(orient="records")
    audit = pd.concat(frames, ignore_index=True)
    audit.to_csv(TABLES / "round10_boundary_profiles.csv", index=False)
    (METRICS / "round10_boundary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    detail = pd.read_csv(TABLES / "round9_holdout_layers.csv")
    detail["profile_id"] = detail["profile_id"].astype(str)
    detail["unit_selected"] = detail["grid_selected_stock_abs"] / detail["thickness_cm"]
    detail["unit_1e-4"] = detail["grid_1e-4_stock_abs"] / detail["thickness_cm"]
    detail["unit_10"] = detail["grid_10_stock_abs"] / detail["thickness_cm"]
    detail["unit_fourier"] = detail["fourier_stock_abs"] / detail["thickness_cm"]
    detail["is_endpoint"] = (detail["top_cm"] <= TOL) | (detail["bottom_cm"] >= 100 - TOL)
    rng = np.random.default_rng(10)
    columns = {
        "selected": "grid_selected_stock_abs",
        "lambda_1e-4": "grid_1e-4_stock_abs",
        "lambda_10": "grid_10_stock_abs",
        "fourier": "fourier_stock_abs",
    }
    holdout = {
        "label": "within-profile leave-one-layer-out; interpolation when the hidden layer is between other layers, extrapolation when it is an endpoint; not a spatial score",
        "same_visible_layers_for_the_scored_fits": True,
        "lambda_selection": "When at least three visible layers remain, one of them, the layer whose midpoint is closest to the median midpoint, is hidden only to choose lambda. The scored fit then uses every visible layer, including that selection layer. Fixed penalties 1e-4 and 10 skip that choice. Two visible layers use preset lambda 0.01.",
        "not_compared_with_spatial_mae": True,
        "overall": {},
        "two_visible_layers": {},
        "three_or_more_visible_layers": {},
        "paired_profile_cluster": [],
        "by_thickness": [],
        "endpoint_share": {
            "n_endpoint": int(detail["is_endpoint"].sum()),
            "n_interior": int((~detail["is_endpoint"]).sum()),
        },
        "lambda_selected_counts": {str(k): int(v) for k, v in detail["lambda_selected"].round(6).value_counts().items()},
    }
    for name, column in columns.items():
        holdout["overall"][name] = {
            "layer_median": _cluster_median(detail, column, rng),
            "macro_mean": _macro(detail, column, rng),
        }
    two = detail.loc[detail["n_visible"] == 2]
    more = detail.loc[detail["n_visible"] >= 3]
    for label, part in (("two_visible_layers", two), ("three_or_more_visible_layers", more)):
        for name, column in columns.items():
            holdout[label][name] = {
                "layer_median": _cluster_median(part, column, rng) if len(part) else None,
                "macro_mean": _macro(part, column, rng) if len(part) else None,
            }
    for left, right in (
        ("grid_10_stock_abs", "grid_selected_stock_abs"),
        ("grid_10_stock_abs", "grid_1e-4_stock_abs"),
        ("grid_selected_stock_abs", "grid_1e-4_stock_abs"),
        ("fourier_stock_abs", "grid_selected_stock_abs"),
    ):
        holdout["paired_profile_cluster"].append(_paired(detail, left, right, rng))
    detail["thick_bin"] = pd.cut(detail["thickness_cm"], [0, 10, 20, 40, 200], labels=["0-10", "10-20", "20-40", "40-200"])
    for name, part in detail.groupby("thick_bin", observed=True):
        holdout["by_thickness"].append(
            {
                "bin": str(name),
                "n": int(len(part)),
                "median_stock_abs": float(part["grid_selected_stock_abs"].median()),
                "median_unit_abs": float(part["unit_selected"].median()),
                "median_observed_stock": float(part["observed_stock"].median()),
                "observed_stock_p25": float(part["observed_stock"].quantile(0.25)),
                "observed_stock_p75": float(part["observed_stock"].quantile(0.75)),
            }
        )
    (METRICS / "round10_holdout_cluster.json").write_text(json.dumps(holdout, indent=2), encoding="utf-8")
    print(json.dumps({"boundary": summaries, "holdout_thickness": holdout["by_thickness"], "overall_medians": {k: v["layer_median"]["median"] for k, v in holdout["overall"].items()}}, indent=2))


if __name__ == "__main__":
    run()

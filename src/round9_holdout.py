"""Hide one measured layer and reconstruct it from the other layers.

This is a within-profile diagnostic. The 1 cm grid is not Bishop's
piecewise-quadratic spline. The Fourier fit is a ridge regression on that
profile's visible layer stocks, not the spatial network. Field data have no
known q(z), so the score is the hidden layer stock.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import METRICS, PROCESSED, TABLES
from .round8_spline import _fit

LAMBDAS = np.logspace(-4, 1, 8)
PRESET_LAMBDA = 1e-2
RIDGE = 1e-2


def _mean_on(z: np.ndarray, q: np.ndarray, top: float, bottom: float) -> float:
    acc = 0.0
    weight = 0.0
    for k in range(len(z) - 1):
        left = max(top, float(z[k]))
        right = min(bottom, float(z[k + 1]))
        if right > left:
            acc += float(q[k]) * (right - left)
            weight += right - left
    return acc / weight if weight > 0 else np.nan


def _basis_row(top: float, bottom: float) -> np.ndarray:
    vals = []
    for k in range(4):
        omega = (2.0**k) * np.pi / 200.0
        vals.append((-np.cos(omega * bottom) + np.cos(omega * top)) / omega)
        vals.append((np.sin(omega * bottom) - np.sin(omega * top)) / omega)
    return np.asarray(vals, dtype=float)


def _fourier_stock(visible_top, visible_bottom, visible_stock, hidden_top, hidden_bottom) -> float:
    design = np.vstack([_basis_row(a, b) for a, b in zip(visible_top, visible_bottom)])
    gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    coef = np.linalg.solve(gram, design.T @ np.asarray(visible_stock, dtype=float))
    return float(_basis_row(hidden_top, hidden_bottom) @ coef)


def _choose_lambda(top, bottom, stock) -> tuple[float, str]:
    if len(stock) < 3:
        return PRESET_LAMBDA, "preset_0.01_fewer_than_3_visible_layers"
    mids = (top + bottom) / 2.0
    tune = int(np.argsort(np.abs(mids - np.median(mids)))[0])
    fit_idx = [i for i in range(len(stock)) if i != tune]
    best_lam = float(LAMBDAS[0])
    best_err = np.inf
    for lam in LAMBDAS:
        z, q = _fit(top[fit_idx], bottom[fit_idx], stock[fit_idx], float(lam))
        err = abs(_mean_on(z, q, float(top[tune]), float(bottom[tune])) - float(stock[tune]) / float(bottom[tune] - top[tune]))
        if err < best_err:
            best_err = err
            best_lam = float(lam)
    return best_lam, "visible_layer_with_median_midpoint"


def run() -> None:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    keep = set(profiles.loc[profiles["cover_0_100"].astype(bool) & ~profiles["has_overlap"].astype(bool), "profile_id"])
    rows = []
    usable = layers.loc[layers["profile_id"].isin(keep)].copy()
    for pid, group in usable.groupby("profile_id", sort=False):
        group = group.sort_values(["top_cm", "bottom_cm"])
        group = group.loc[(group["bottom_cm"] > group["top_cm"]) & (group["top_cm"] < 100) & (group["bottom_cm"] > 0)]
        if len(group) < 3:
            continue
        top = group["top_cm"].to_numpy(float)
        bottom = group["bottom_cm"].to_numpy(float)
        stock = group["socd_kgm2"].to_numpy(float)
        for hidden in range(len(group)):
            if bottom[hidden] > 100:
                continue
            visible = [i for i in range(len(group)) if i != hidden]
            vis_top = top[visible]
            vis_bottom = bottom[visible]
            vis_stock = stock[visible]
            lam, rule = _choose_lambda(vis_top, vis_bottom, vis_stock)
            observed = float(stock[hidden])
            thickness = float(bottom[hidden] - top[hidden])
            fits = {"selected": lam, "1e-4": 1e-4, "10": 10.0}
            record = {
                "profile_id": pid,
                "n_layers_0_100": int(len(group)),
                "n_visible": int(len(visible)),
                "top_cm": float(top[hidden]),
                "bottom_cm": float(bottom[hidden]),
                "thickness_cm": thickness,
                "mid_cm": float((top[hidden] + bottom[hidden]) / 2.0),
                "observed_stock": observed,
                "lambda_selected": lam,
                "lambda_rule": rule,
            }
            for name, value in fits.items():
                z, q = _fit(vis_top, vis_bottom, vis_stock, float(value))
                pred_mean = _mean_on(z, q, float(top[hidden]), float(bottom[hidden]))
                pred_stock = pred_mean * thickness if np.isfinite(pred_mean) else np.nan
                record[f"grid_{name}_stock_abs"] = abs(pred_stock - observed) if np.isfinite(pred_stock) else np.nan
                record[f"grid_{name}_mean_abs"] = abs(pred_mean - observed / thickness) if np.isfinite(pred_mean) else np.nan
            pred_f = _fourier_stock(vis_top, vis_bottom, vis_stock, float(top[hidden]), float(bottom[hidden]))
            record["fourier_stock_abs"] = abs(pred_f - observed)
            record["fourier_negative_stock"] = bool(pred_f < 0)
            rows.append(record)
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "round9_holdout_layers.csv", index=False)
    detail["thick_bin"] = pd.cut(detail["thickness_cm"], [0, 10, 20, 40, 200], labels=["0-10", "10-20", "20-40", "40-200"])
    detail["depth_bin"] = pd.cut(detail["mid_cm"], [0, 30, 60, 100], labels=["0-30", "30-60", "60-100"])
    summary = {
        "n_profiles": int(detail["profile_id"].nunique()),
        "n_hidden_layers": int(len(detail)),
        "not_a_spatial_score": True,
        "not_bishop_piecewise_quadratic": True,
        "fourier": "within-profile ridge fit of k=0..3 sine and cosine integrals, ridge 0.01, not the spatial network",
        "lambda_grid": [float(v) for v in LAMBDAS],
        "preset_lambda_when_fewer_than_3_visible": PRESET_LAMBDA,
        "median_grid_selected_stock_abs": float(detail["grid_selected_stock_abs"].median()),
        "median_grid_1e-4_stock_abs": float(detail["grid_1e-4_stock_abs"].median()),
        "median_grid_10_stock_abs": float(detail["grid_10_stock_abs"].median()),
        "median_fourier_stock_abs": float(detail["fourier_stock_abs"].median()),
        "by_thickness": {},
        "by_depth": {},
    }
    for column, key in (("thick_bin", "by_thickness"), ("depth_bin", "by_depth")):
        for name, part in detail.groupby(column, observed=True):
            summary[key][str(name)] = {
                "n": int(len(part)),
                "median_grid_selected_stock_abs": float(part["grid_selected_stock_abs"].median()),
                "median_grid_1e-4_stock_abs": float(part["grid_1e-4_stock_abs"].median()),
                "median_grid_10_stock_abs": float(part["grid_10_stock_abs"].median()),
                "median_fourier_stock_abs": float(part["fourier_stock_abs"].median()),
            }
    (METRICS / "round9_holdout.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()

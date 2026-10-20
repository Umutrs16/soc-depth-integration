"""Within-profile penalized equal-area reconstruction. Not a spatial score.

The fit uses only the layers of that profile. Two penalties select two
curves whose layer means both stay close to the observations. This is the
same integral-matching problem as an equal-area spline. It is not the
piecewise-quadratic basis of Bishop et al. (1999), and it is not a model
for an unsampled location.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from numpy.linalg import solve

from .paths import METRICS, PROCESSED, TABLES

LAMBDAS = (1e-4, 10.0)


def _fit(top: np.ndarray, bottom: np.ndarray, stock: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    z_max = int(np.ceil(max(float(bottom.max()), 100.0)))
    z_max = min(max(z_max, 2), 200)
    z = np.arange(z_max + 1, dtype=float)
    n = z.size
    rows = []
    targets = []
    for a, b, value in zip(top, bottom, stock):
        if not np.isfinite(value) or b <= a:
            continue
        lo = max(0, int(np.floor(a)))
        hi = min(n - 1, int(np.ceil(b)))
        if hi <= lo:
            continue
        weights = np.zeros(n)
        for k in range(lo, hi):
            left = max(a, float(k))
            right = min(b, float(k + 1))
            if right > left:
                weights[k] += right - left
        thickness = float(b - a)
        rows.append(weights / thickness)
        targets.append(float(value) / thickness)
    if len(rows) < 1:
        return z, np.full(n, np.nan)
    design = np.vstack(rows)
    y = np.asarray(targets)
    second = np.zeros((n - 2, n))
    for i in range(n - 2):
        second[i, i : i + 3] = (1.0, -2.0, 1.0)
    gram = design.T @ design + lam * (second.T @ second) + 1e-8 * np.eye(n)
    coef = solve(gram, design.T @ y)
    return z, coef


def _layer_mean_error(z: np.ndarray, q: np.ndarray, top: np.ndarray, bottom: np.ndarray, stock: np.ndarray) -> float:
    errors = []
    for a, b, value in zip(top, bottom, stock):
        if not np.isfinite(value) or b <= a:
            continue
        mask = (z >= a) & (z < b)
        if mask.sum() == 0:
            continue
        errors.append(abs(float(q[mask].mean()) - float(value) / float(b - a)))
    return float(np.median(errors)) if errors else np.nan


def _peaks(q: np.ndarray) -> int:
    dq = np.diff(q)
    return int(np.sum((dq[:-1] > 0) & (dq[1:] <= 0)))


def run() -> None:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    profiles["profile_id"] = profiles["profile_id"].astype(str)
    keep = set(profiles.loc[profiles["cover_0_100"].astype(bool) & ~profiles["has_overlap"].astype(bool), "profile_id"])
    rows = []
    for pid, group in layers.loc[layers["profile_id"].isin(keep)].groupby("profile_id", sort=False):
        top = group["top_cm"].to_numpy(float)
        bottom = group["bottom_cm"].to_numpy(float)
        stock = group["socd_kgm2"].to_numpy(float)
        curves = {}
        for lam in LAMBDAS:
            z, q = _fit(top, bottom, stock, lam)
            curves[lam] = (z, q)
            on = z <= 100
            pred100 = float(np.trapz(q[on], z[on])) if on.sum() > 1 else np.nan
            obs = profiles.loc[profiles["profile_id"] == pid, "stock_0_100"]
            observed = float(obs.iloc[0]) if len(obs) else np.nan
            rows.append(
                {
                    "profile_id": pid,
                    "lambda": lam,
                    "layer_mean_abs_median": _layer_mean_error(z, q, top, bottom, stock),
                    "s100_abs": abs(pred100 - observed) if np.isfinite(observed) else np.nan,
                    "peaks_0_100": _peaks(q[on]),
                    "negative_fraction": float(np.mean(q[on] < 0)),
                }
            )
        z0, q0 = curves[LAMBDAS[0]]
        z1, q1 = curves[LAMBDAS[1]]
        n = min(len(q0), len(q1))
        diff = q0[:n] - q1[:n]
        scale = np.sqrt(np.mean(q0[:n] ** 2))
        rows.append(
            {
                "profile_id": pid,
                "lambda": "between",
                "rel_l2": float(np.sqrt(np.mean(diff**2)) / max(scale, 1e-8)),
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "round8_spline_profiles.csv", index=False)
    summary = {"n_profiles": int(detail["profile_id"].nunique()), "lambdas": list(LAMBDAS), "not_a_spatial_score": True}
    for lam in LAMBDAS:
        part = detail.loc[detail["lambda"] == lam]
        summary[str(lam)] = {
            "median_layer_mean_abs": float(part["layer_mean_abs_median"].median()),
            "median_s100_abs": float(part["s100_abs"].median()),
            "median_peaks_0_100": float(part["peaks_0_100"].median()),
            "median_negative_fraction": float(part["negative_fraction"].median()),
        }
    between = detail.loc[detail["lambda"] == "between", "rel_l2"]
    summary["between_penalties"] = {
        "median_rel_l2": float(between.median()),
        "p95_rel_l2": float(between.quantile(0.95)),
    }
    (METRICS / "round8_spline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    run()

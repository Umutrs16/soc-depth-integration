"""Fold-safe Gaussian intervals for the archived spatial predictions.

The point predictions stay the archived out-of-fold values. A constant
residual scale and a prediction-dependent scale are fit on the other folds.
Neither replaces the archived percentile interval.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erf

from .paths import FIGURES, METRICS, TABLES

Z90 = 1.6448536269514722
SIGMA_FLOOR = 0.05
SIGMA_CAP = 200.0


def _crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (y - mu) / sigma
    cdf = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / np.sqrt(np.pi))


def _fit_scale(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    rmse = min(max(rmse, SIGMA_FLOOR), SIGMA_CAP)

    def nll(theta: np.ndarray) -> float:
        sigma = np.exp(theta[0] + theta[1] * mu)
        sigma = np.clip(sigma, SIGMA_FLOOR, SIGMA_CAP)
        z = (y - mu) / sigma
        return float(np.mean(np.log(sigma) + 0.5 * z * z))

    fit = minimize(nll, x0=np.array([np.log(rmse), 0.0]), method="L-BFGS-B")
    if not fit.success:
        return np.array([np.log(rmse), 0.0])
    return fit.x


def _score(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> dict:
    sigma = np.clip(sigma, SIGMA_FLOOR, SIGMA_CAP)
    lo = mu - Z90 * sigma
    hi = mu + Z90 * sigma
    cover = (y >= lo) & (y <= hi)
    order = np.argsort(mu)
    upper = order[int(np.floor(0.9 * len(order))) :]
    return {
        "n": int(len(y)),
        "crps": float(np.mean(_crps(y, mu, sigma))),
        "picp": float(np.mean(cover)),
        "mpiw": float(np.mean(hi - lo)),
        "upper_predicted_decile_n": int(len(upper)),
        "upper_predicted_decile_picp": float(np.mean(cover[upper])),
        "upper_predicted_decile_mpiw": float(np.mean(hi[upper] - lo[upper])),
    }


def _one_depth(frame: pd.DataFrame, ycol: str, pcol: str, locol: str, hicol: str) -> dict:
    rows = []
    for fold, test in frame.groupby("fold"):
        train = frame.loc[frame["fold"] != fold]
        y = test[ycol].to_numpy(float)
        mu = test[pcol].to_numpy(float)
        y_train = train[ycol].to_numpy(float)
        mu_train = train[pcol].to_numpy(float)
        constant = float(np.sqrt(np.mean((y_train - mu_train) ** 2)))
        theta = _fit_scale(mu_train, y_train)
        varying = np.exp(theta[0] + theta[1] * mu)
        archived_sigma = (test[hicol].to_numpy(float) - test[locol].to_numpy(float)) / (2.0 * Z90)
        rows.append(
            {
                "fold": int(fold),
                "constant_rmse_gaussian": _score(y, mu, np.full(len(y), constant)),
                "prediction_dependent_gaussian": _score(y, mu, varying),
                "archived_width_as_gaussian": _score(y, mu, archived_sigma),
                "scale_intercept": float(theta[0]),
                "scale_slope_per_kg_m2": float(theta[1]),
            }
        )
    detail = pd.DataFrame(rows)
    pooled = {}
    for name in ("constant_rmse_gaussian", "prediction_dependent_gaussian", "archived_width_as_gaussian"):
        block = pd.DataFrame(list(detail[name]))
        pooled[name] = {col: float(block[col].mean()) if col != "n" else int(block[col].sum()) for col in block.columns}
    return {"folds": rows, "pooled_over_folds": pooled}


def _figure(summary: dict) -> None:
    part = summary["stock_0_100"]["pooled_over_folds"]
    labels = ["Constant\nresidual scale", "Prediction-\ndependent scale", "Archived width\nas a Gaussian"]
    keys = ["constant_rmse_gaussian", "prediction_dependent_gaussian", "archived_width_as_gaussian"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    xpos = np.arange(len(keys))
    axes[0].bar(xpos, [part[key]["crps"] for key in keys], color=["#4d4d4d", "#2166ac", "#b2182b"], width=0.7)
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Mean CRPS (kg m$^{-2}$)", fontsize=11)
    axes[0].set_title("(a)", loc="left", fontsize=12)
    axes[1].bar(xpos - 0.15, [part[key]["mpiw"] for key in keys], width=0.3, color="#2166ac", label="All profiles")
    axes[1].bar(
        xpos + 0.15,
        [part[key]["upper_predicted_decile_mpiw"] for key in keys],
        width=0.3,
        color="#b2182b",
        label="Upper predicted decile",
    )
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Mean 90% width (kg m$^{-2}$)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    axes[1].set_title("(b)", loc="left", fontsize=12)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig20_interval_scale.png", dpi=300)
    fig.savefig(FIGURES / "fig20_interval_scale.pdf")
    plt.close(fig)


def run() -> dict:
    stocks = pd.read_parquet(METRICS / "oof_stocks.parquet")
    stocks = stocks.loc[(stocks["model"] == "mcu") & (stocks["split"] == "spatial_block")].copy()
    stocks["profile_id"] = stocks["profile_id"].astype(str)
    coverage = pd.read_csv(TABLES / "round11_coverage_profiles.csv", dtype={"profile_id": str})
    valid = {}
    for depth in (30.0, 100.0):
        part = coverage.loc[coverage["depth_cm"] == depth]
        valid[depth] = set(part.loc[part["status"].isin(["exact_boundary", "prorated"]), "profile_id"])
    summary = {
        "point_predictions": "archived spatial-block out-of-fold predictions; not refit",
        "distribution": "Gaussian centred at the archived prediction",
        "nominal_interval": "prediction plus or minus 1.64485 residual standard deviations",
        "scale_fit": "other spatial folds only",
        "does_not_replace_archived_percentile_interval": True,
        "sigma_clipped_to": [SIGMA_FLOOR, SIGMA_CAP],
    }
    for depth, ycol, pcol, locol, hicol in (
        (30.0, "y30", "p30", "lo30", "hi30"),
        (100.0, "y100", "p100", "lo100", "hi100"),
    ):
        frame = stocks.loc[stocks["profile_id"].isin(valid[depth]), [ycol, pcol, locol, hicol, "fold"]].dropna()
        summary[f"stock_0_{int(depth)}"] = _one_depth(frame, ycol, pcol, locol, hicol)
    _figure(summary)
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "round17_intervals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["stock_0_100"]["pooled_over_folds"], indent=2))
    return summary


if __name__ == "__main__":
    run()

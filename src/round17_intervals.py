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
ALPHA = 0.10
WIS_SEED = 37
N_BOOT = 2000


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


def _wis(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (hi - lo) + (2.0 / ALPHA) * np.maximum(lo - y, 0.0) + (2.0 / ALPHA) * np.maximum(y - hi, 0.0)


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
        "wis": float(np.mean(_wis(y, lo, hi))),
        "picp": float(np.mean(cover)),
        "mpiw": float(np.mean(hi - lo)),
        "upper_predicted_decile_n": int(len(upper)),
        "upper_predicted_decile_picp": float(np.mean(cover[upper])),
        "upper_predicted_decile_mpiw": float(np.mean(hi[upper] - lo[upper])),
    }


def _bounds(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma = np.clip(sigma, SIGMA_FLOOR, SIGMA_CAP)
    return mu - Z90 * sigma, mu + Z90 * sigma


def _deciles(mu: np.ndarray, y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> list[dict]:
    order = np.argsort(mu)
    rows = []
    for rank, index in enumerate(np.array_split(order, 10), start=1):
        cover = (y[index] >= lo[index]) & (y[index] <= hi[index])
        rows.append(
            {
                "decile": rank,
                "n": int(len(index)),
                "picp": float(np.mean(cover)),
                "mpiw": float(np.mean(hi[index] - lo[index])),
            }
        )
    return rows


def _block_mean(values: np.ndarray, blocks: np.ndarray, rng: np.random.Generator) -> dict:
    work = pd.DataFrame({"value": values, "block": blocks}).dropna()
    groups = {key: grp["value"].to_numpy(float) for key, grp in work.groupby("block", sort=False)}
    keys = np.array(list(groups))
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        chosen = rng.choice(keys, size=len(keys), replace=True)
        draws[i] = float(np.mean(np.concatenate([groups[key] for key in chosen])))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"n": int(len(work)), "mean": float(work["value"].mean()), "ci95_low": float(low), "ci95_high": float(high)}


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
        empirical_lo = test[locol].to_numpy(float)
        empirical_hi = test[hicol].to_numpy(float)
        empirical = _score(y, mu, np.maximum((empirical_hi - empirical_lo) / (2.0 * Z90), SIGMA_FLOOR))
        empirical["wis"] = float(np.mean(_wis(y, empirical_lo, empirical_hi)))
        empirical["picp"] = float(np.mean((y >= empirical_lo) & (y <= empirical_hi)))
        empirical["mpiw"] = float(np.mean(empirical_hi - empirical_lo))
        upper = np.argsort(mu)[int(np.floor(0.9 * len(mu))) :]
        empirical["upper_predicted_decile_picp"] = float(np.mean((y[upper] >= empirical_lo[upper]) & (y[upper] <= empirical_hi[upper])))
        empirical["upper_predicted_decile_mpiw"] = float(np.mean(empirical_hi[upper] - empirical_lo[upper]))
        empirical["crps"] = None
        rows.append(
            {
                "fold": int(fold),
                "constant_rmse_gaussian": _score(y, mu, np.full(len(y), constant)),
                "prediction_dependent_gaussian": _score(y, mu, varying),
                "archived_width_as_gaussian": _score(y, mu, archived_sigma),
                "archived_percentile_interval": empirical,
                "scale_intercept": float(theta[0]),
                "scale_slope_per_kg_m2": float(theta[1]),
                "_profiles": {
                    "profile_id": test["profile_id"].astype(str).to_numpy(),
                    "mu": mu,
                    "y": y,
                    "constant": _bounds(y, mu, np.full(len(y), constant)),
                    "dependent": _bounds(y, mu, varying),
                    "archived_gaussian": _bounds(y, mu, archived_sigma),
                    "empirical": (empirical_lo, empirical_hi),
                },
            }
        )
    profiles = [row.pop("_profiles") for row in rows]
    detail = pd.DataFrame(rows)
    names = (
        "constant_rmse_gaussian",
        "prediction_dependent_gaussian",
        "archived_width_as_gaussian",
        "archived_percentile_interval",
    )
    pooled = {}
    for name in names:
        block = pd.DataFrame(list(detail[name]))
        reduced = {}
        for col in block.columns:
            if col == "n":
                reduced[col] = int(block[col].sum())
            else:
                numeric = pd.to_numeric(block[col], errors="coerce")
                reduced[col] = None if numeric.isna().all() else float(numeric.mean())
        pooled[name] = reduced
    return {"folds": rows, "pooled_over_folds": pooled, "profiles": profiles}


def _figure(summary: dict) -> None:
    part = summary["stock_0_100"]["pooled_over_folds"]
    labels = ["Constant", "Depends on\nprediction", "Archived\nwidth"]
    keys = ["constant_rmse_gaussian", "prediction_dependent_gaussian", "archived_width_as_gaussian"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    xpos = np.arange(len(keys))
    axes[0].bar(xpos, [part[key]["crps"] for key in keys], color=["#4d4d4d", "#2166ac", "#b2182b"], width=0.7)
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels(labels, fontsize=12)
    axes[0].tick_params(labelsize=12)
    axes[0].set_ylabel("Mean CRPS (kg m$^{-2}$)", fontsize=13)
    axes[0].set_title("(a)", loc="left", fontsize=13)
    styles = {
        "constant_rmse_gaussian": ("o", "#4d4d4d", "Constant scale"),
        "prediction_dependent_gaussian": ("s", "#2166ac", "Prediction-dependent"),
        "archived_width_as_gaussian": ("D", "#b2182b", "Archived-width Gaussian"),
        "archived_percentile_interval": ("X", "black", "Archived percentile interval"),
    }
    for key, (marker, color, label) in styles.items():
        deciles = summary["stock_0_100"]["predicted_deciles"][key]
        axes[1].plot(
            [row["mpiw"] for row in deciles],
            [row["picp"] for row in deciles],
            marker=marker,
            color=color,
            ls="-",
            ms=7,
            label=label,
        )
    axes[1].axhline(0.90, color="black", lw=0.8, ls=":")
    axes[1].set_xlabel("Mean 90% width (kg m$^{-2}$)", fontsize=13)
    axes[1].set_ylabel("Coverage", fontsize=13)
    axes[1].tick_params(labelsize=12)
    axes[1].legend(frameon=False, fontsize=11, loc="lower right")
    axes[1].set_title("(b)", loc="left", fontsize=13)
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
        "interval_score": "Gneiting interval score at alpha 0.10, averaged over profiles. A larger score is worse.",
        "wis_interval_resamples_spatial_blocks": True,
        "wis_seed": WIS_SEED,
        "wis_draws": N_BOOT,
        "sigma_clipped_to": [SIGMA_FLOOR, SIGMA_CAP],
        "wider_interval_is_not_an_information_gain": True,
    }
    blocks = coverage.loc[coverage["depth_cm"] == 100.0, ["profile_id", "block_id"]].drop_duplicates("profile_id")
    for depth, ycol, pcol, locol, hicol in (
        (30.0, "y30", "p30", "lo30", "hi30"),
        (100.0, "y100", "p100", "lo100", "hi100"),
    ):
        frame = stocks.loc[stocks["profile_id"].isin(valid[depth]), ["profile_id", ycol, pcol, locol, hicol, "fold"]].dropna()
        fitted = _one_depth(frame, ycol, pcol, locol, hicol)
        if depth == 100.0:
            pieces = []
            decile_source = {key: [] for key in ("constant_rmse_gaussian", "prediction_dependent_gaussian", "archived_width_as_gaussian", "archived_percentile_interval")}
            for item in fitted.pop("profiles"):
                y = item["y"]
                mu = item["mu"]
                bounds = {
                    "constant_rmse_gaussian": item["constant"],
                    "prediction_dependent_gaussian": item["dependent"],
                    "archived_width_as_gaussian": item["archived_gaussian"],
                    "archived_percentile_interval": item["empirical"],
                }
                row = pd.DataFrame({"profile_id": item["profile_id"], "mu": mu, "y": y})
                for key, (lo, hi) in bounds.items():
                    row[key] = _wis(y, lo, hi)
                    decile_source[key].append((mu, y, lo, hi))
                pieces.append(row)
            scored = pd.concat(pieces, ignore_index=True).merge(blocks, on="profile_id", how="left")
            rng = np.random.default_rng(WIS_SEED)
            base = scored["prediction_dependent_gaussian"].to_numpy(float)
            block_ids = scored["block_id"].to_numpy()
            fitted["wis_paired_dependent_minus_alternative"] = {
                "constant_rmse_gaussian": _block_mean(base - scored["constant_rmse_gaussian"].to_numpy(float), block_ids, rng),
                "archived_width_as_gaussian": _block_mean(base - scored["archived_width_as_gaussian"].to_numpy(float), block_ids, rng),
                "archived_percentile_interval": _block_mean(base - scored["archived_percentile_interval"].to_numpy(float), block_ids, rng),
            }
            fitted["predicted_deciles"] = {}
            for key, chunks in decile_source.items():
                mu = np.concatenate([chunk[0] for chunk in chunks])
                y = np.concatenate([chunk[1] for chunk in chunks])
                lo = np.concatenate([chunk[2] for chunk in chunks])
                hi = np.concatenate([chunk[3] for chunk in chunks])
                fitted["predicted_deciles"][key] = _deciles(mu, y, lo, hi)
        else:
            fitted.pop("profiles", None)
        summary[f"stock_0_{int(depth)}"] = fitted
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "round17_intervals.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["stock_0_100"]["pooled_over_folds"], indent=2))
    return summary


if __name__ == "__main__":
    run()

"""Bishop equal-area quadratic smoothing spline on hidden interior layers.

The fit is the mass-preserving spline of Bishop et al. (1999), using the
linear system in the mpspline2 implementation of that paper. It uses only
the other layers of the same profile. It does not refit a spatial model,
does not replace archived scores, and does not choose the smoothing
parameter from the hidden-layer errors. The predeclared value is 0.1.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES, METRICS, PROCESSED, TABLES

PRIMARY_LAMBDA = 0.1
SENSITIVITY = (0.01, 1.0)
N_BOOT = 2000
SEED = 11


def _parameters(top: np.ndarray, bot: np.ndarray, value: np.ndarray, lam: float):
    n = int(len(value))
    if n < 2:
        return None
    thickness = bot - top
    if np.any(thickness <= 0) or np.any(~np.isfinite(value)):
        return None
    gap = top[1:] - bot[:-1]
    if np.any(gap < -1e-8):
        return None
    size = n - 1
    roughness = np.zeros((size, size))
    for i in range(size):
        roughness[i, i] = 2.0 * (thickness[i] + thickness[i + 1]) + 6.0 * gap[i]
    for i in range(size - 1):
        roughness[i, i + 1] = roughness[i + 1, i] = thickness[i + 1]
    difference = np.zeros((size, n))
    for i in range(size):
        difference[i, i] = -1.0
        difference[i, i + 1] = 1.0
    try:
        inverse = np.linalg.inv(roughness)
    except np.linalg.LinAlgError:
        return None
    system = np.eye(n) + (6.0 * n * lam) * (difference.T @ inverse @ difference)
    try:
        fitted = np.linalg.solve(system, value)
    except np.linalg.LinAlgError:
        return None
    slope_jump = 6.0 * (inverse @ difference @ fitted)
    slope_start = np.concatenate(([0.0], slope_jump))
    slope_end = np.concatenate((slope_jump, [0.0]))
    gamma = (slope_end - slope_start) / (2.0 * thickness)
    alpha = fitted - slope_start * thickness / 2.0 - gamma * thickness**2 / 3.0
    return {
        "top": top,
        "bottom": bot,
        "alpha": alpha,
        "slope_end": slope_end,
        "fitted_mean": fitted,
        "thickness": thickness,
        "slope_start": slope_start,
        "gamma": gamma,
    }


def _preserves_layer_means(fit: dict) -> bool:
    total = (
        fit["alpha"] * fit["thickness"]
        + fit["slope_start"] * fit["thickness"] ** 2 / 2.0
        + fit["gamma"] * fit["thickness"] ** 3 / 3.0
    )
    return bool(np.max(np.abs(total / fit["thickness"] - fit["fitted_mean"])) < 1e-8)


def _gap_stock(fit: dict, above: int, below: int, start: float, end: float) -> float:
    gap = float(fit["top"][below] - fit["bottom"][above])
    level = float(fit["alpha"][below] - fit["slope_end"][above] * gap)
    left = start - float(fit["bottom"][above])
    right = end - float(fit["bottom"][above])
    slope = float(fit["slope_end"][above])
    return float(level * (right - left) + slope * (right**2 - left**2) / 2.0)


def _eligible(top: np.ndarray, bot: np.ndarray, hidden: int) -> bool:
    above = np.where(bot <= top[hidden] + 1e-6)[0]
    below = np.where(top >= bot[hidden] - 1e-6)[0]
    above = above[above != hidden]
    below = below[below != hidden]
    return bool(len(above) and len(below))


def _predict_stock(top: np.ndarray, bot: np.ndarray, density: np.ndarray, hidden: int, lam: float) -> float | None:
    keep = np.ones(len(density), dtype=bool)
    keep[hidden] = False
    fit = _parameters(top[keep], bot[keep], density[keep], lam)
    if fit is None:
        return None
    above = np.where(fit["bottom"] <= top[hidden] + 1e-6)[0]
    below = np.where(fit["top"] >= bot[hidden] - 1e-6)[0]
    if len(above) == 0 or len(below) == 0:
        return None
    return _gap_stock(fit, int(above[-1]), int(below[0]), float(top[hidden]), float(bot[hidden]))


def _nearest_stock(top, bot, density, stock_thickness, hidden: int) -> float:
    others = np.arange(len(density))
    others = others[others != hidden]
    centre = 0.5 * (top + bot)
    target = centre[hidden]
    chosen = others[np.argmin(np.abs(centre[others] - target))]
    return float(density[chosen] * stock_thickness)


def _linear_stock(top, bot, density, hidden: int) -> float:
    above = np.where(bot <= top[hidden] + 1e-6)[0]
    below = np.where(top >= bot[hidden] - 1e-6)[0]
    above = above[above != hidden]
    below = below[below != hidden]
    left = above[np.argmax(bot[above])]
    right = below[np.argmin(top[below])]
    left_centre = 0.5 * (top[left] + bot[left])
    right_centre = 0.5 * (top[right] + bot[right])
    span = right_centre - left_centre
    midpoint = 0.5 * (top[hidden] + bot[hidden])
    if abs(span) < 1e-8:
        predicted = float(density[left])
    else:
        weight = (midpoint - left_centre) / span
        predicted = float(density[left] + weight * (density[right] - density[left]))
    return predicted * float(bot[hidden] - top[hidden])


def _profile_bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict:
    observed = float(np.mean(values))
    draws = np.empty(N_BOOT)
    size = len(values)
    for i in range(N_BOOT):
        chosen = rng.choice(size, size=size, replace=True)
        draws[i] = float(np.mean(values[chosen]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"mean": observed, "ci95_low": float(low), "ci95_high": float(high), "n_profiles": int(size)}


def _example_check() -> dict:
    top = np.array([0.0, 20.0, 40.0, 60.0])
    bot = np.array([10.0, 30.0, 50.0, 70.0])
    value = np.array([6.0, 4.0, 3.0, 10.0])
    fit = _parameters(top, bot, value, PRIMARY_LAMBDA)
    if fit is None or not _preserves_layer_means(fit):
        raise RuntimeError("The equal-area spline did not preserve the example layer means.")
    scaled = _parameters(top, bot, value * 0.05, PRIMARY_LAMBDA)
    if scaled is None or np.max(np.abs(scaled["fitted_mean"] - fit["fitted_mean"] * 0.05)) > 1e-8:
        raise RuntimeError("The smoothing matrix is not homogeneous in the layer values.")
    hidden = 1
    predicted = _predict_stock(top, bot, value, hidden, PRIMARY_LAMBDA)
    if predicted is None or not np.isfinite(predicted):
        raise RuntimeError("The gap integral did not return a finite stock.")
    return {
        "layer_means_preserved": True,
        "constant_rescaling_preserved": True,
        "lambda_depends_on_depth_geometry_not_on_a_constant_unit_change": True,
        "example_hidden_20_30_integral": float(predicted),
    }


def _macro(frame: pd.DataFrame, column: str, rng: np.random.Generator) -> dict:
    means = frame.groupby("profile_id", sort=False)[column].mean()
    return _profile_bootstrap(means.to_numpy(float), rng)


def run() -> dict:
    checks = _example_check()
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    usable = layers.loc[~layers["has_overlap"].astype(bool)].copy()
    usable = usable.loc[np.isfinite(usable["socd_kgm2"]) & ((usable["bottom_cm"] - usable["top_cm"]) > 0)]
    rows = []
    failed = 0
    for pid, group in usable.groupby("profile_id", sort=False):
        group = group.sort_values(["top_cm", "bottom_cm"])
        top = group["top_cm"].to_numpy(float)
        bot = group["bottom_cm"].to_numpy(float)
        stock = group["socd_kgm2"].to_numpy(float)
        if len(stock) < 3 or np.any(top[1:] < bot[:-1] - 1e-8):
            continue
        density = stock / (bot - top)
        for hidden in range(len(stock)):
            if not _eligible(top, bot, hidden):
                continue
            predicted = {}
            for lam in (PRIMARY_LAMBDA, *SENSITIVITY):
                value = _predict_stock(top, bot, density, hidden, lam)
                predicted[str(lam)] = np.nan if value is None else value
            if not np.isfinite(predicted[str(PRIMARY_LAMBDA)]):
                failed += 1
                continue
            thickness = float(bot[hidden] - top[hidden])
            rows.append(
                {
                    "profile_id": pid,
                    "top_cm": float(top[hidden]),
                    "bottom_cm": float(bot[hidden]),
                    "observed_stock": float(stock[hidden]),
                    "eaqss_0p1": predicted[str(PRIMARY_LAMBDA)],
                    "eaqss_0p01": predicted["0.01"],
                    "eaqss_1": predicted["1.0"],
                    "nearest": _nearest_stock(top, bot, density, thickness, hidden),
                    "linear": _linear_stock(top, bot, density, hidden),
                }
            )
    detail = pd.DataFrame(rows)
    for column in ("eaqss_0p1", "eaqss_0p01", "eaqss_1", "nearest", "linear"):
        detail[f"abs_{column}"] = (detail[column] - detail["observed_stock"]).abs()
    profile = (
        detail.groupby("profile_id", sort=False)
        .agg(
            n_layers=("observed_stock", "size"),
            mae_eaqss=("abs_eaqss_0p1", "mean"),
            mae_nearest=("abs_nearest", "mean"),
            mae_linear=("abs_linear", "mean"),
            mae_eaqss_0p01=("abs_eaqss_0p01", "mean"),
            mae_eaqss_1=("abs_eaqss_1", "mean"),
        )
        .reset_index()
    )
    rng = np.random.default_rng(SEED)
    summary = {
        "method": "Bishop equal-area quadratic smoothing spline, mpspline2 linear system",
        "primary_lambda": PRIMARY_LAMBDA,
        "lambda_was_not_chosen_from_hidden_errors": True,
        "sampling_unit": "profile",
        "bootstrap_seed": SEED,
        "bootstrap_draws": N_BOOT,
        "n_profiles": int(profile.shape[0]),
        "n_hidden_layers": int(detail.shape[0]),
        "failed_fits": int(failed),
        "not_a_spatial_score": True,
        "not_an_adaptive_sampling_trial": True,
        "unit_checks": checks,
        "profile_macro_mae": {
            "eaqss_0.1": _profile_bootstrap(profile["mae_eaqss"].to_numpy(float), rng),
            "nearest_layer": _profile_bootstrap(profile["mae_nearest"].to_numpy(float), rng),
            "linear_between_neighbours": _profile_bootstrap(profile["mae_linear"].to_numpy(float), rng),
            "eaqss_0.01_sensitivity": _profile_bootstrap(profile["mae_eaqss_0p01"].dropna().to_numpy(float), rng),
            "eaqss_1_sensitivity": _profile_bootstrap(profile["mae_eaqss_1"].dropna().to_numpy(float), rng),
        },
    }
    summary["paired_other_minus_eaqss"] = {
        "nearest_layer": _profile_bootstrap((profile["mae_nearest"] - profile["mae_eaqss"]).to_numpy(float), rng),
        "linear_between_neighbours": _profile_bootstrap((profile["mae_linear"] - profile["mae_eaqss"]).to_numpy(float), rng),
    }
    without = profile.loc[profile["profile_id"] != "3637"]
    summary["excluding_profile_3637"] = {
        "n_profiles": int(without.shape[0]),
        "mae_eaqss": float(without["mae_eaqss"].mean()),
        "mae_nearest": float(without["mae_nearest"].mean()),
        "mae_linear": float(without["mae_linear"].mean()),
    }
    summary["layer_median_abs"] = {
        "eaqss_0.1": float(detail["abs_eaqss_0p1"].median()),
        "nearest_layer": float(detail["abs_nearest"].median()),
        "linear_between_neighbours": float(detail["abs_linear"].median()),
    }
    summary["negative_predictions"] = {
        "n": int((detail["eaqss_0p1"] < 0).sum()),
        "n_hidden_layers": int(detail.shape[0]),
        "clipped_at_zero": False,
    }
    inside = detail.loc[(detail["top_cm"] >= -1e-6) & (detail["bottom_cm"] <= 100 + 1e-6)]
    inside_profile = (
        inside.groupby("profile_id", sort=False)[["abs_eaqss_0p1", "abs_nearest", "abs_linear"]].mean().reset_index()
    )
    summary["hidden_layers_inside_0_100"] = {
        "n_profiles": int(inside_profile.shape[0]),
        "n_hidden_layers": int(inside.shape[0]),
        "note": "Same eligibility rule, restricted to hidden layers contained in 0-100 cm. Not used to choose the primary set.",
        "profile_macro_mae": {
            "eaqss_0.1": _profile_bootstrap(inside_profile["abs_eaqss_0p1"].to_numpy(float), rng),
            "nearest_layer": _profile_bootstrap(inside_profile["abs_nearest"].to_numpy(float), rng),
            "linear_between_neighbours": _profile_bootstrap(inside_profile["abs_linear"].to_numpy(float), rng),
        },
        "paired_other_minus_eaqss": {
            "nearest_layer": _profile_bootstrap((inside_profile["abs_nearest"] - inside_profile["abs_eaqss_0p1"]).to_numpy(float), rng),
            "linear_between_neighbours": _profile_bootstrap((inside_profile["abs_linear"] - inside_profile["abs_eaqss_0p1"]).to_numpy(float), rng),
        },
    }
    detail["thickness_cm"] = detail["bottom_cm"] - detail["top_cm"]
    detail["signed_eaqss"] = detail["eaqss_0p1"] - detail["observed_stock"]
    detail["abs_density"] = detail["abs_eaqss_0p1"] / detail["thickness_cm"]
    detail["clipped_stock"] = np.maximum(detail["eaqss_0p1"], 0.0)
    detail["abs_clipped"] = (detail["clipped_stock"] - detail["observed_stock"]).abs()
    relative = detail.loc[detail["observed_stock"] >= 0.1].copy()
    relative["rel_abs"] = relative["abs_eaqss_0p1"] / relative["observed_stock"]
    edges = np.quantile(detail["thickness_cm"], [0.0, 0.25, 0.5, 0.75, 1.0])
    bands = {}
    for left, right, last in zip(edges[:-1], edges[1:], [False, False, False, True]):
        if last:
            part = detail.loc[(detail["thickness_cm"] >= left) & (detail["thickness_cm"] <= right + 1e-9)]
        else:
            part = detail.loc[(detail["thickness_cm"] >= left) & (detail["thickness_cm"] < right)]
        label = f"{left:.1f}–{right:.1f} cm"
        bands[label] = {
            "n_layers": int(len(part)),
            "n_profiles": int(part["profile_id"].nunique()),
            "mae_eaqss": _macro(part, "abs_eaqss_0p1", rng),
            "mae_linear": _macro(part, "abs_linear", rng),
            "paired_linear_minus_eaqss": _macro(part.assign(diff=part["abs_linear"] - part["abs_eaqss_0p1"]), "diff", rng),
        }
    shallow = detail.loc[0.5 * (detail["top_cm"] + detail["bottom_cm"]) <= 30]
    deep = detail.loc[0.5 * (detail["top_cm"] + detail["bottom_cm"]) > 30]
    inside30 = detail.loc[(detail["top_cm"] >= -1e-6) & (detail["bottom_cm"] <= 30 + 1e-6)]
    summary["additional_summaries"] = {
        "primary_remains_unclipped_lambda_0.1": True,
        "clip_at_zero_is_sensitivity_not_a_selected_model": True,
        "density_mae_kg_m2_cm": _macro(detail, "abs_density", rng),
        "signed_bias_prediction_minus_observed": _macro(detail, "signed_eaqss", rng),
        "relative_abs_where_observed_at_least_0.1": {
            **_macro(relative, "rel_abs", rng),
            "n_layers": int(len(relative)),
            "layers_excluded_below_0.1": int((detail["observed_stock"] < 0.1).sum()),
        },
        "clipped_at_zero_mae": _macro(detail, "abs_clipped", rng),
        "midpoint_at_or_below_30_cm": {
            "n_layers": int(len(shallow)),
            "mae_eaqss": _macro(shallow, "abs_eaqss_0p1", rng),
        },
        "midpoint_below_is_deeper_than_30_cm": {
            "n_layers": int(len(deep)),
            "mae_eaqss": _macro(deep, "abs_eaqss_0p1", rng),
        },
        "layers_inside_0_30": {
            "n_layers": int(len(inside30)),
            "n_profiles": int(inside30["profile_id"].nunique()),
            "mae_eaqss": _macro(inside30, "abs_eaqss_0p1", rng) if len(inside30) else None,
            "signed_bias": _macro(inside30, "signed_eaqss", rng) if len(inside30) else None,
        },
        "thickness_quartiles": bands,
        "one_hidden_layer_counts_as_one_recorded_assay": True,
        "not_a_prospective_sampling_budget": True,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "round16_eaqss_layers.csv", index=False)
    profile.to_csv(TABLES / "round16_eaqss_profiles.csv", index=False)
    (METRICS / "round16_eaqss.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _figure(summary)
    return summary


def _figure(summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    paired = summary["paired_other_minus_eaqss"]
    labels = ["Nearest − spline", "Linear − spline"]
    keys = ["nearest_layer", "linear_between_neighbours"]
    means = [paired[key]["mean"] for key in keys]
    low = [paired[key]["ci95_low"] for key in keys]
    high = [paired[key]["ci95_high"] for key in keys]
    y = np.array([1, 0])
    axes[0].errorbar(
        means,
        y,
        xerr=[np.array(means) - np.array(low), np.array(high) - np.array(means)],
        fmt="o",
        color="#2166ac",
        ecolor="#4d4d4d",
        capsize=3,
        ms=6,
    )
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].set_xlabel("Paired absolute-error difference (kg m$^{-2}$)", fontsize=10)
    axes[0].set_title("(a)", fontsize=12, loc="left")
    bands = summary["additional_summaries"]["thickness_quartiles"]
    names = list(bands)
    xpos = np.arange(len(names))
    spline = [bands[name]["mae_eaqss"]["mean"] for name in names]
    linear = [bands[name]["mae_linear"]["mean"] for name in names]
    spline_err = np.array([
        [bands[name]["mae_eaqss"]["mean"] - bands[name]["mae_eaqss"]["ci95_low"] for name in names],
        [bands[name]["mae_eaqss"]["ci95_high"] - bands[name]["mae_eaqss"]["mean"] for name in names],
    ])
    linear_err = np.array([
        [bands[name]["mae_linear"]["mean"] - bands[name]["mae_linear"]["ci95_low"] for name in names],
        [bands[name]["mae_linear"]["ci95_high"] - bands[name]["mae_linear"]["mean"] for name in names],
    ])
    axes[1].errorbar(xpos - 0.08, spline, yerr=spline_err, fmt="o", color="#2166ac", capsize=2, label="Equal-area spline")
    axes[1].errorbar(xpos + 0.08, linear, yerr=linear_err, fmt="s", color="#4d4d4d", capsize=2, label="Linear neighbours")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(names, fontsize=8, rotation=15, ha="right")
    axes[1].set_ylabel("Profile-mean absolute error (kg m$^{-2}$)", fontsize=10)
    axes[1].set_xlabel("Recorded layer thickness", fontsize=10)
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_title("(b)", fontsize=12, loc="left")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig19_eaqss_hidden.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig19_eaqss_hidden.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

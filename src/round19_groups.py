"""Signed bias of the saved equal-area spline, by thickness and depth.

Reads the layer table written by the hidden-layer spline. It does not refit
that spline and does not change the published absolute-error intervals.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES, METRICS, TABLES
from .round16_eaqss import _profile_bootstrap

SEED = 53


def _band(frame: pd.DataFrame, rng: np.random.Generator) -> dict:
    signed = frame.groupby("profile_id", sort=False)["signed_eaqss"].mean()
    paired = frame.groupby("profile_id", sort=False)["abs_linear"].mean() - frame.groupby("profile_id", sort=False)["abs_eaqss_0p1"].mean()
    return {
        "n_layers": int(len(frame)),
        "n_profiles": int(frame["profile_id"].nunique()),
        "signed_bias_prediction_minus_observed": _profile_bootstrap(signed.to_numpy(float), rng),
        "paired_linear_minus_spline": _profile_bootstrap(paired.to_numpy(float), rng),
    }


def summarise() -> dict:
    layers = pd.read_csv(TABLES / "round16_eaqss_layers.csv")
    layers["profile_id"] = layers["profile_id"].astype(str)
    layers["midpoint_cm"] = 0.5 * (layers["top_cm"] + layers["bottom_cm"])
    edges = np.quantile(layers["thickness_cm"], [0.0, 0.25, 0.5, 0.75, 1.0])
    rng = np.random.default_rng(SEED)
    thickness = {}
    published = json.loads((METRICS / "round16_eaqss.json").read_text(encoding="utf-8"))
    expected = published["additional_summaries"]["thickness_quartiles"]
    for left, right, last in zip(edges[:-1], edges[1:], [False, False, False, True]):
        if last:
            part = layers.loc[(layers["thickness_cm"] >= left) & (layers["thickness_cm"] <= right + 1e-9)]
        else:
            part = layers.loc[(layers["thickness_cm"] >= left) & (layers["thickness_cm"] < right)]
        label = f"{left:.1f}–{right:.1f} cm"
        if label not in expected or int(len(part)) != int(expected[label]["n_layers"]):
            raise RuntimeError(f"thickness band {label} has {len(part)} layers, not the published count")
        thickness[label] = _band(part, rng)
    depth = {
        "midpoint_at_or_below_30_cm": _band(layers.loc[layers["midpoint_cm"] <= 30], rng),
        "midpoint_deeper_than_30_cm": _band(layers.loc[layers["midpoint_cm"] > 30], rng),
    }
    summary = {
        "source": "round16_eaqss_layers.csv",
        "spline_was_not_refitted": True,
        "seed": SEED,
        "draws": 2000,
        "signed_error_is_prediction_minus_recorded_stock": True,
        "thickness_quartiles": thickness,
        "depth_groups": depth,
    }
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "round19_eaqss_groups.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _errorbar(ax, records: list[dict], labels: list[str], key: str, colour: str) -> None:
    means = np.array([row[key]["mean"] for row in records])
    low = np.array([row[key]["ci95_low"] for row in records])
    high = np.array([row[key]["ci95_high"] for row in records])
    y = np.arange(len(records))
    ax.errorbar(means, y, xerr=[means - low, high - means], fmt="o", color=colour, ecolor="#4d4d4d", capsize=3, ms=6)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.tick_params(labelsize=11)


def figure(groups: dict) -> None:
    published = json.loads((METRICS / "round16_eaqss.json").read_text(encoding="utf-8"))
    paired = published["paired_other_minus_eaqss"]
    bands = published["additional_summaries"]["thickness_quartiles"]
    names = list(bands)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    overall = [
        paired["nearest_layer"],
        paired["linear_between_neighbours"],
    ]
    _errorbar(axes[0, 0], [{"paired": row} for row in overall], ["Nearest − spline", "Linear − spline"], "paired", "#2166ac")
    axes[0, 0].set_xlabel("Paired absolute-error difference (kg m$^{-2}$)", fontsize=12)
    axes[0, 0].set_title("(a)  n = 3,906 profiles", fontsize=12, loc="left")
    axes[0, 0].annotate("0.026", xy=(overall[1]["mean"], 1), xytext=(14, -2), textcoords="offset points", fontsize=12, color="#2166ac", va="center")
    thick_labels = [f"{name}\nn={bands[name]['n_profiles']:,}" for name in names]
    _errorbar(
        axes[0, 1],
        [bands[name] for name in names],
        thick_labels,
        "paired_linear_minus_eaqss",
        "#2166ac",
    )
    axes[0, 1].set_xlabel("Linear − spline (kg m$^{-2}$)", fontsize=12)
    axes[0, 1].set_title("(b) Thickness", fontsize=12, loc="left")
    new_bands = groups["thickness_quartiles"]
    _errorbar(
        axes[1, 0],
        [new_bands[name] for name in names],
        thick_labels,
        "signed_bias_prediction_minus_observed",
        "#b2182b",
    )
    axes[1, 0].set_xlabel("Signed bias, prediction − recorded (kg m$^{-2}$)", fontsize=12)
    axes[1, 0].set_title("(c) Thickness", fontsize=12, loc="left")
    depth = groups["depth_groups"]
    depth_rows = [depth["midpoint_at_or_below_30_cm"], depth["midpoint_deeper_than_30_cm"]]
    depth_labels = [
        f"Midpoint ≤ 30 cm\nn={depth_rows[0]['n_profiles']:,}",
        f"Midpoint > 30 cm\nn={depth_rows[1]['n_profiles']:,}",
    ]
    _errorbar(axes[1, 1], depth_rows, depth_labels, "paired_linear_minus_spline", "#2166ac")
    axes[1, 1].set_xlabel("Linear − spline (kg m$^{-2}$)", fontsize=12)
    axes[1, 1].set_title("(d) Depth", fontsize=12, loc="left")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig19_eaqss_hidden.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig19_eaqss_hidden.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups = summarise()
    figure(groups)
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()

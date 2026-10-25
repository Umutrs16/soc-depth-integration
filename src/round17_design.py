"""Retrospective layer subsets on profiles that already have the layers.

Keeping three or five recorded layers is not a field campaign and not an
uncertainty-guided sampler. Both designs were declared before the errors
were summarised. The seven-layer subset is reported only where a profile
already has at least seven layers.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .paths import FIGURES, METRICS, PROCESSED, TABLES
from .round16_eaqss import PRIMARY_LAMBDA, _gap_stock, _parameters, _profile_bootstrap

SEED = 17
N_BOOT = 2000
DESIGNS = {
    3: (10.0, 50.0, 90.0),
    5: (10.0, 30.0, 50.0, 70.0, 90.0),
    7: (10.0, 25.0, 40.0, 55.0, 70.0, 85.0, 100.0),
}


def _fixed(centre: np.ndarray, nodes: tuple[float, ...]) -> list[int]:
    available = set(range(len(centre)))
    chosen = []
    for node in nodes:
        pick = min(available, key=lambda i: (abs(float(centre[i]) - node), i))
        chosen.append(pick)
        available.remove(pick)
    return sorted(chosen)


def _gaps(centre: np.ndarray, k: int) -> list[int]:
    order = np.argsort(centre, kind="mergesort")
    chosen = {int(order[0]), int(order[-1])}
    while len(chosen) < k:
        retained = sorted(chosen, key=lambda i: float(centre[i]))
        spans = [(float(centre[b] - centre[a]), a, b) for a, b in zip(retained[:-1], retained[1:])]
        _, left, right = max(spans)
        candidates = [i for i in range(len(centre)) if i not in chosen and centre[left] < centre[i] < centre[right]]
        if not candidates:
            break
        mid = 0.5 * (float(centre[left]) + float(centre[right]))
        chosen.add(min(candidates, key=lambda i: (abs(float(centre[i]) - mid), i)))
    return sorted(chosen)


def _from_kept(top, bot, density, keep: np.ndarray, hidden: int) -> float | None:
    fit = _parameters(top[keep], bot[keep], density[keep], PRIMARY_LAMBDA)
    if fit is None:
        return None
    above = np.where(fit["bottom"] <= top[hidden] + 1e-6)[0]
    below = np.where(fit["top"] >= bot[hidden] - 1e-6)[0]
    if len(above) == 0 or len(below) == 0:
        return None
    return _gap_stock(fit, int(above[-1]), int(below[0]), float(top[hidden]), float(bot[hidden]))


def _linear_kept(top, bot, density, keep: np.ndarray, hidden: int) -> float:
    kept = np.where(keep)[0]
    above = kept[bot[kept] <= top[hidden] + 1e-6]
    below = kept[top[kept] >= bot[hidden] - 1e-6]
    left = above[np.argmax(bot[above])]
    right = below[np.argmin(top[below])]
    left_centre = 0.5 * (top[left] + bot[left])
    right_centre = 0.5 * (top[right] + bot[right])
    span = right_centre - left_centre
    midpoint = 0.5 * (top[hidden] + bot[hidden])
    predicted = float(density[left]) if abs(span) < 1e-8 else float(density[left] + (midpoint - left_centre) / span * (density[right] - density[left]))
    return predicted * float(bot[hidden] - top[hidden])


def _nearest_kept(top, bot, density, keep: np.ndarray, hidden: int) -> float:
    kept = np.where(keep)[0]
    centre = 0.5 * (top + bot)
    chosen = kept[np.argmin(np.abs(centre[kept] - centre[hidden]))]
    return float(density[chosen] * (bot[hidden] - top[hidden]))


def _score_profile(top, bot, stock, chosen: list[int], pid: str, design: str, k: int) -> list[dict]:
    keep = np.zeros(len(stock), dtype=bool)
    keep[np.array(chosen, dtype=int)] = True
    if keep.sum() < 2:
        return []
    density = stock / (bot - top)
    rows = []
    for hidden in np.where(~keep)[0]:
        above = np.where(keep & (bot <= top[hidden] + 1e-6))[0]
        below = np.where(keep & (top >= bot[hidden] - 1e-6))[0]
        if len(above) == 0 or len(below) == 0:
            continue
        spline = _from_kept(top, bot, density, keep, int(hidden))
        if spline is None or not np.isfinite(spline):
            continue
        observed = float(stock[hidden])
        linear = _linear_kept(top, bot, density, keep, int(hidden))
        nearest = _nearest_kept(top, bot, density, keep, int(hidden))
        rows.append(
            {
                "profile_id": pid,
                "design": design,
                "k": k,
                "top_cm": float(top[hidden]),
                "bottom_cm": float(bot[hidden]),
                "observed_stock": observed,
                "abs_spline": abs(spline - observed),
                "abs_linear": abs(linear - observed),
                "abs_nearest": abs(nearest - observed),
            }
        )
    return rows


def _summarise(detail: pd.DataFrame, rng: np.random.Generator) -> dict:
    if detail.empty:
        return {"n_profiles": 0, "n_layers": 0}
    means = detail.groupby("profile_id", sort=False)[["abs_spline", "abs_linear", "abs_nearest"]].mean()
    out = {
        "n_profiles": int(len(means)),
        "n_layers": int(len(detail)),
        "spline": _profile_bootstrap(means["abs_spline"].to_numpy(float), rng),
        "linear": _profile_bootstrap(means["abs_linear"].to_numpy(float), rng),
        "nearest": _profile_bootstrap(means["abs_nearest"].to_numpy(float), rng),
        "linear_minus_spline": _profile_bootstrap((means["abs_linear"] - means["abs_spline"]).to_numpy(float), rng),
        "nearest_minus_spline": _profile_bootstrap((means["abs_nearest"] - means["abs_spline"]).to_numpy(float), rng),
    }
    return out


def _figure(summary: dict) -> None:
    labels, spline, linear = [], [], []
    for key, label in (
        ("k3_fixed_nodes", "3 fixed"),
        ("k3_gap_fill", "3 gaps"),
        ("k5_fixed_nodes", "5 fixed"),
        ("k5_gap_fill", "5 gaps"),
    ):
        part = summary["designs"].get(key)
        if not part or part.get("n_profiles", 0) == 0:
            continue
        labels.append(label)
        spline.append(part["spline"]["mean"])
        linear.append(part["linear"]["mean"])
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xpos = np.arange(len(labels))
    ax.plot(xpos, spline, "o", color="#2166ac", ms=7, label="Equal-area spline")
    ax.plot(xpos, linear, "s", color="#4d4d4d", ms=6, label="Linear neighbours")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Profile-mean absolute error (kg m$^{-2}$)", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig21_layer_subsets.png", dpi=300)
    fig.savefig(FIGURES / "fig21_layer_subsets.pdf")
    plt.close(fig)


def run() -> dict:
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    layers["profile_id"] = layers["profile_id"].astype(str)
    usable = layers.loc[~layers["has_overlap"].astype(bool)].copy()
    usable = usable.loc[np.isfinite(usable["socd_kgm2"]) & ((usable["bottom_cm"] - usable["top_cm"]) > 0)]
    rows = []
    for pid, group in usable.groupby("profile_id", sort=False):
        group = group.sort_values(["top_cm", "bottom_cm"])
        top = group["top_cm"].to_numpy(float)
        bot = group["bottom_cm"].to_numpy(float)
        stock = group["socd_kgm2"].to_numpy(float)
        if len(stock) < 4 or np.any(top[1:] < bot[:-1] - 1e-8):
            continue
        centre = 0.5 * (top + bot)
        for k, nodes in DESIGNS.items():
            if len(stock) < k + 1:
                continue
            chosen = {
                "fixed_nodes": _fixed(centre, nodes),
                "gap_fill": _gaps(centre, k),
            }
            for name, index in chosen.items():
                rows.extend(_score_profile(top, bot, stock, index, pid, name, k))
    detail = pd.DataFrame(rows)
    rng = np.random.default_rng(SEED)
    designs = {}
    for (k, name), part in detail.groupby(["k", "design"], sort=True):
        designs[f"k{int(k)}_{name}"] = _summarise(part, rng)
    summary = {
        "experiment": "retrospective omission of already recorded layers",
        "not_an_active_sampling_trial": True,
        "not_centimetre_truth": True,
        "lambda": PRIMARY_LAMBDA,
        "bootstrap_seed": SEED,
        "bootstrap_draws": N_BOOT,
        "fixed_nodes_cm": {str(k): list(v) for k, v in DESIGNS.items()},
        "designs": designs,
    }
    TABLES.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "round17_layer_subsets.csv", index=False)
    _figure(summary)
    (METRICS / "round17_layer_subsets.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: {"n_profiles": value.get("n_profiles"), "spline": value.get("spline", {}).get("mean"), "linear_minus_spline": value.get("linear_minus_spline", {}).get("mean")} for key, value in designs.items()}, indent=2))
    return summary


if __name__ == "__main__":
    run()

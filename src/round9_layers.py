"""Layer-support sweep on labeled curves. Not a field score.

Equal-thickness layers, and two-layer cuts, are integrated from known curves.
The deterministic curves use the round-8 formulas with the profile perturbation
set to zero. A smaller independent-parameter run uses the round-6 formulas.
Neither replaces an archived field score.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _load, _profile_features
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round6_synthetic import _paint, _params as _independent_params, _score
from .round7_linked import _standardize
from .round8_deterministic import _params as _deterministic_params

LABEL = "synthetic_round9_layer_support_not_a_field_score"
SEEDS = (42, 43, 44, 45, 46)


def _equal(frame: pd.DataFrame, n_layers: int) -> pd.DataFrame:
    rows = []
    thickness = 100.0 / n_layers
    for _pid, group in frame.groupby("profile_id", sort=False):
        base = group.iloc[0]
        for i in range(n_layers):
            item = base.copy()
            item["top_cm"] = i * thickness
            item["bottom_cm"] = (i + 1) * thickness
            rows.append(item)
    return pd.DataFrame(rows)


def _cut(frame: pd.DataFrame, cut_cm: float) -> pd.DataFrame:
    rows = []
    for _pid, group in frame.groupby("profile_id", sort=False):
        base = group.iloc[0]
        for top, bottom in ((0.0, cut_cm), (cut_cm, 100.0)):
            item = base.copy()
            item["top_cm"] = top
            item["bottom_cm"] = bottom
            rows.append(item)
    return pd.DataFrame(rows)


def _bootstrap_median(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    point = float(np.median(values))
    draws = [float(np.median(rng.choice(values, size=values.size, replace=True))) for _ in range(1000)]
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return point, float(lo), float(hi)


def run() -> None:
    torch.backends.cudnn.deterministic = True
    layers, profiles = _load()
    layers["profile_id"] = layers["profile_id"].astype(str)
    cover = set(profiles.loc[profiles["cover_0_100"].astype(bool), "profile_id"].astype(str))
    base = layers.loc[layers["profile_id"].isin(cover)].copy()
    ids = np.array(sorted(base["profile_id"].unique()))
    rng = np.random.default_rng(7)
    rng.shuffle(ids)
    ids = ids[:1100]
    base = base.loc[base["profile_id"].isin(set(ids))]
    fit = [str(v) for v in ids[:800]]
    val = [str(v) for v in ids[800:900]]
    test = [str(v) for v in ids[900:]]
    z_values = _standardize(base, set(fit))
    spec_lookup = ModelSpec("lookup", "mcu", n_freq=8, aux=0.0, epochs=1, patience=1, seeds=(42,))
    lookup = _profile_features(base, spec_lookup, set(fit))[0]
    specs = [
        ("highfreq", ModelSpec("highfreq", "mcu", n_freq=8, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
        ("lowfreq", ModelSpec("lowfreq", "mcu", n_freq=4, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
    ]
    designs = {f"equal_{n}": _equal(base, n) for n in (1, 2, 3, 5, 10)}
    designs.update({f"cut_{int(cut)}": _cut(base, cut) for cut in (40.0, 60.0, 80.0)})
    jobs = []
    for n in (1, 2, 3, 5, 10):
        jobs.append(("deterministic", "deep_peak", f"equal_{n}", (0.0, 0.15), SEEDS))
        jobs.append(("deterministic", "decay", f"equal_{n}", (0.15,), (42, 43, 44)))
    for cut in (40, 60, 80):
        jobs.append(("deterministic", "deep_peak", f"cut_{cut}", (0.15,), (42, 43, 44)))
    for n in (1, 2, 5, 10):
        jobs.append(("independent", "deep_peak", f"equal_{n}", (0.15,), (42, 43, 44)))
    det = {
        kind: {pid: _deterministic_params(pid, kind, z_values[pid]) for pid in map(str, ids)}
        for kind in ("decay", "deep_peak")
    }
    ind = {pid: _independent_params(pid, "deep_peak") for pid in map(str, ids)}
    detail_rows = []
    for source, kind, design_name, noises, seeds in jobs:
        params = det[kind] if source == "deterministic" else ind
        for noise in noises:
            for seed in seeds:
                painted = _paint(designs[design_name], kind, params, noise, seed)
                for model_name, spec in specs:
                    print(f"layers {source} {kind} {design_name} noise {noise} seed {seed} {model_name}", flush=True)
                    model, _mx, _device = train_mcu(painted, lookup, spec, seed, fit, val)
                    scored = _score(model, lookup, [pid for pid in test if pid in lookup], kind, params)
                    for row in scored:
                        row.update(
                            {
                                "parameter_source": source,
                                "truth": kind,
                                "design": design_name,
                                "noise": noise,
                                "seed": seed,
                                "model": model_name,
                                "label": LABEL,
                            }
                        )
                    detail_rows.extend(scored)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    detail = pd.DataFrame(detail_rows)
    detail["s30_abs"] = (detail["s30_true"] - detail["s30_pred"]).abs()
    detail["peak_le_10"] = detail["peak_abs_err_cm"] <= 10.0
    detail["peak_le_20"] = detail["peak_abs_err_cm"] <= 20.0
    detail.to_csv(TABLES / "round9_layer_profiles.csv", index=False)
    boot = np.random.default_rng(9)
    summary_rows = []
    for keys, group in detail.groupby(["parameter_source", "truth", "design", "noise", "model"]):
        by_seed = group.groupby("seed")[["rel_l2_0_100", "s30_abs", "peak_abs_err_cm", "peak_le_10", "peak_le_20"]].median()
        per_profile = group.groupby("profile_id")[["rel_l2_0_100", "s30_abs", "peak_abs_err_cm"]].median()
        rel_med, rel_lo, rel_hi = _bootstrap_median(per_profile["rel_l2_0_100"].to_numpy(), boot)
        s30_med, s30_lo, s30_hi = _bootstrap_median(per_profile["s30_abs"].to_numpy(), boot)
        peak_med, peak_lo, peak_hi = _bootstrap_median(per_profile["peak_abs_err_cm"].to_numpy(), boot)
        n_layers = 2 if str(keys[2]).startswith("cut_") else int(str(keys[2]).split("_")[1])
        max_thickness = 100.0 - float(str(keys[2]).split("_")[1]) if str(keys[2]).startswith("cut_") else 100.0 / n_layers
        if str(keys[2]).startswith("cut_"):
            cut = float(str(keys[2]).split("_")[1])
            max_thickness = max(cut, 100.0 - cut)
        summary_rows.append(
            {
                "parameter_source": keys[0],
                "truth": keys[1],
                "design": keys[2],
                "noise": float(keys[3]),
                "model": keys[4],
                "n_layers": n_layers,
                "max_thickness_cm": max_thickness,
                "n_test": int(group["profile_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "rel_l2_mean_of_seed_medians": float(by_seed["rel_l2_0_100"].mean()),
                "rel_l2_min_seed": float(by_seed["rel_l2_0_100"].min()),
                "rel_l2_max_seed": float(by_seed["rel_l2_0_100"].max()),
                "rel_l2_profile_median": rel_med,
                "rel_l2_profile_median_ci95_low": rel_lo,
                "rel_l2_profile_median_ci95_high": rel_hi,
                "s30_mean_of_seed_medians": float(by_seed["s30_abs"].mean()),
                "s30_profile_median": s30_med,
                "s30_profile_median_ci95_low": s30_lo,
                "s30_profile_median_ci95_high": s30_hi,
                "peak_err_mean_of_seed_medians": float(by_seed["peak_abs_err_cm"].mean()),
                "peak_err_min_seed": float(by_seed["peak_abs_err_cm"].min()),
                "peak_err_max_seed": float(by_seed["peak_abs_err_cm"].max()),
                "peak_err_profile_median": peak_med,
                "peak_err_profile_median_ci95_low": peak_lo,
                "peak_err_profile_median_ci95_high": peak_hi,
                "frac_peak_le_10_mean_of_seed_means": float(group.groupby("seed")["peak_le_10"].mean().mean()),
                "frac_peak_le_20_mean_of_seed_means": float(group.groupby("seed")["peak_le_20"].mean().mean()),
                "label": LABEL,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "round9_layer_summary.csv", index=False)
    config = {
        "label": LABEL,
        "not_a_field_score": True,
        "profiles": "same 1100 cover_0_100 profiles and seed-7 split as round 8",
        "n_fit": 800,
        "n_val": 100,
        "n_test": 200,
        "deterministic": "round 8 formulas with epsilon exactly 0",
        "independent": "round 6 per-profile random parameters, not a function of covariates",
        "designs": "equal layers divide 0-100 cm into 1, 2, 3, 5 or 10 pieces; cut designs are 0-c and c-100 cm",
        "intervals": "95% intervals are profile-bootstrap intervals for the median of seed-wise profile medians, 1000 resamples",
        "models": "eight-frequency and four-frequency integral networks, epochs 18, patience 4",
    }
    (METRICS / "round9_layer_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (METRICS / "round9_layer_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    from .figures import figure_layer_recovery

    figure_layer_recovery()


if __name__ == "__main__":
    run()

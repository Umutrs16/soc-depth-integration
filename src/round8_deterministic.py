"""Deterministic covariate-linked control. Not a field score.

Same profiles, split, curve families and label designs as round 7, but the
profile-level perturbation is exactly zero. Observation noise is still 0 or
0.15. This file does not replace round 7.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _load, _profile_features
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round6_synthetic import _designs, _example_curves, _paint, _score
from .round7_linked import COVARIATES, LABEL as LINKED_LABEL, _clip, _standardize

LABEL = "synthetic_round8_deterministic_covariates_not_a_field_score"
SEEDS = (42, 43, 44, 45, 46)
NOISE = (0.0, 0.15)
TRUTHS = ("decay", "deep_peak")
DESIGNS = ("irregular_layers", "one_slab_0_100")


def _params(pid: str, kind: str, z: dict[str, float]) -> dict:
    z_elev = z["elevation"]
    z_mat = z["mat_c"]
    z_map = z["map_mm"]
    z_ndvi = z["ndvi_p50"]
    if kind == "decay":
        return {
            "a": _clip(0.14 + 0.05 * np.tanh(z_map), 0.04, 0.32),
            "L": _clip(28.0 + 8.0 * np.tanh(z_mat), 10.0, 55.0),
        }
    if kind == "deep_peak":
        return {
            "a": _clip(0.045 + 0.012 * np.tanh(z_ndvi), 0.02, 0.08),
            "L": 22.0,
            "b": _clip(0.18 + 0.05 * np.tanh(z_map), 0.08, 0.32),
            "c": _clip(77.0 + 8.0 * np.tanh(z_elev), 62.0, 95.0),
            "w": 12.0,
        }
    raise KeyError(kind)


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
    designs = _designs(base)
    z_values = _standardize(base, set(fit))
    params = {kind: {pid: _params(pid, kind, z_values[pid]) for pid in map(str, ids)} for kind in TRUTHS}
    lookups = {}
    spec_for_lookup = ModelSpec("lookup", "mcu", n_freq=8, aux=0.0, epochs=1, patience=1, seeds=(42,))
    for design_name, design in designs.items():
        lookups[design_name] = _profile_features(design, spec_for_lookup, set(fit))[0]
    specs = [
        ("highfreq", ModelSpec("highfreq", "mcu", n_freq=8, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
        ("lowfreq", ModelSpec("lowfreq", "mcu", n_freq=4, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
    ]
    detail_rows = []
    curve_rows = []
    for kind in TRUTHS:
        for design_name in DESIGNS:
            for noise in NOISE:
                for seed in SEEDS:
                    painted = _paint(designs[design_name], kind, params[kind], noise, seed)
                    lookup = lookups[design_name]
                    for model_name, spec in specs:
                        print(f"deterministic {kind} {design_name} noise {noise} seed {seed} {model_name}", flush=True)
                        model, _mx, _device = train_mcu(painted, lookup, spec, seed, fit, val)
                        scored = _score(model, lookup, [pid for pid in test if pid in lookup], kind, params[kind])
                        for row in scored:
                            row.update(
                                {
                                    "truth": kind,
                                    "design": design_name,
                                    "noise": noise,
                                    "seed": seed,
                                    "model": model_name,
                                    "label": LABEL,
                                }
                            )
                        detail_rows.extend(scored)
                        if seed == 42 and noise == 0.0 and kind == "deep_peak":
                            curve_rows.extend(
                                _example_curves(
                                    model,
                                    lookup,
                                    test[:1],
                                    kind,
                                    params[kind],
                                    {"truth": kind, "design": design_name, "noise": noise, "seed": seed, "model": model_name},
                                )
                            )
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(TABLES / "round8_deterministic_profiles.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(TABLES / "round8_deterministic_curves.csv", index=False)
    metrics = ["s100_abs", "s30_abs", "s30_60_abs", "rel_l2_0_100", "rel_l2_100_200", "peaks_0_100", "peak_abs_err_cm"]
    detail["s100_abs"] = (detail["s100_true"] - detail["s100_pred"]).abs()
    detail["s30_abs"] = (detail["s30_true"] - detail["s30_pred"]).abs()
    detail["s30_60_abs"] = (detail["s30_60_true"] - detail["s30_60_pred"]).abs()
    summary_rows = []
    for keys, group in detail.groupby(["truth", "design", "noise", "model"]):
        by_seed = group.groupby("seed")[metrics].median()
        item = {
            "truth": keys[0],
            "design": keys[1],
            "noise": float(keys[2]),
            "model": keys[3],
            "n_test": int(group["profile_id"].nunique()),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metrics:
            item[f"{metric}_mean_of_seed_medians"] = float(by_seed[metric].mean())
            item[f"{metric}_min_seed"] = float(by_seed[metric].min())
            item[f"{metric}_max_seed"] = float(by_seed[metric].max())
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "round8_deterministic_summary.csv", index=False)
    config = {
        "label": LABEL,
        "compared_with": LINKED_LABEL,
        "difference_from_round7": "profile perturbation epsilon is 0; covariates fully determine a, L, b and c",
        "same_profiles_and_split_as": "round6 and round7, seed 7, first 1100 cover_0_100 profiles",
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(test),
        "seeds": list(SEEDS),
        "noise_sd_fraction": list(NOISE),
        "covariates": list(COVARIATES),
        "decay": "a=clip(0.14+0.05*tanh(z_map), 0.04, 0.32); L=clip(28+8*tanh(z_mat), 10, 55)",
        "deep_peak": "a=clip(0.045+0.012*tanh(z_ndvi), 0.02, 0.08); L=22; b=clip(0.18+0.05*tanh(z_map), 0.08, 0.32); c=clip(77+8*tanh(z_elev), 62, 95); w=12",
        "not_a_field_score": True,
    }
    (METRICS / "round8_deterministic_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (METRICS / "round8_deterministic_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()

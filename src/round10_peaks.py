"""Peak-width and peak-centre sensitivity. Not a field score.

The baseline deterministic deep peak is not refit here except for three
extra noise seeds. Width and centre changes are separate files.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _load, _profile_features
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round6_synthetic import _paint, _score
from .round7_linked import _standardize
from .round8_deterministic import _params as _deterministic_params
from .round9_layers import _equal

LABEL = "synthetic_round10_peak_geometry_not_a_field_score"


def _variant(base: dict[str, dict], name: str) -> dict[str, dict]:
    out = {}
    for pid, params in base.items():
        item = dict(params)
        if name == "narrow_w6":
            item["w"] = 6.0
        elif name == "wide_w24":
            item["w"] = 24.0
        elif name == "centre_40":
            item["c"] = 40.0
            item["w"] = 12.0
        elif name == "centre_90":
            item["c"] = 90.0
            item["w"] = 12.0
        out[pid] = item
    return out


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
    original = {pid: _deterministic_params(pid, "deep_peak", z_values[pid]) for pid in map(str, ids)}
    spec_lookup = ModelSpec("lookup", "mcu", n_freq=8, aux=0.0, epochs=1, patience=1, seeds=(42,))
    lookup = _profile_features(base, spec_lookup, set(fit))[0]
    specs = [
        ("highfreq", ModelSpec("highfreq", "mcu", n_freq=8, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
        ("lowfreq", ModelSpec("lowfreq", "mcu", n_freq=4, smooth=0.02, aux=0.0, epochs=18, patience=4, seeds=(42,))),
    ]
    designs = {f"equal_{n}": _equal(base, n) for n in (1, 2, 3, 5, 10)}
    jobs = []
    for name in ("narrow_w6", "wide_w24", "centre_40", "centre_90"):
        jobs.append((name, _variant(original, name), (42, 43, 44)))
    jobs.append(("baseline_extra_seeds", original, (47, 48, 49)))
    rows = []
    for name, params, seeds in jobs:
        for n_name, design in designs.items():
            for seed in seeds:
                painted = _paint(design, "deep_peak", params, 0.15, seed)
                for model_name, spec in specs:
                    print(f"round10 {name} {n_name} seed {seed} {model_name}", flush=True)
                    model, _mx, _device = train_mcu(painted, lookup, spec, seed, fit, val)
                    scored = _score(model, lookup, [pid for pid in test if pid in lookup], "deep_peak", params)
                    for row in scored:
                        row.update(
                            {
                                "variant": name,
                                "design": n_name,
                                "noise": 0.15,
                                "seed": seed,
                                "model": model_name,
                                "label": LABEL,
                            }
                        )
                    rows.extend(scored)
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    detail = pd.DataFrame(rows)
    detail["s30_abs"] = (detail["s30_true"] - detail["s30_pred"]).abs()
    detail["s30_60_abs"] = (detail["s30_60_true"] - detail["s30_60_pred"]).abs()
    detail["peak_le_20"] = detail["peak_abs_err_cm"] <= 20.0
    detail.to_csv(TABLES / "round10_peak_profiles.csv", index=False)
    summary_rows = []
    for keys, group in detail.groupby(["variant", "design", "model"]):
        by_seed = group.groupby("seed")[["rel_l2_0_100", "s30_abs", "s30_60_abs", "peak_abs_err_cm"]].median()
        summary_rows.append(
            {
                "variant": keys[0],
                "design": keys[1],
                "model": keys[2],
                "n_layers": int(str(keys[1]).split("_")[1]),
                "n_test": int(group["profile_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "rel_l2_mean_of_seed_medians": float(by_seed["rel_l2_0_100"].mean()),
                "peak_err_mean_of_seed_medians": float(by_seed["peak_abs_err_cm"].mean()),
                "s30_mean_of_seed_medians": float(by_seed["s30_abs"].mean()),
                "s30_60_mean_of_seed_medians": float(by_seed["s30_60_abs"].mean()),
                "frac_peak_le_20": float(group.groupby("seed")["peak_le_20"].mean().mean()),
                "label": LABEL,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "round10_peak_summary.csv", index=False)
    (METRICS / "round10_peak_config.json").write_text(
        json.dumps(
            {
                "label": LABEL,
                "not_a_field_score": True,
                "not_a_sampling_recommendation": True,
                "noise": 0.15,
                "variants": {
                    "narrow_w6": "same covariate-linked centre, Gaussian width 6 cm instead of 12 cm",
                    "wide_w24": "same covariate-linked centre, Gaussian width 24 cm",
                    "centre_40": "peak fixed at 40 cm, width 12 cm",
                    "centre_90": "peak fixed at 90 cm, width 12 cm",
                    "baseline_extra_seeds": "original width 12 cm and elevation-linked centre, seeds 47-49 only",
                },
                "designs": "equal layers 1, 2, 3, 5, 10",
                "seeds_for_geometry": [42, 43, 44],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()

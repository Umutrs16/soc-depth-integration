"""Labeled synthetic recovery. Not a field score.

Real 0–100 cm layer boundaries and covariates are kept. The carbon labels are
replaced by exact integrals of two known nonnegative curves, plus noise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.special import erf

from .experiment import _load, _profile_features
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round4_experiments import _peaks, _q_grid, _stock

LABEL = "synthetic_known_q_not_a_field_score"


def q_decay(z: np.ndarray) -> np.ndarray:
    return 0.20 * np.exp(-z / 25.0)


def q_bulge(z: np.ndarray) -> np.ndarray:
    return 0.08 * np.exp(-z / 40.0) + 0.12 * np.exp(-((z - 40.0) / 12.0) ** 2)


def stock_decay(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    return 0.20 * 25.0 * (np.exp(-top / 25.0) - np.exp(-bottom / 25.0))


def stock_bulge(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    decay = 0.08 * 40.0 * (np.exp(-top / 40.0) - np.exp(-bottom / 40.0))
    scale = 0.12 * 12.0 * np.sqrt(np.pi) / 2.0
    bulge = scale * (erf((bottom - 40.0) / 12.0) - erf((top - 40.0) / 12.0))
    return decay + bulge


CURVES = {"decay": (q_decay, stock_decay), "bulge": (q_bulge, stock_bulge)}


def _designs(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    irregular = frame.copy()
    slabs = []
    single = []
    for pid, g in frame.groupby("profile_id", sort=False):
        row = g.iloc[0]
        for top, bottom, name in ((0.0, 30.0, slabs), (30.0, 100.0, slabs)):
            item = row.copy()
            item["top_cm"] = top
            item["bottom_cm"] = bottom
            name.append(item)
        item = row.copy()
        item["top_cm"] = 0.0
        item["bottom_cm"] = 100.0
        single.append(item)
    return {
        "irregular_layers": irregular,
        "two_slabs": pd.DataFrame(slabs),
        "one_slab_0_100": pd.DataFrame(single),
    }


def _paint(frame: pd.DataFrame, stock_fn, rng: np.random.Generator) -> pd.DataFrame:
    out = frame.copy()
    top = out["top_cm"].to_numpy(float)
    bottom = out["bottom_cm"].to_numpy(float)
    truth = stock_fn(top, bottom)
    noise = rng.normal(0.0, 0.15 * np.maximum(truth, 0.05))
    out["socd_kgm2"] = np.clip(truth + noise, 0, None)
    out["soc_gkg"] = 0.0
    out["bd_gcm3"] = 0.0
    out["soc_observed"] = False
    out["bd_observed"] = False
    out["closure_ok"] = False
    out["train_weight"] = 1.0
    return out


def _score(name, models, lookup, ids, q_fn, stock_fn) -> dict:
    matrix = np.stack([lookup[pid] for pid in ids])
    device = next(models[0].parameters()).device
    x = torch.tensor(matrix, dtype=torch.float32, device=device)
    q, z = _q_grid(models, x, 1.0)
    truth = q_fn(z)[None, :]
    diff = q - truth
    l2 = np.sqrt(np.mean(diff**2, axis=1))
    scale = np.sqrt(np.mean(truth**2))
    pred100 = _stock(models, x, 100.0)
    pred30 = _stock(models, x, 30.0)
    true100 = float(stock_fn(np.array([0.0]), np.array([100.0]))[0])
    true30 = float(stock_fn(np.array([0.0]), np.array([30.0]))[0])
    return {
        "model": name,
        "n": int(len(ids)),
        "median_l2": float(np.median(l2)),
        "median_l2_over_rms": float(np.median(l2) / max(scale, 1e-8)),
        "median_peaks": float(np.median(_peaks(q))),
        "true_peaks": int(_peaks(truth)[0]),
        "mae_s100": float(np.mean(np.abs(pred100 - true100))),
        "mae_s30": float(np.mean(np.abs(pred30 - true30))),
        "true_s100": true100,
    }


def run() -> None:
    torch.backends.cudnn.deterministic = True
    layers, profiles = _load()
    layers["profile_id"] = layers["profile_id"].astype(str)
    cover = profiles.loc[profiles["cover_0_100"].astype(bool), "profile_id"].astype(str)
    base = layers.loc[layers["profile_id"].isin(set(cover))].copy()
    ids = np.array(sorted(base["profile_id"].unique()))
    rng = np.random.default_rng(42)
    rng.shuffle(ids)
    ids = ids[:1200]
    base = base.loc[base["profile_id"].isin(set(ids))]
    fit, val, test = ids[:900], ids[900:1050], ids[1050:]
    designs = _designs(base)
    rows = []
    specs = [
        ("highfreq", ModelSpec("highfreq", kind="mcu", n_freq=8, smooth=0.02, epochs=25, patience=6, seeds=(42,))),
        ("lowfreq", ModelSpec("lowfreq", kind="mcu", n_freq=4, smooth=0.02, epochs=25, patience=6, seeds=(42,))),
    ]
    for curve_name, (q_fn, stock_fn) in CURVES.items():
        for design_name, design in designs.items():
            painted = _paint(design.loc[design["profile_id"].isin(set(ids))], stock_fn, np.random.default_rng(42))
            for model_name, spec in specs:
                print(f"synthetic {curve_name} {design_name} {model_name}", flush=True)
                lookup, *_rest = _profile_features(painted, spec, set(fit))
                model, _mx, _device = train_mcu(painted, lookup, spec, 42, list(fit), list(val))
                shape = _score(model_name, [model], lookup, [str(pid) for pid in test if str(pid) in lookup], q_fn, stock_fn)
                test_rows = painted.loc[painted["profile_id"].isin(set(test))]
                del test_rows
                shape.update(
                    {
                        "curve": curve_name,
                        "design": design_name,
                        "label": LABEL,
                        "noise": "gaussian, sd = 0.15 * true layer stock",
                        "n_fit": int(len(fit)),
                        "n_test": int(shape["n"]),
                    }
                )
                rows.append(shape)
                del model
                torch.cuda.empty_cache()
    payload = {"label": LABEL, "runs": rows}
    (METRICS / "round5_synthetic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(TABLES / "round5_synthetic.csv", index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()

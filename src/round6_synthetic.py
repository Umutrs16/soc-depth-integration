"""Multi-seed synthetic recovery. Not a field score.

Four known nonnegative curves, three noise levels, four label designs, and
five training seeds. Real covariates and, for the irregular design, real layer
boundaries are kept. Carbon labels are replaced.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch
from scipy.special import erf

from .experiment import _load, _profile_features
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round4_experiments import _peaks, _q_grid

LABEL = "synthetic_round6_not_a_field_score"
SEEDS = (42, 43, 44, 45, 46)
NOISE = (0.0, 0.15, 0.30)
TRUTHS = ("decay", "bulge", "horizons", "deep_peak")
DESIGNS = ("irregular_layers", "two_slabs", "one_slab_0_100", "regular_10cm")


def _seed_from(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def _params(pid: str, kind: str) -> dict:
    rng = np.random.default_rng(_seed_from(f"{pid}|{kind}"))
    if kind == "decay":
        return {"a": float(rng.uniform(0.08, 0.30)), "L": float(rng.uniform(12.0, 45.0))}
    if kind == "bulge":
        return {
            "a": float(rng.uniform(0.04, 0.16)),
            "L": float(rng.uniform(20.0, 50.0)),
            "b": float(rng.uniform(0.08, 0.25)),
            "c": float(rng.uniform(20.0, 50.0)),
            "w": float(rng.uniform(8.0, 18.0)),
        }
    if kind == "horizons":
        cuts = np.sort(rng.uniform(15.0, 85.0, size=2))
        return {
            "e1": float(cuts[0]),
            "e2": float(cuts[1]),
            "q1": float(rng.uniform(0.04, 0.28)),
            "q2": float(rng.uniform(0.02, 0.20)),
            "q3": float(rng.uniform(0.01, 0.12)),
        }
    if kind == "deep_peak":
        return {
            "a": float(rng.uniform(0.02, 0.08)),
            "L": float(rng.uniform(15.0, 30.0)),
            "b": float(rng.uniform(0.12, 0.30)),
            "c": float(rng.uniform(65.0, 90.0)),
            "w": float(rng.uniform(8.0, 16.0)),
        }
    raise KeyError(kind)


def _q(kind: str, z: np.ndarray, p: dict) -> np.ndarray:
    if kind == "decay":
        return p["a"] * np.exp(-z / p["L"])
    if kind == "bulge":
        return p["a"] * np.exp(-z / p["L"]) + p["b"] * np.exp(-((z - p["c"]) / p["w"]) ** 2)
    if kind == "horizons":
        out = np.full(z.shape, p["q3"], dtype=float)
        out = np.where(z < p["e2"], p["q2"], out)
        out = np.where(z < p["e1"], p["q1"], out)
        return out
    if kind == "deep_peak":
        return p["a"] * np.exp(-z / p["L"]) + p["b"] * np.exp(-((z - p["c"]) / p["w"]) ** 2)
    raise KeyError(kind)


def _stock_one(kind: str, top: float, bottom: float, p: dict) -> float:
    if bottom <= top:
        return 0.0
    if kind == "decay":
        return p["a"] * p["L"] * (np.exp(-top / p["L"]) - np.exp(-bottom / p["L"]))
    if kind in ("bulge", "deep_peak"):
        decay = p["a"] * p["L"] * (np.exp(-top / p["L"]) - np.exp(-bottom / p["L"]))
        scale = p["b"] * p["w"] * np.sqrt(np.pi) / 2.0
        bulge = scale * (erf((bottom - p["c"]) / p["w"]) - erf((top - p["c"]) / p["w"]))
        return float(decay + bulge)
    if kind == "horizons":
        edges = (0.0, p["e1"], p["e2"], 800.0)
        levels = (p["q1"], p["q2"], p["q3"])
        total = 0.0
        for edge0, edge1, level in zip(edges[:-1], edges[1:], levels):
            lo = max(top, edge0)
            hi = min(bottom, edge1)
            if hi > lo:
                total += level * (hi - lo)
        return float(total)
    raise KeyError(kind)


def _stocks(kind: str, top: np.ndarray, bottom: np.ndarray, params: list[dict]) -> np.ndarray:
    return np.array([_stock_one(kind, float(a), float(b), p) for a, b, p in zip(top, bottom, params)])


def _designs(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    slabs = []
    single = []
    regular = []
    for _pid, g in frame.groupby("profile_id", sort=False):
        row = g.iloc[0]
        for top, bottom, bucket in ((0.0, 30.0, slabs), (30.0, 100.0, slabs)):
            item = row.copy()
            item["top_cm"] = top
            item["bottom_cm"] = bottom
            bucket.append(item)
        item = row.copy()
        item["top_cm"] = 0.0
        item["bottom_cm"] = 100.0
        single.append(item)
        for top in range(0, 100, 10):
            item = row.copy()
            item["top_cm"] = float(top)
            item["bottom_cm"] = float(top + 10)
            regular.append(item)
    return {
        "irregular_layers": frame.copy(),
        "two_slabs": pd.DataFrame(slabs),
        "one_slab_0_100": pd.DataFrame(single),
        "regular_10cm": pd.DataFrame(regular),
    }


def _paint(frame: pd.DataFrame, kind: str, params: dict[str, dict], noise: float, seed: int) -> pd.DataFrame:
    out = frame.copy()
    pid = out["profile_id"].astype(str).to_numpy()
    top = out["top_cm"].to_numpy(float)
    bottom = out["bottom_cm"].to_numpy(float)
    truth = _stocks(kind, top, bottom, [params[i] for i in pid])
    rng = np.random.default_rng(seed + int(noise * 1000))
    sigma = noise * np.maximum(truth, 0.05)
    noisy = truth if noise == 0 else truth + rng.normal(0.0, sigma)
    out["socd_kgm2"] = np.clip(noisy, 0, None)
    out["soc_gkg"] = 0.0
    out["bd_gcm3"] = 0.0
    out["soc_observed"] = False
    out["bd_observed"] = False
    out["closure_ok"] = False
    out["train_weight"] = 1.0
    return out


def _interval(model, x: torch.Tensor, top: float, bottom: float) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        h = model.encode(x)
        t = torch.full((x.shape[0], 1), top, device=x.device)
        b = torch.full((x.shape[0], 1), bottom, device=x.device)
        return model.integrate(h, t, b).detach().cpu().numpy()[:, 0]


def _score(model, lookup, ids, kind, params) -> list[dict]:
    matrix = np.stack([lookup[pid] for pid in ids])
    device = next(model.parameters()).device
    x = torch.tensor(matrix, dtype=torch.float32, device=device)
    q, z = _q_grid([model], x, 1.0)
    on = (z >= 0) & (z <= 100)
    below = (z > 100) & (z <= 200)
    rows = []
    pred100 = _interval(model, x, 0.0, 100.0)
    pred30 = _interval(model, x, 0.0, 30.0)
    pred3060 = _interval(model, x, 30.0, 60.0)
    z_on = z[on]
    for i, pid in enumerate(ids):
        truth = _q(kind, z, params[pid])
        diff = q[i, on] - truth[on]
        scale = np.sqrt(np.mean(truth[on] ** 2))
        diff_below = q[i, below] - truth[below]
        scale_below = np.sqrt(np.mean(truth[below] ** 2))
        peak_true = float(z_on[int(np.argmax(truth[on]))])
        peak_hat = float(z_on[int(np.argmax(q[i, on]))])
        true100 = _stock_one(kind, 0.0, 100.0, params[pid])
        true30 = _stock_one(kind, 0.0, 30.0, params[pid])
        true3060 = _stock_one(kind, 30.0, 60.0, params[pid])
        rows.append(
            {
                "profile_id": pid,
                "s100_true": true100,
                "s100_pred": float(pred100[i]),
                "s30_true": true30,
                "s30_pred": float(pred30[i]),
                "s30_60_true": true3060,
                "s30_60_pred": float(pred3060[i]),
                "rel_l2_0_100": float(np.sqrt(np.mean(diff**2)) / max(scale, 1e-8)),
                "rel_l2_100_200": float(np.sqrt(np.mean(diff_below**2)) / max(scale_below, 1e-8)),
                "peaks_0_100": int(_peaks(q[i : i + 1, on])[0]),
                "peaks_true_0_100": int(_peaks(truth[on][None, :])[0]),
                "peak_cm_true": peak_true,
                "peak_cm_pred": peak_hat,
                "peak_abs_err_cm": abs(peak_hat - peak_true),
            }
        )
    return rows


def _example_curves(model, lookup, ids, kind, params, meta: dict) -> list[dict]:
    matrix = np.stack([lookup[pid] for pid in ids])
    device = next(model.parameters()).device
    x = torch.tensor(matrix, dtype=torch.float32, device=device)
    q, z = _q_grid([model], x, 1.0)
    rows = []
    for i, pid in enumerate(ids):
        truth = _q(kind, z, params[pid])
        for depth, pred, true in zip(z, q[i], truth):
            if depth > 120:
                continue
            rows.append({**meta, "profile_id": pid, "depth_cm": float(depth), "q_pred": float(pred), "q_true": float(true)})
    return rows


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
    fit, val, test = [str(v) for v in ids[:800]], [str(v) for v in ids[800:900]], [str(v) for v in ids[900:]]
    designs = _designs(base)
    params = {kind: {pid: _params(pid, kind) for pid in map(str, ids)} for kind in TRUTHS}
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
                        print(f"synthetic {kind} {design_name} noise {noise} seed {seed} {model_name}", flush=True)
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
                        if seed == 42 and noise == 0.15 and design_name in ("irregular_layers", "one_slab_0_100"):
                            curve_rows.extend(
                                _example_curves(
                                    model,
                                    lookup,
                                    test[:2],
                                    kind,
                                    params[kind],
                                    {"truth": kind, "design": design_name, "noise": noise, "seed": seed, "model": model_name},
                                )
                            )
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(TABLES / "round6_synthetic_profiles.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(TABLES / "round6_synthetic_curves.csv", index=False)
    summary_rows = []
    metrics = ["s100_abs", "s30_abs", "s30_60_abs", "rel_l2_0_100", "rel_l2_100_200", "peaks_0_100", "peak_abs_err_cm"]
    detail["s100_abs"] = (detail["s100_true"] - detail["s100_pred"]).abs()
    detail["s30_abs"] = (detail["s30_true"] - detail["s30_pred"]).abs()
    detail["s30_60_abs"] = (detail["s30_60_true"] - detail["s30_60_pred"]).abs()
    for keys, g in detail.groupby(["truth", "design", "noise", "model"]):
        by_seed = g.groupby("seed")[metrics].median()
        item = {"truth": keys[0], "design": keys[1], "noise": float(keys[2]), "model": keys[3], "n_test": int(g["profile_id"].nunique()), "n_seeds": int(g["seed"].nunique())}
        for metric in metrics:
            item[f"{metric}_mean_of_seed_medians"] = float(by_seed[metric].mean())
            item[f"{metric}_min_seed"] = float(by_seed[metric].min())
            item[f"{metric}_max_seed"] = float(by_seed[metric].max())
        summary_rows.append(item)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(TABLES / "round6_synthetic_summary.csv", index=False)
    (METRICS / "round6_synthetic_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    config = {
        "label": LABEL,
        "n_fit": len(fit),
        "n_val": len(val),
        "n_test": len(test),
        "seeds": list(SEEDS),
        "noise_sd_fraction": list(NOISE),
        "truths": {
            "decay": "q=a*exp(-z/L), a and L drawn once per profile",
            "bulge": "decay plus b*exp(-((z-c)/w)^2), c between 20 and 50 cm",
            "horizons": "three nonnegative constant pieces with two cuts between 15 and 85 cm",
            "deep_peak": "small decay plus a Gaussian peak centred between 65 and 90 cm",
        },
        "designs": {
            "irregular_layers": "recorded layer boundaries",
            "two_slabs": "only 0-30 and 30-100",
            "one_slab_0_100": "only the 0-100 total; internal labels removed",
            "regular_10cm": "ten layers of 10 cm on the same profiles",
        },
        "scores": "0-100, 0-30 and 30-60 integrals, relative L2 on 0-100 and on 100-200, peak count and peak-depth error on 0-100",
        "not_a_field_score": True,
    }
    (METRICS / "round6_synthetic_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()

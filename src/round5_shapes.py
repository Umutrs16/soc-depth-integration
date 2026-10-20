"""Profile-level disagreement among depth curves, plus a smoothness-weight sweep.

Retrains the eight-frequency, four-frequency, and cumulative-stock models on the
archived 2 degree folds. Saves curve distances. Does not replace archived scores.
A one-seed sweep then varies the fine-grid penalty weight.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from .experiment import _folds, _load, _profile_features, _profile_table, _split_fit_cal_val
from .integral_audit import _mae
from .models import ModelSpec, predict_stock_batch, train_mcu
from .paths import METRICS, TABLES
from .round4_experiments import _eval_fold, _peaks, _q_grid

PAIRS = (("highfreq", "lowfreq"), ("highfreq", "cumulative"), ("lowfreq", "cumulative"))


def _specs() -> list[tuple[str, ModelSpec]]:
    common = dict(kind="mcu", smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=50, patience=8)
    return [
        ("highfreq", ModelSpec("highfreq", n_freq=8, **common)),
        ("lowfreq", ModelSpec("lowfreq", n_freq=4, **common)),
        ("cumulative", ModelSpec("cumulative", n_freq=4, cumulative_dz=2.0, **common)),
    ]


def _curve_rows(name_a, name_b, qa, qb, ids, fold):
    diff = qa - qb
    l1 = np.mean(np.abs(diff), axis=1)
    l2 = np.sqrt(np.mean(diff**2, axis=1))
    scale = np.maximum(np.sqrt(np.mean(qa**2, axis=1)), 1e-8)
    rows = []
    for i, pid in enumerate(ids):
        rows.append(
            {
                "fold": fold,
                "profile_id": pid,
                "pair": f"{name_a}_vs_{name_b}",
                "l1": float(l1[i]),
                "l2": float(l2[i]),
                "l2_over_rms": float(l2[i] / scale[i]),
            }
        )
    return rows


def _summarize_pair(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "median_l1": float(frame["l1"].median()),
        "median_l2": float(frame["l2"].median()),
        "median_l2_over_rms": float(frame["l2_over_rms"].median()),
        "p95_l2_over_rms": float(frame["l2_over_rms"].quantile(0.95)),
    }


def _stock_gap(a: pd.DataFrame, b: pd.DataFrame, col: str) -> dict:
    merged = a.merge(b, on=["profile_id", "fold"], suffixes=("_a", "_b"))
    delta = (merged[f"{col}_a"] - merged[f"{col}_b"]).abs()
    ok = np.isfinite(delta)
    corr = float(np.corrcoef(merged.loc[ok, f"{col}_a"], merged.loc[ok, f"{col}_b"])[0, 1]) if ok.sum() > 2 else None
    return {"n": int(ok.sum()), "median_abs": float(np.median(delta[ok])), "p95_abs": float(np.quantile(delta[ok], 0.95)), "corr": corr}


def run_shapes() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    observed = {str(pid): g["socd_kgm2"].to_numpy(float) for pid, g in layers.groupby("profile_id", sort=False)}
    observed["_frame"] = layers
    pair_rows = []
    stock_rows = []
    curve_rows = []
    figure_curves = []
    own_rows = []
    for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
        print(f"round5 shapes fold {fold}", flush=True)
        fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        curves = {}
        ids = None
        for name, spec in _specs():
            print(f"round5 shapes {name} fold {fold}", flush=True)
            lookup, _n, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
            models = []
            for seed in spec.seeds:
                model, _mx, _device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
                models.append(model)
            test_ids_ok = [str(pid) for pid in test_ids if str(pid) in lookup]
            part, _layer = _eval_fold(name, models, spec, lookup, test_ids_ok, profile_meta, observed, fold, curve_rows)
            part["model"] = name
            stock_rows.append(part)
            matrix = np.stack([lookup[pid] for pid in test_ids_ok])
            device = next(models[0].parameters()).device
            x = torch.tensor(matrix, dtype=torch.float32, device=device)
            q, z = _q_grid(models, x, 1.0)
            curves[name] = q
            ids = test_ids_ok
            tv = np.abs(np.diff(q, axis=1)).sum(axis=1)
            d2 = q[:, 2:] - 2 * q[:, 1:-1] + q[:, :-2]
            energy = (d2**2).mean(axis=1)
            peaks = _peaks(q)
            for i, pid in enumerate(ids):
                own_rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "profile_id": pid,
                        "tv": float(tv[i]),
                        "second_difference_energy": float(energy[i]),
                        "peaks_1cm": int(peaks[i]),
                    }
                )
                if i < 2:
                    for depth, value in zip(z, q[i]):
                        figure_curves.append(
                            {"model": name, "fold": int(fold), "profile_id": pid, "depth_cm": float(depth), "q": float(value)}
                        )
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for left, right in PAIRS:
            pair_rows.extend(_curve_rows(left, right, curves[left], curves[right], ids, fold))
    pairs = pd.DataFrame(pair_rows)
    stocks = pd.concat(stock_rows, ignore_index=True)
    own = pd.DataFrame(own_rows)
    pairs.to_csv(TABLES / "round5_q_pairs.csv", index=False)
    own.to_csv(TABLES / "round5_q_shape.csv", index=False)
    stocks.to_csv(TABLES / "round5_shape_stocks.csv", index=False)
    pd.DataFrame(figure_curves).to_csv(TABLES / "round5_q_examples.csv", index=False)
    summary = {"pairs": {}, "stocks": {}, "shape": {}}
    for pair, g in pairs.groupby("pair"):
        summary["pairs"][pair] = _summarize_pair(g)
    wide = {name: g for name, g in stocks.groupby("model")}
    for left, right in PAIRS:
        key = f"{left}_vs_{right}"
        summary["stocks"][key] = {
            "s30": _stock_gap(wide[left], wide[right], "p30"),
            "s100": _stock_gap(wide[left], wide[right], "p100"),
        }
    for name, g in own.groupby("model"):
        summary["shape"][name] = {
            "median_tv": float(g["tv"].median()),
            "median_second_difference_energy": float(g["second_difference_energy"].median()),
            "median_peaks_1cm": float(g["peaks_1cm"].median()),
            "p95_peaks_1cm": float(g["peaks_1cm"].quantile(0.95)),
        }
    (METRICS / "round5_shapes.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def run_sweep() -> None:
    print("round5 penalty sweep", flush=True)
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    observed = {str(pid): g["socd_kgm2"].to_numpy(float) for pid, g in layers.groupby("profile_id", sort=False)}
    observed["_frame"] = layers
    rows = []
    for weight in (0.02, 0.2, 2.0, 20.0):
        spec = ModelSpec(
            f"smooth_{weight}",
            "mcu",
            smooth=weight,
            aux=0.05,
            use_weights=True,
            seeds=(42,),
            epochs=50,
            patience=8,
            smooth_nodes=201,
        )
        started = time.perf_counter()
        y100, p100, peaks = [], [], []
        for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
            print(f"sweep {weight} fold {fold}", flush=True)
            fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
            lookup, _n, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
            model, _mx, _device = train_mcu(layers, lookup, spec, 42, sorted(fit_ids), sorted(val_ids))
            ids = [str(pid) for pid in test_ids if str(pid) in lookup]
            info = profile_meta.set_index("profile_id")
            matrix = np.stack([lookup[pid] for pid in ids])
            device = next(model.parameters()).device
            pred = predict_stock_batch(model, matrix, 100, device)
            x = torch.tensor(matrix, dtype=torch.float32, device=device)
            q, _z = _q_grid([model], x, 1.0)
            pk = _peaks(q)
            for i, pid in enumerate(ids):
                y100.append(info.loc[pid, "stock_0_100"])
                p100.append(float(pred[i]))
                peaks.append(int(pk[i]))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        y = np.asarray(y100, float)
        p = np.asarray(p100, float)
        pk = np.asarray(peaks)
        rows.append(
            {
                "smooth_weight": weight,
                "smooth_nodes": 201,
                "seeds": 1,
                "seconds": time.perf_counter() - started,
                "mae_0_100": _mae(y, p),
                "median_peaks_1cm": float(np.median(pk)),
                "p95_peaks_1cm": float(np.quantile(pk, 0.95)),
                "max_peaks_1cm": int(pk.max()),
            }
        )
        print(rows[-1], flush=True)
    (METRICS / "round5_smooth_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def run() -> None:
    run_shapes()
    run_sweep()


if __name__ == "__main__":
    run()

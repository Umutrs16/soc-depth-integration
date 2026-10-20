"""Depth-support shape statistics and a source-holdout retrain.

Does not replace archived scores. Peak counts on 0–200 cm are split into the
interval that was actually sampled and the interval below the last layer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _folds, _load, _profile_features, _profile_table, _split_fit_cal_val
from .integral_audit import _mae
from .models import ModelSpec, train_mcu
from .paths import METRICS, TABLES
from .round4_experiments import _q_grid

PAIRS = (("highfreq", "lowfreq"), ("highfreq", "cumulative"), ("lowfreq", "cumulative"))


def _specs() -> list[tuple[str, ModelSpec]]:
    common = dict(kind="mcu", smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=50, patience=8)
    return [
        ("highfreq", ModelSpec("highfreq", n_freq=8, **common)),
        ("lowfreq", ModelSpec("lowfreq", n_freq=4, **common)),
        ("cumulative", ModelSpec("cumulative", n_freq=4, cumulative_dz=2.0, **common)),
    ]


def _window_mask(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (z >= lo) & (z <= hi)


def _pair_on_mask(qa: np.ndarray, qb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff2 = (qa - qb) ** 2
    denom = mask.sum(axis=1).clip(min=1)
    l2 = np.sqrt((diff2 * mask).sum(axis=1) / denom)
    rms = np.sqrt((qa**2 * mask).sum(axis=1) / denom)
    rel = l2 / np.maximum(rms, 1e-8)
    return l2, rel


def _peaks_on_mask(q: np.ndarray, z: np.ndarray, mask_nodes: np.ndarray) -> np.ndarray:
    dq = np.diff(q, axis=1)
    peak = (dq[:, :-1] > 0) & (dq[:, 1:] <= 0)
    node_z = z[1:-1]
    keep = mask_nodes[:, 1:-1]
    return (peak & keep).sum(axis=1)


def _tv_on_mask(q: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    return (np.abs(np.diff(q, axis=1)) * edge_mask).sum(axis=1)


def run_shapes() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    z_obs = layers.groupby("profile_id")["bottom_cm"].max()
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    rows = []
    curve_rows = []
    for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
        print(f"round6 support fold {fold}", flush=True)
        fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        curves = {}
        ids = None
        z = None
        for name, spec in _specs():
            print(f"round6 support {name} fold {fold}", flush=True)
            lookup, *_rest = _profile_features(layers, spec, fit_ids)
            models = []
            for seed in spec.seeds:
                model, _mx, _device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
                models.append(model)
            ids = [str(pid) for pid in test_ids if str(pid) in lookup]
            matrix = np.stack([lookup[pid] for pid in ids])
            device = next(models[0].parameters()).device
            x = torch.tensor(matrix, dtype=torch.float32, device=device)
            q, z = _q_grid(models, x, 1.0)
            curves[name] = q
            for model in models:
                del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if fold == 0 and len(curve_rows) < 8 * 201 * 3:
                for i, pid in enumerate(ids[:8]):
                    for depth, value in zip(z, q[i]):
                        curve_rows.append(
                            {"model": name, "fold": int(fold), "profile_id": pid, "depth_cm": float(depth), "q": float(value)}
                        )
        assert ids is not None and z is not None
        observed = np.array([float(z_obs.get(pid, 0.0)) for pid in ids])
        support = z[None, :] <= observed[:, None]
        edge = z[1:][None, :] <= observed[:, None]
        windows = {
            "0_30": _window_mask(z, 0, 30),
            "30_100": _window_mask(z, 30, 100),
            "100_200": _window_mask(z, 100, 200),
            "0_200": _window_mask(z, 0, 200),
        }
        for i, pid in enumerate(ids):
            item = {"fold": int(fold), "profile_id": pid, "z_obs_cm": float(observed[i])}
            for name, q in curves.items():
                item[f"peaks_support_{name}"] = int(_peaks_on_mask(q[i : i + 1], z, support[i : i + 1])[0])
                item[f"peaks_0_200_{name}"] = int(_peaks_on_mask(q[i : i + 1], z, np.ones((1, z.size), dtype=bool))[0])
                item[f"tv_support_{name}"] = float(_tv_on_mask(q[i : i + 1], edge[i : i + 1])[0])
            for left, right in PAIRS:
                _l2, rel = _pair_on_mask(curves[left][i : i + 1], curves[right][i : i + 1], support[i : i + 1])
                item[f"rel_l2_support_{left}_vs_{right}"] = float(rel[0])
                full = np.ones((1, z.size), dtype=bool)
                _l2b, relb = _pair_on_mask(curves[left][i : i + 1], curves[right][i : i + 1], full)
                item[f"rel_l2_0_200_{left}_vs_{right}"] = float(relb[0])
            for label, mask in windows.items():
                ok = observed[i] + 1e-6 >= (30 if label == "0_30" else 100 if label == "30_100" else 200 if label == "100_200" else 0)
                item[f"window_supported_{label}"] = bool(ok)
                if ok:
                    rel = _pair_on_mask(
                        curves["highfreq"][i : i + 1], curves["lowfreq"][i : i + 1], mask[None, :]
                    )[1]
                    item[f"rel_l2_{label}"] = float(rel[0])
                    item[f"peaks_{label}_highfreq"] = int(
                        _peaks_on_mask(curves["highfreq"][i : i + 1], z, np.broadcast_to(mask, (1, z.size)))[0]
                    )
            rows.append(item)
    detail = pd.DataFrame(rows)
    detail.to_csv(TABLES / "round6_depth_support.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(TABLES / "round6_q_examples.csv", index=False)
    summary = {
        "n_profiles": int(len(detail)),
        "z_obs_cm": {
            "median": float(detail["z_obs_cm"].median()),
            "p10": float(detail["z_obs_cm"].quantile(0.10)),
            "frac_at_least_30": float((detail["z_obs_cm"] >= 30).mean()),
            "frac_at_least_100": float((detail["z_obs_cm"] >= 100).mean()),
            "frac_at_least_200": float((detail["z_obs_cm"] >= 200).mean()),
            "n_below_100": int((detail["z_obs_cm"] < 100).sum()),
            "n_100_to_200": int(((detail["z_obs_cm"] >= 100) & (detail["z_obs_cm"] < 200)).sum()),
            "n_at_least_200": int((detail["z_obs_cm"] >= 200).sum()),
        },
    }
    for name in ("highfreq", "lowfreq", "cumulative"):
        summary[name] = {
            "median_peaks_on_sampled_depth": float(detail[f"peaks_support_{name}"].median()),
            "median_peaks_0_200": float(detail[f"peaks_0_200_{name}"].median()),
            "median_tv_on_sampled_depth": float(detail[f"tv_support_{name}"].median()),
        }
    summary["high_vs_low_rel_l2_on_sampled_depth"] = {
        "median": float(detail["rel_l2_support_highfreq_vs_lowfreq"].median()),
        "p95": float(detail["rel_l2_support_highfreq_vs_lowfreq"].quantile(0.95)),
    }
    summary["high_vs_low_rel_l2_0_200"] = {
        "median": float(detail["rel_l2_0_200_highfreq_vs_lowfreq"].median()),
    }
    for label in ("0_30", "30_100", "100_200"):
        part = detail.loc[detail[f"window_supported_{label}"]]
        summary[f"supported_{label}"] = {
            "n": int(len(part)),
            "median_rel_l2_high_vs_low": float(part[f"rel_l2_{label}"].median()) if len(part) else None,
            "median_peaks_highfreq": float(part[f"peaks_{label}_highfreq"].median()) if len(part) else None,
        }
    (METRICS / "round6_depth_support.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def run_source() -> None:
    print("round6 source holdout", flush=True)
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    ref = layers.groupby("profile_id")["reference"].first().astype(str)
    liu = ref.str.contains("Liu et al., 2022", regex=False)
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    profile_meta = profile_meta.merge(liu.rename("liu"), left_on="profile_id", right_index=True, how="left")
    specs = [
        ("highfreq", ModelSpec("highfreq", "mcu", n_freq=8, smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=40, patience=6)),
        ("lowfreq", ModelSpec("lowfreq", "mcu", n_freq=4, smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=40, patience=6)),
    ]
    rows = []
    for direction, train_flag in (("train_other_test_liu", True), ("train_liu_test_other", False)):
        test_ids = profile_meta.loc[profile_meta["liu"] == train_flag, "profile_id"].astype(str)
        train_ids = profile_meta.loc[profile_meta["liu"] != train_flag, "profile_id"].astype(str)
        fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold=11 if train_flag else 12)
        info = profile_meta.set_index("profile_id")
        scored = [pid for pid in test_ids if np.isfinite(info.loc[pid, "stock_0_100"])]
        for name, spec in specs:
            print(f"round6 source {direction} {name}", flush=True)
            lookup, *_rest = _profile_features(layers, spec, fit_ids)
            preds = []
            for seed in spec.seeds:
                model, _mx, device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
                ids = [pid for pid in scored if pid in lookup]
                matrix = np.stack([lookup[pid] for pid in ids])
                from .models import predict_stock_batch

                pred = predict_stock_batch(model, matrix, 100, device)
                preds.append(pd.Series(pred, index=ids))
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            stacked = pd.concat(preds, axis=1)
            mean = stacked.mean(axis=1)
            y = info.loc[mean.index, "stock_0_100"].to_numpy(float)
            rows.append(
                {
                    "direction": direction,
                    "model": name,
                    "n_train_profiles": int(train_ids.nunique()),
                    "n_test_profiles": int(test_ids.nunique()),
                    "n_test_complete_0_100": int(len(mean)),
                    "mae_0_100": _mae(y, mean.to_numpy(float)),
                    "seeds": 3,
                    "label": "source holdout retrain, not the archived spatial scores",
                }
            )
            print(rows[-1], flush=True)
    (METRICS / "round6_source_holdout.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    run_shapes()
    run_source()


if __name__ == "__main__":
    main()

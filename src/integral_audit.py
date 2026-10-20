"""Quadrature audit of a same-code refit of the primary depth-integrated model.

The archived cross-validation did not save network weights. This script refits
the primary specification on the same 2° folds and seeds, then integrates q(z)
at several Gauss–Legendre orders. It does not replace the archived scores.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad

from .experiment import _folds, _load, _profile_features, _profile_table, _split_fit_cal_val
from .models import SPECS, train_mcu
from .paths import METRICS, TABLES

ORDERS = (8, 16, 32, 64)
CUTS = (0.0, 10.0, 20.0, 30.0, 60.0, 100.0)


def _integrate(model, h: torch.Tensor, top: torch.Tensor, bottom: torch.Tensor, n: int) -> torch.Tensor:
    nodes_np, weights_np = leggauss(n)
    nodes = torch.tensor(nodes_np, dtype=torch.float32, device=h.device)
    weights = torch.tensor(weights_np, dtype=torch.float32, device=h.device)
    span = (bottom - top).clamp_min(0)
    z = 0.5 * span.unsqueeze(-1) * nodes + 0.5 * (top + bottom).unsqueeze(-1)
    h_exp = h[:, None, :].expand(-1, n, -1)
    q = model.q_at(h_exp.reshape(-1, h.shape[-1]), z.reshape(-1)).reshape(h.shape[0], n)
    w = 0.5 * span.unsqueeze(-1) * weights
    return (q * w).sum(dim=-1)


def _ensemble_stock(models, x: torch.Tensor, top, bottom, n: int) -> np.ndarray:
    if np.isscalar(top):
        top_t = torch.full((x.shape[0],), float(top), device=x.device)
        bot_t = torch.full((x.shape[0],), float(bottom), device=x.device)
    else:
        top_t = torch.tensor(np.asarray(top), dtype=torch.float32, device=x.device)
        bot_t = torch.tensor(np.asarray(bottom), dtype=torch.float32, device=x.device)
    vals = []
    with torch.no_grad():
        for model in models:
            h = model.encode(x)
            vals.append(_integrate(model, h, top_t, bot_t, n))
    return torch.stack(vals, dim=0).mean(dim=0).detach().cpu().numpy()


def _summarize(delta: np.ndarray, base: np.ndarray) -> dict:
    ok = np.isfinite(delta)
    rel = np.abs(delta[ok]) / np.maximum(np.abs(base[ok]), 1e-6)
    abs_delta = np.abs(delta[ok])
    return {
        "n": int(ok.sum()),
        "median_abs": float(np.median(abs_delta)),
        "p95_abs": float(np.quantile(abs_delta, 0.95)),
        "max_abs": float(np.max(abs_delta)),
        "median_rel": float(np.median(rel)),
        "p95_rel": float(np.quantile(rel, 0.95)),
        "max_rel": float(np.max(rel)),
    }


def _mae(y, p) -> float | None:
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() == 0:
        return None
    return float(np.mean(np.abs(p[ok] - y[ok])))


def run() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, profiles = _load()
    profile_meta = _profile_table(layers, profiles)
    folds = _folds(profile_meta, "block_id")
    spec = SPECS["mcu"]
    archived = pd.read_parquet(METRICS / "oof_stocks.parquet")
    archived = archived.loc[(archived["model"] == "mcu") & (archived["split"] == "spatial_block")].copy()
    archived["profile_id"] = archived["profile_id"].astype(str)
    records = []
    curve_records = []
    adaptive_records = []
    rng = np.random.default_rng(42)
    for fold, train_ids, test_ids in folds:
        print(f"integral audit fold {fold}", flush=True)
        fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
        models = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for seed in spec.seeds:
            model, _max_layers, device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
            models.append(model)
        ids = [str(pid) for pid in test_ids if str(pid) in lookup]
        matrix = np.stack([lookup[pid] for pid in ids])
        fold_rows = {"profile_id": ids, "fold": fold}
        for n in ORDERS:
            s30 = []
            s100 = []
            s30_100 = []
            parts = []
            for start in range(0, len(ids), 512):
                x = torch.tensor(matrix[start : start + 512], dtype=torch.float32, device=device)
                s30.append(_ensemble_stock(models, x, 0, 30, n))
                s100.append(_ensemble_stock(models, x, 0, 100, n))
                s30_100.append(_ensemble_stock(models, x, 30, 100, n))
                piece = np.zeros(x.shape[0], dtype=float)
                for a, b in zip(CUTS[:-1], CUTS[1:]):
                    piece += _ensemble_stock(models, x, a, b, n)
                parts.append(piece)
            fold_rows[f"s30_{n}"] = np.concatenate(s30)
            fold_rows[f"s100_{n}"] = np.concatenate(s100)
            fold_rows[f"s30_100_{n}"] = np.concatenate(s30_100)
            fold_rows[f"pieces_{n}"] = np.concatenate(parts)
        cuts = rng.uniform(10.0, 90.0, size=len(ids))
        random_sum = []
        for start in range(0, len(ids), 512):
            x = torch.tensor(matrix[start : start + 512], dtype=torch.float32, device=device)
            cut = cuts[start : start + x.shape[0]]
            left = _ensemble_stock(models, x, np.zeros(len(cut)), cut, 8)
            right = _ensemble_stock(models, x, cut, np.full(len(cut), 100.0), 8)
            random_sum.append(left + right)
        fold_rows["random_cut"] = cuts
        fold_rows["random_sum_8"] = np.concatenate(random_sum)
        fold_rows["random_whole_8"] = fold_rows["s100_8"]
        peaks = []
        z = torch.linspace(0, 200, 101, device=device)
        for start in range(0, len(ids), 256):
            x = torch.tensor(matrix[start : start + 256], dtype=torch.float32, device=device)
            qs = []
            with torch.no_grad():
                for model in models:
                    h = model.encode(x)
                    h_exp = h[:, None, :].expand(-1, z.numel(), -1)
                    q = model.q_at(h_exp.reshape(-1, h.shape[-1]), z.repeat(h.shape[0])).reshape(h.shape[0], -1)
                    qs.append(q)
            q_mean = torch.stack(qs, dim=0).mean(dim=0).detach().cpu().numpy()
            dq = np.diff(q_mean, axis=1)
            peak = ((dq[:, :-1] > 0) & (dq[:, 1:] <= 0)).sum(axis=1)
            peaks.append(peak)
            if start == 0:
                example_ids = set(ids[:3])
            else:
                example_ids = set()
            for local, pid in enumerate(ids[start : start + 256]):
                if pid in example_ids:
                    for depth, value in zip(z.detach().cpu().numpy(), q_mean[local]):
                        curve_records.append({"profile_id": pid, "fold": fold, "depth_cm": float(depth), "q": float(value)})
        fold_rows["n_peaks"] = np.concatenate(peaks)
        sample_ix = rng.choice(len(ids), size=min(8, len(ids)), replace=False)
        for ix in sample_ix:
            pid = ids[int(ix)]
            x = torch.tensor(matrix[int(ix) : int(ix) + 1], dtype=torch.float32, device=device)
            with torch.no_grad():
                encoded = [model.encode(x) for model in models]

                def q_scalar(depth: float) -> float:
                    zz = torch.tensor([depth], dtype=torch.float32, device=device)
                    vals = [float(model.q_at(h, zz).detach().cpu()) for model, h in zip(models, encoded)]
                    return float(np.mean(vals))

            adaptive, err = quad(q_scalar, 0.0, 100.0, epsabs=1e-3, limit=200)
            adaptive_records.append(
                {
                    "profile_id": pid,
                    "fold": fold,
                    "adaptive_0_100": float(adaptive),
                    "adaptive_err": float(err),
                    "gl64_0_100": float(fold_rows["s100_64"][int(ix)]),
                    "gl8_0_100": float(fold_rows["s100_8"][int(ix)]),
                }
            )
        frame = pd.DataFrame(fold_rows)
        records.append(frame)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    detail = pd.concat(records, ignore_index=True)
    detail = detail.merge(archived[["profile_id", "fold", "y30", "y100", "p30", "p100"]], on=["profile_id", "fold"], how="left")
    TABLES.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "integral_additivity.csv", index=False)
    if curve_records:
        pd.DataFrame(curve_records).to_csv(TABLES / "q_curves_examples.csv", index=False)
    summary = {"orders": {}, "refit_versus_archived_oof": {}, "adaptive_sample": {}}
    for n in ORDERS:
        delta = detail[f"s100_{n}"] - detail[f"s30_{n}"] - detail[f"s30_100_{n}"]
        piece = detail[f"pieces_{n}"] - detail[f"s100_{n}"]
        summary["orders"][str(n)] = {
            "additivity_0_30_100": _summarize(delta.to_numpy(), detail[f"s100_{n}"].to_numpy()),
            "pieces_0_10_20_30_60_100_minus_whole": _summarize(piece.to_numpy(), detail[f"s100_{n}"].to_numpy()),
            "mae_0_100_against_observed": _mae(detail["y100"].to_numpy(), detail[f"s100_{n}"].to_numpy()),
            "mae_0_30_against_observed": _mae(detail["y30"].to_numpy(), detail[f"s30_{n}"].to_numpy()),
        }
    random_delta = detail["random_whole_8"] - detail["random_sum_8"]
    summary["random_cut_8_point"] = _summarize(random_delta.to_numpy(), detail["random_whole_8"].to_numpy())
    summary["q_local_maxima_0_to_200_cm"] = {
        "median": float(np.median(detail["n_peaks"])),
        "p95": float(np.quantile(detail["n_peaks"], 0.95)),
        "max": int(np.max(detail["n_peaks"])),
    }
    agree = (detail["s100_8"] - detail["p100"]).abs()
    ok = agree.notna()
    summary["refit_versus_archived_oof"] = {
        "n": int(ok.sum()),
        "median_abs_0_100": float(agree[ok].median()) if ok.any() else None,
        "p95_abs_0_100": float(agree[ok].quantile(0.95)) if ok.any() else None,
        "max_abs_0_100": float(agree[ok].max()) if ok.any() else None,
        "correlation_0_100": float(detail.loc[ok, "s100_8"].corr(detail.loc[ok, "p100"])) if ok.sum() > 2 else None,
    }
    if adaptive_records:
        ad = pd.DataFrame(adaptive_records)
        ad.to_csv(TABLES / "integral_adaptive_sample.csv", index=False)
        diff64 = (ad["adaptive_0_100"] - ad["gl64_0_100"]).abs()
        diff8 = (ad["adaptive_0_100"] - ad["gl8_0_100"]).abs()
        summary["adaptive_sample"] = {
            "n": int(len(ad)),
            "seed": "three-seed mean, same q as the 64-point integral",
            "median_abs_vs_64": float(diff64.median()),
            "max_abs_vs_64": float(diff64.max()),
            "median_abs_vs_8": float(diff8.median()),
            "max_abs_vs_8": float(diff8.max()),
        }
    summary["note"] = (
        "Same folds, seeds, and loss as the primary model. CUDA replay is not bit-identical, "
        "so refit-versus-archived reports whether this run reproduced the saved 8-point stocks."
    )
    (METRICS / "integral_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

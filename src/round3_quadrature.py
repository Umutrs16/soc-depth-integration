"""Higher-order quadrature and fine-grid q(z) for the archived model.

Refits the same primary specification. Does not replace scored predictions
and does not train a new depth basis.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _folds, _load, _profile_features, _profile_table, _split_fit_cal_val
from .integral_audit import _ensemble_stock, _mae, _summarize
from .models import SPECS, train_mcu
from .paths import METRICS, TABLES

ORDERS = (128, 256, 512)
EXAMPLE_IDS = {"3878", "3880", "3929"}


def _peaks(q: np.ndarray) -> np.ndarray:
    dq = np.diff(q, axis=1)
    return ((dq[:, :-1] > 0) & (dq[:, 1:] <= 0)).sum(axis=1)


def _q_grid(models, x: torch.Tensor, step: float) -> np.ndarray:
    z = torch.arange(0.0, 200.0 + step * 0.5, step, device=x.device)
    parts = []
    with torch.no_grad():
        for model in models:
            h = model.encode(x)
            h_exp = h[:, None, :].expand(-1, z.numel(), -1)
            q = model.q_at(h_exp.reshape(-1, h.shape[-1]), z.repeat(h.shape[0])).reshape(h.shape[0], -1)
            parts.append(q)
    return torch.stack(parts, dim=0).mean(dim=0).detach().cpu().numpy(), z.detach().cpu().numpy()


def run() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, _profiles = _load()
    profile_meta = _profile_table(layers, _profiles)
    spec = SPECS["mcu"]
    archived = pd.read_parquet(METRICS / "oof_stocks.parquet")
    archived = archived.loc[(archived["model"] == "mcu") & (archived["split"] == "spatial_block"), ["profile_id", "fold", "y100", "p100"]]
    archived["profile_id"] = archived["profile_id"].astype(str)
    records = []
    curves = []
    for fold, train_ids, test_ids in _folds(profile_meta, "block_id"):
        print(f"round3 quadrature fold {fold}", flush=True)
        fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
        lookup, _names, _med, _mu, _sd = _profile_features(layers, spec, fit_ids)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        models = []
        for seed in spec.seeds:
            model, _max_layers, device = train_mcu(layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
            models.append(model)
        ids = [str(pid) for pid in test_ids if str(pid) in lookup]
        matrix = np.stack([lookup[pid] for pid in ids])
        row = {"profile_id": ids, "fold": np.full(len(ids), fold)}
        peaks_fine = []
        peaks_coarse = []
        for start in range(0, len(ids), 128):
            x = torch.tensor(matrix[start : start + 128], dtype=torch.float32, device=device)
            batch_ids = ids[start : start + x.shape[0]]
            for n in ORDERS:
                s30 = _ensemble_stock(models, x, 0, 30, n)
                s70 = _ensemble_stock(models, x, 30, 100, n)
                s100 = _ensemble_stock(models, x, 0, 100, n)
                row.setdefault(f"s30_{n}", [])
                row.setdefault(f"s30_100_{n}", [])
                row.setdefault(f"s100_{n}", [])
                row[f"s30_{n}"].append(s30)
                row[f"s30_100_{n}"].append(s70)
                row[f"s100_{n}"].append(s100)
            q_fine, z_fine = _q_grid(models, x, 0.25)
            q_coarse, _z_coarse = _q_grid(models, x, 2.0)
            peaks_fine.append(_peaks(q_fine))
            peaks_coarse.append(_peaks(q_coarse))
            for local, pid in enumerate(batch_ids):
                if pid in EXAMPLE_IDS:
                    for depth, value in zip(z_fine, q_fine[local]):
                        curves.append({"profile_id": pid, "fold": int(fold), "depth_cm": float(depth), "q": float(value), "step_cm": 0.25})
        for n in ORDERS:
            row[f"s30_{n}"] = np.concatenate(row[f"s30_{n}"])
            row[f"s30_100_{n}"] = np.concatenate(row[f"s30_100_{n}"])
            row[f"s100_{n}"] = np.concatenate(row[f"s100_{n}"])
        row["peaks_0_25cm"] = np.concatenate(peaks_fine)
        row["peaks_2cm"] = np.concatenate(peaks_coarse)
        records.append(pd.DataFrame(row))
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()
    detail = pd.concat(records, ignore_index=True)
    detail = detail.merge(archived, on=["profile_id", "fold"], how="left")
    TABLES.mkdir(parents=True, exist_ok=True)
    detail.to_csv(TABLES / "integral_high_order.csv", index=False)
    if curves:
        pd.DataFrame(curves).to_csv(TABLES / "q_curves_0p25cm.csv", index=False)
    summary = {"orders": {}, "peaks": {}, "refit_versus_archived_8point_proxy": {}}
    for n in ORDERS:
        delta = detail[f"s100_{n}"] - detail[f"s30_{n}"] - detail[f"s30_100_{n}"]
        summary["orders"][str(n)] = {
            "additivity_0_30_100": _summarize(delta.to_numpy(), detail[f"s100_{n}"].to_numpy()),
            "mae_0_100_against_observed": _mae(detail["y100"].to_numpy(), detail[f"s100_{n}"].to_numpy()),
        }
    for name, col in (("2cm", "peaks_2cm"), ("0.25cm", "peaks_0_25cm")):
        values = detail[col].to_numpy()
        summary["peaks"][name] = {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": int(np.max(values)),
        }
    agree = (detail["s100_128"] - detail["p100"]).abs()
    summary["note"] = (
        "Orders 128, 256 and 512 integrate the same refit q used for the archived eight-point scores. "
        "They are not a retrained model. Peak counts at 2 cm can alias the 3.125 cm Fourier period; 0.25 cm does not."
    )
    summary["archived_match_not_for_128"] = {
        "median_abs_128_minus_archived_p100": float(agree.median()),
    }
    (METRICS / "round3_quadrature.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()

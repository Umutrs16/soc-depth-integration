"""Source-label retrains. Separate from the archived scores and from round 6.

Rules:
- modal: Liu et al. (2022) is the modal layer reference, same rule as the profile table.
- exclude_mixed: profiles with more than one reference string are removed, then the modal rule is applied.
One seed, the same epoch budget as the round-6 source holdout. Not a replacement for that three-seed result.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from .experiment import _load, _profile_features, _profile_table, _split_fit_cal_val
from .models import ModelSpec, predict_stock_batch, train_mcu
from .paths import METRICS


def _mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def _modal(series: pd.Series) -> str:
    modes = series.astype(str).mode()
    return str(modes.iloc[0]) if len(modes) else ""


def run() -> None:
    torch.backends.cudnn.deterministic = True
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    nunique = layers.groupby("profile_id")["reference"].nunique()
    mixed = set(nunique[nunique > 1].index.astype(str))
    modal = layers.groupby("profile_id")["reference"].agg(_modal)
    specs = [
        ("highfreq", ModelSpec("highfreq", "mcu", n_freq=8, smooth=0.02, aux=0.05, use_weights=True, seeds=(42,), epochs=40, patience=6)),
        ("lowfreq", ModelSpec("lowfreq", "mcu", n_freq=4, smooth=0.02, aux=0.05, use_weights=True, seeds=(42,), epochs=40, patience=6)),
    ]
    rows = []
    for rule in ("modal", "exclude_mixed"):
        use_layers = layers.loc[~layers["profile_id"].isin(mixed)].copy() if rule == "exclude_mixed" else layers
        profile_meta = _profile_table(use_layers, profiles)
        profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
        profile_meta = profile_meta.merge(modal.rename("modal_reference"), left_on="profile_id", right_index=True, how="left")
        profile_meta["liu"] = profile_meta["modal_reference"].astype(str).str.contains("Liu et al., 2022", regex=False)
        for direction, test_is_liu in (("train_other_test_liu", True), ("train_liu_test_other", False)):
            test_ids = profile_meta.loc[profile_meta["liu"] == test_is_liu, "profile_id"].astype(str)
            train_ids = profile_meta.loc[profile_meta["liu"] != test_is_liu, "profile_id"].astype(str)
            fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold=11 if test_is_liu else 12)
            info = profile_meta.set_index("profile_id")
            scored = [pid for pid in test_ids if np.isfinite(info.loc[pid, "stock_0_100"])]
            for name, spec in specs:
                print(f"round9 source {rule} {direction} {name}", flush=True)
                lookup = _profile_features(use_layers, spec, fit_ids)[0]
                preds = []
                for seed in spec.seeds:
                    model, _mx, device = train_mcu(use_layers, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
                    ids = [pid for pid in scored if pid in lookup]
                    matrix = np.stack([lookup[pid] for pid in ids])
                    pred = predict_stock_batch(model, matrix, 100, device)
                    preds.append(pd.Series(pred, index=ids))
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                mean = pd.concat(preds, axis=1).mean(axis=1)
                y = info.loc[mean.index, "stock_0_100"].to_numpy(float)
                rows.append(
                    {
                        "rule": rule,
                        "direction": direction,
                        "model": name,
                        "n_train_profiles": int(train_ids.nunique()),
                        "n_test_profiles": int(test_ids.nunique()),
                        "n_test_complete_0_100": int(len(mean)),
                        "mae_0_100": _mae(y, mean.to_numpy(float)),
                        "seeds": list(spec.seeds),
                        "epochs": spec.epochs,
                        "label": "one-seed source retrain; not the round-6 three-seed score and not the archived spatial score",
                    }
                )
                print(rows[-1], flush=True)
    (METRICS / "round9_source_retrain.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run()

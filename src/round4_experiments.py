"""Fourth-round retrains on the archived 2 degree folds.

Does not replace archived out-of-fold scores. Compares depth representations
that were trained, not only post-hoc integrals of the original network.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from pyproj import Geod

from .experiment import _folds, _load, _profile_features, _profile_table, _split_fit_cal_val
from .integral_audit import _ensemble_stock, _mae, _summarize
from .models import MCUSoilNet, ModelSpec, predict_layers, predict_stock_batch, train_mcu
from .paths import METRICS, TABLES

GEOD = Geod(ellps="WGS84")
EXAMPLE_IDS = {"3878", "3880", "3929"}
PRIMARY = ModelSpec(
    "primary_refit",
    "mcu",
    smooth=0.02,
    aux=0.05,
    use_weights=True,
    seeds=(42, 43, 44),
    epochs=50,
    patience=8,
)


def _variants() -> list[tuple[str, ModelSpec, str]]:
    common = dict(kind="mcu", smooth=0.02, aux=0.05, use_weights=True, seeds=(42, 43, 44), epochs=50, patience=8)
    return [
        ("lowfreq", ModelSpec("lowfreq", n_freq=4, **common), "full"),
        ("fine_smooth", ModelSpec("fine_smooth", smooth_nodes=201, **common), "full"),
        ("cumulative", ModelSpec("cumulative", n_freq=4, cumulative_dz=2.0, **common), "full"),
        ("q32", ModelSpec("q32", gauss_n=32, **common), "full"),
        ("q64", ModelSpec("q64", gauss_n=64, **common), "full"),
        ("dedup_drop", ModelSpec("dedup_drop", **common), "drop"),
        ("dedup_invweight", ModelSpec("dedup_invweight", **common), "inv"),
        ("dedup_cross", ModelSpec("dedup_cross", **common), "cross"),
    ]


def _layer_keys(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(
            zip(
                frame["Latitude"].round(4),
                frame["Longitude"].round(4),
                frame["top_cm"].round(1),
                frame["bottom_cm"].round(1),
            )
        ),
        index=frame.index,
    )


def _training_frame(layers: pd.DataFrame, fit_ids: set[str], val_ids: set[str], mode: str) -> pd.DataFrame:
    if mode == "full":
        return layers
    work = layers.copy()
    train_ids = {str(pid) for pid in fit_ids | val_ids}
    mask = work["profile_id"].astype(str).isin(train_ids)
    keys = _layer_keys(work)
    if mode == "drop":
        subset = work.loc[mask].copy()
        subset["_key"] = keys.loc[subset.index].to_numpy()
        keep = subset.sort_values("sample_id").drop_duplicates("_key", keep="first").index
        return work.drop(index=subset.index.difference(keep))
    if mode == "inv":
        counts = keys.loc[mask].map(keys.loc[mask].value_counts())
        work.loc[mask, "train_weight"] = work.loc[mask, "train_weight"].to_numpy(float) / counts.to_numpy(float)
        return work
    if mode == "cross":
        owners = work.groupby(keys, sort=False)["profile_id"].nunique()
        bad = set(owners[owners > 1].index)
        drop = work.index[mask & keys.isin(bad)]
        return work.drop(index=drop)
    raise KeyError(mode)


def _peaks(q: np.ndarray) -> np.ndarray:
    dq = np.diff(q, axis=1)
    return ((dq[:, :-1] > 0) & (dq[:, 1:] <= 0)).sum(axis=1)


def _q_grid(models, x: torch.Tensor, step: float) -> tuple[np.ndarray, np.ndarray]:
    z = torch.arange(0.0, 200.0 + step * 0.5, step, device=x.device)
    parts = []
    with torch.no_grad():
        for model in models:
            h = model.encode(x)
            h_exp = h[:, None, :].expand(-1, z.numel(), -1)
            q = model.q_at(h_exp.reshape(-1, h.shape[-1]), z.repeat(h.shape[0])).reshape(h.shape[0], -1)
            parts.append(q)
    return torch.stack(parts, dim=0).mean(dim=0).detach().cpu().numpy(), z.detach().cpu().numpy()


def _stock(models, x: torch.Tensor, depth: float) -> np.ndarray:
    vals = [predict_stock_batch(model, x.detach().cpu().numpy(), depth, x.device) for model in models]
    return np.mean(np.stack(vals, axis=0), axis=0)


def _summarize_variant(name: str, detail: pd.DataFrame, seconds: float, spec: ModelSpec) -> dict:
    y = detail["y100"].to_numpy(float)
    out = {
        "model": name,
        "seconds": seconds,
        "n_freq": spec.n_freq,
        "gauss_n": spec.gauss_n,
        "smooth_nodes": spec.smooth_nodes,
        "smooth": spec.smooth,
        "cumulative_dz": spec.cumulative_dz,
        "layer_mae": _mae(detail["y_layer"].to_numpy(float), detail["p_layer"].to_numpy(float)),
        "mae_0_30": _mae(detail["y30"].to_numpy(float), detail["p30"].to_numpy(float)),
        "mae_0_100_scored": _mae(y, detail["p100"].to_numpy(float)),
        "mae_0_100_256": _mae(y, detail["p100_256"].to_numpy(float)),
        "additivity_scored": _summarize(detail["gap"].to_numpy(float), detail["p100"].to_numpy(float)),
        "additivity_256": _summarize(detail["gap256"].to_numpy(float), detail["p100_256"].to_numpy(float)),
        "peaks_0_25cm": {
            "n": int(detail["peaks"].notna().sum()),
            "median": float(detail["peaks"].median()),
            "p95": float(detail["peaks"].quantile(0.95)),
            "max": int(detail["peaks"].max()),
        },
        "seed_mae_0_100": {
            "seed42": _mae(y, detail["p100_s42"].to_numpy(float)),
            "seed43": _mae(y, detail["p100_s43"].to_numpy(float)),
            "seed44": _mae(y, detail["p100_s44"].to_numpy(float)),
        },
    }
    return out


def _exact_geodesic(profile_meta: pd.DataFrame, folds) -> None:
    rows = []
    for fold, train_ids, test_ids in folds:
        print(f"exact geodesic fold {fold}", flush=True)
        test = profile_meta.loc[profile_meta["profile_id"].isin(test_ids), ["profile_id", "longitude", "latitude"]]
        train = profile_meta.loc[profile_meta["profile_id"].isin(train_ids), ["longitude", "latitude"]]
        test_xy = test[["longitude", "latitude"]].to_numpy(float)
        train_xy = train[["longitude", "latitude"]].to_numpy(float)
        best = np.full(len(test_xy), np.inf)
        for start in range(0, len(test_xy), 40):
            stop = min(len(test_xy), start + 40)
            tlon = test_xy[start:stop, 0]
            tlat = test_xy[start:stop, 1]
            local = np.full(stop - start, np.inf)
            n_test = stop - start
            for r0 in range(0, len(train_xy), 2000):
                r1 = min(len(train_xy), r0 + 2000)
                width = r1 - r0
                _az1, _az2, metres = GEOD.inv(
                    np.repeat(tlon, width),
                    np.repeat(tlat, width),
                    np.tile(train_xy[r0:r1, 0], n_test),
                    np.tile(train_xy[r0:r1, 1], n_test),
                )
                local = np.minimum(local, metres.reshape(n_test, width).min(axis=1) / 1000.0)
            best[start:stop] = local
        part = test[["profile_id"]].copy()
        part["profile_id"] = part["profile_id"].astype(str)
        part["fold"] = fold
        part["geodesic_exact_km"] = best
        rows.append(part)
    detail = pd.concat(rows, ignore_index=True)
    saved = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
    saved["profile_id"] = saved["profile_id"].astype(str)
    detail = detail.merge(saved, on=["profile_id", "fold"], how="left")
    detail.to_csv(TABLES / "distance_exact_geodesic.csv", index=False)
    bins = [0, 25, 50, 100, np.inf]
    labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    ok = detail["y100"].notna()
    detail.loc[ok, "bin"] = pd.cut(detail.loc[ok, "geodesic_exact_km"], bins, labels=labels, right=False).astype(str)
    old = pd.read_csv(TABLES / "distance_geodesic.csv")
    old["profile_id"] = old["profile_id"].astype(str)
    if "geodesic_km" in old.columns:
        both = detail.merge(old[["profile_id", "fold", "geodesic_km"]], on=["profile_id", "fold"], how="left")
        moved = int((both["geodesic_exact_km"] - both["geodesic_km"]).abs().gt(0).sum())
    else:
        both = detail
        moved = None
    summary = {"rows": int(len(detail)), "eight_neighbour_rows_compared": moved}
    (METRICS / "round4_geodesic.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("exact geodesic written", summary, flush=True)


def _eval_fold(name, models, spec, lookup, ids, meta, observed, fold, curves):
    matrix = np.stack([lookup[pid] for pid in ids])
    device = next(models[0].parameters()).device
    x = torch.tensor(matrix, dtype=torch.float32, device=device)
    p30 = _stock(models, x, 30)
    p100 = _stock(models, x, 100)
    p30_100 = []
    # scored gap uses the training integral, via separate calls
    s30 = p30
    s100 = p100
    # 30-100 through the same integrate
    vals = []
    with torch.no_grad():
        for model in models:
            h = model.encode(x)
            top = torch.full((x.shape[0], 1), 30.0, device=device)
            bot = torch.full((x.shape[0], 1), 100.0, device=device)
            vals.append(model.integrate(h, top, bot).detach().cpu().numpy()[:, 0])
    s70 = np.mean(np.stack(vals, axis=0), axis=0)
    ref30 = _ensemble_stock(models, x, 0, 30, 256)
    ref70 = _ensemble_stock(models, x, 30, 100, 256)
    ref100 = _ensemble_stock(models, x, 0, 100, 256)
    seed_stock = {}
    for model, seed in zip(models, spec.seeds):
        seed_stock[seed] = predict_stock_batch(model, matrix, 100, device)
    q, z = _q_grid(models, x, 0.25)
    peaks = _peaks(q)
    layer_obs, layer_pred = [], []
    test_frame = observed["_frame"]
    test_frame = test_frame.loc[test_frame["profile_id"].astype(str).isin(ids)]
    per_model = [predict_layers(model, test_frame, lookup, ids, 1, device) for model in models]
    for pid in ids:
        y_obs = observed[pid]
        predicted = np.mean([item[pid] for item in per_model], axis=0)
        if len(y_obs) != len(predicted):
            continue
        layer_obs.append(y_obs)
        layer_pred.append(predicted)
    info = meta.set_index("profile_id")
    rows = []
    y_layer = np.concatenate(layer_obs) if layer_obs else np.array([])
    p_layer = np.concatenate(layer_pred) if layer_pred else np.array([])
    for i, pid in enumerate(ids):
        if pid in EXAMPLE_IDS:
            for depth, value in zip(z, q[i]):
                curves.append({"model": name, "fold": fold, "profile_id": pid, "depth_cm": float(depth), "q": float(value)})
        rows.append(
            {
                "model": name,
                "fold": fold,
                "profile_id": pid,
                "y30": info.loc[pid, "stock_0_30"] if pid in info.index else np.nan,
                "y100": info.loc[pid, "stock_0_100"] if pid in info.index else np.nan,
                "p30": float(s30[i]),
                "p100": float(s100[i]),
                "gap": float(s100[i] - s30[i] - s70[i]),
                "p100_256": float(ref100[i]),
                "gap256": float(ref100[i] - ref30[i] - ref70[i]),
                "peaks": int(peaks[i]),
                "p100_s42": float(seed_stock[42][i]),
                "p100_s43": float(seed_stock[43][i]),
                "p100_s44": float(seed_stock[44][i]),
                "block_id": info.loc[pid, "block_id"] if pid in info.index else None,
            }
        )
    layer_frame = pd.DataFrame({"model": name, "y_layer": y_layer, "p_layer": p_layer})
    return pd.DataFrame(rows), layer_frame


def _self_check() -> None:
    torch.manual_seed(0)
    model = MCUSoilNet(4, n_freq=4, cumulative_dz=2.0)
    h = torch.zeros(2, 64)
    top = torch.tensor([[0.0], [0.0]])
    bot = torch.tensor([[100.0], [100.0]])
    mid = torch.tensor([[30.0], [30.0]])
    whole = model.integrate(h, top, bot)
    parts = model.integrate(h, top, mid) + model.integrate(h, mid, bot)
    if float((whole - parts).abs().max()) > 1e-4:
        raise RuntimeError("cumulative stock is not additive")


def run(only: tuple[str, ...] | None = None) -> None:
    _self_check()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    layers, profiles = _load()
    layers = layers.copy()
    layers["profile_id"] = layers["profile_id"].astype(str)
    profile_meta = _profile_table(layers, profiles)
    profile_meta["profile_id"] = profile_meta["profile_id"].astype(str)
    folds = list(_folds(profile_meta, "block_id"))
    observed = {
        str(pid): g["socd_kgm2"].to_numpy(float)
        for pid, g in layers.groupby("profile_id", sort=False)
    }
    observed["_frame"] = layers
    if (TABLES / "distance_exact_geodesic.csv").exists():
        print("exact geodesic already saved", flush=True)
    else:
        _exact_geodesic(profile_meta, folds)
    prior_path = METRICS / "round4_models.json"
    summaries = json.loads(prior_path.read_text(encoding="utf-8"))["models"] if prior_path.exists() else []
    details = []
    curves = []
    for name, spec, mode in _variants():
        if only is not None and name not in only:
            continue
        summaries = [item for item in summaries if item["model"] != name]
        started = time.perf_counter()
        print(f"round4 {name}", flush=True)
        fold_rows = []
        layer_rows = []
        for fold, train_ids, test_ids in folds:
            print(f"round4 {name} fold {fold}", flush=True)
            fit_ids, val_ids, _cal, _tree = _split_fit_cal_val(profile_meta, set(train_ids), fold)
            frame = _training_frame(layers, fit_ids, val_ids, mode)
            alive = set(frame["profile_id"].astype(str))
            fit_ids = {pid for pid in fit_ids if str(pid) in alive}
            val_ids = {pid for pid in val_ids if str(pid) in alive}
            lookup, _names, _med, _mu, _sd = _profile_features(frame, spec, fit_ids)
            models = []
            for seed in spec.seeds:
                model, _max_layers, _device = train_mcu(frame, lookup, spec, seed, sorted(fit_ids), sorted(val_ids))
                models.append(model)
            ids = [str(pid) for pid in test_ids if str(pid) in lookup]
            part, layer_part = _eval_fold(name, models, spec, lookup, ids, profile_meta, observed, fold, curves)
            fold_rows.append(part)
            layer_rows.append(layer_part)
            del models
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        detail = pd.concat(fold_rows, ignore_index=True)
        layers_scored = pd.concat(layer_rows, ignore_index=True)
        detail["y_layer"] = np.nan
        detail["p_layer"] = np.nan
        # layer errors are not profile-aligned; store the pooled MAE on the summary only
        detail.attrs["layer_mae"] = _mae(layers_scored["y_layer"].to_numpy(float), layers_scored["p_layer"].to_numpy(float))
        seconds = time.perf_counter() - started
        summary = _summarize_variant(name, detail.assign(y_layer=np.nan, p_layer=np.nan), seconds, spec)
        summary["layer_mae"] = detail.attrs["layer_mae"]
        summaries.append(summary)
        details.append(detail.drop(columns=["y_layer", "p_layer"]))
        payload = {"models": summaries}
        (METRICS / "round4_models.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(name, json.dumps(summary)[:500], flush=True)
    pd.concat(details, ignore_index=True).to_csv(TABLES / "round4_stocks.csv", index=False)
    pd.DataFrame(curves).to_csv(TABLES / "round4_q_examples.csv", index=False)
    print("round4 models done", flush=True)


if __name__ == "__main__":
    run(only=("lowfreq", "cumulative", "dedup_drop", "dedup_invweight", "dedup_cross"))

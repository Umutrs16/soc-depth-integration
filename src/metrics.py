"""Point and interval metrics. Inputs are paired observations only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(y, p, lo=None, hi=None):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if lo is not None:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        mask = mask & np.isfinite(lo) & np.isfinite(hi)
        return y[mask], p[mask], lo[mask], hi[mask]
    return y[mask], p[mask]


def regression_metrics(y, p) -> dict:
    y, p = _clean(y, p)
    n = int(y.size)
    if n == 0:
        return {"n": 0}
    err = p - y
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = float("nan") if sst <= 0 else float(1.0 - np.sum((y - p) ** 2) / sst)
    cov = float(np.mean((y - y.mean()) * (p - p.mean())))
    ccc_den = float(y.var() + p.var() + (y.mean() - p.mean()) ** 2)
    slope, intercept = np.polyfit(y, p, 1) if n >= 2 else (float("nan"), float("nan"))
    return {
        "n": n,
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "ccc": float(2 * cov / ccc_den) if ccc_den > 0 else float("nan"),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def interval_metrics(y, lo, hi) -> dict:
    y, _, lo, hi = _clean(y, lo, lo, hi)
    n = int(y.size)
    if n == 0:
        return {"n": 0}
    cover = (y >= lo) & (y <= hi)
    return {
        "n": n,
        "picp": float(np.mean(cover)),
        "mpiw": float(np.mean(hi - lo)),
    }


def metric_row(y, p, lo=None, hi=None) -> dict:
    row = regression_metrics(y, p)
    if lo is not None:
        row.update(interval_metrics(y, lo, hi))
    return row


def summarize_folds(frame: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    group_cols = [c for c in ("model", "target", "split") if c in frame.columns]
    for keys, g in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        rec["n_folds"] = int(len(g))
        for col in value_cols:
            if col not in g:
                continue
            rec[f"{col}_mean"] = float(g[col].mean())
            rec[f"{col}_min"] = float(g[col].min())
            rec[f"{col}_max"] = float(g[col].max())
        rows.append(rec)
    return pd.DataFrame(rows)

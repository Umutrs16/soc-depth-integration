"""Freeze the map interval rasters to the union-valid calibration widths.

The mean rasters are not changed. Lower and upper grids are the mean plus
one pair of residual quantiles. This script does not retrain the map.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio

from .paths import MANIFESTS, METRICS, RASTERS, TABLES

NEW = {
    "30": {"q05": -2.3690036763509115, "q95": 5.8839501073234945},
    "100": {"q05": -4.773122800161674, "q95": 11.079017415046689},
}


def _shift_intervals() -> dict:
    archived = json.loads((MANIFESTS / "map_calibration.json").read_text(encoding="utf-8"))
    report = {}
    for key in ("30", "100"):
        mean_path = RASTERS / f"socd_0_{key}_mean.tif"
        lo_path = RASTERS / f"socd_0_{key}_lo.tif"
        hi_path = RASTERS / f"socd_0_{key}_hi.tif"
        with rasterio.open(mean_path) as src:
            mean = src.read(1)
            profile = src.profile.copy()
        with rasterio.open(lo_path) as src:
            lo = src.read(1)
        with rasterio.open(hi_path) as src:
            hi = src.read(1)
        land = np.isfinite(mean)
        old_q05 = float(np.median((lo - mean)[land]))
        old_q95 = float(np.median((hi - mean)[land]))
        width = (hi - lo)[land]
        if abs(old_q05 - float(archived[key]["q05"])) > 1e-3 or abs(old_q95 - float(archived[key]["q95"])) > 1e-3:
            raise SystemExit(f"raster {key} does not match the archived calibration quantiles")
        if float(np.nanmax(width) - np.nanmin(width)) > 1e-3:
            raise SystemExit(f"raster {key} interval width is not constant")
        new_lo = np.full_like(mean, np.nan)
        new_hi = np.full_like(mean, np.nan)
        new_lo[land] = mean[land] + NEW[key]["q05"]
        new_hi[land] = mean[land] + NEW[key]["q95"]
        profile.update(dtype="float32", count=1, compress="deflate")
        for path, grid in ((lo_path, new_lo), (hi_path, new_hi)):
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(grid.astype(np.float32), 1)
        report[key] = {
            "archived_q05": old_q05,
            "archived_q95": old_q95,
            "archived_width": float(np.median(width)),
            "union_valid_q05": NEW[key]["q05"],
            "union_valid_q95": NEW[key]["q95"],
            "union_valid_width": NEW[key]["q95"] - NEW[key]["q05"],
            "mean_raster_unchanged": True,
            "lower_and_upper_not_summed": True,
        }
    return report


def _label_refit() -> None:
    path = TABLES / "round12_interval_profiles.csv"
    frame = pd.read_csv(path)
    frame["run_id"] = "round12_calibration_refit"
    frame["prediction_source"] = "calibration_refit"
    frame["is_planned_contrast"] = False
    frame["replaces_table4"] = False
    frame.to_csv(path, index=False)


def run() -> None:
    report = _shift_intervals()
    _label_refit()
    payload = {
        "label": "interval rasters rewritten as mean plus union-valid residual quantiles; mean rasters and national sums were not recomputed",
        "depths": report,
        "round12_intervals": "calibration_refit only. The random-forest column is not the archived forest used in Table 4. is_planned_contrast is false for every row.",
    }
    (METRICS / "round13_versions.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    run()

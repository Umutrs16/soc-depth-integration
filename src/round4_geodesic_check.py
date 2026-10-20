import json
import numpy as np
import pandas as pd
from .paths import TABLES, METRICS

exact = pd.read_csv(TABLES / "distance_exact_geodesic.csv")
old = pd.read_csv(TABLES / "distance_geodesic.csv")
exact["profile_id"] = exact["profile_id"].astype(str)
old["profile_id"] = old["profile_id"].astype(str)
m = exact.merge(old[["profile_id", "fold", "geodesic_km"]], on=["profile_id", "fold"], how="left")
delta = (m["geodesic_exact_km"] - m["geodesic_km"]).abs()
ok = m["y100"].notna()
bins = [0, 25, 50, 100, np.inf]
labels = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
m["b1"] = pd.cut(m["geodesic_exact_km"], bins, labels=labels, right=False).astype(str)
m["b0"] = pd.cut(m["geodesic_km"], bins, labels=labels, right=False).astype(str)
out = {
    "n_exact": int(len(exact)),
    "n_old": int(len(old)),
    "n_merged": int(len(m)),
    "gt0": int(delta.gt(0).sum()),
    "max_repr": repr(float(delta.max())),
    "abs_diff_gt_1m": int((delta > 0.001).sum()),
    "median_abs_km": float(delta.median()),
    "p95_abs_km": float(delta.quantile(0.95)),
    "max_abs_km": float(delta.max()),
    "y100_changed_distance_gt_1m": int((delta[ok] > 0.001).sum()),
    "y100_changed_bin": int((m.loc[ok, "b1"] != m.loc[ok, "b0"]).sum()),
    "all_changed_bin": int((m["b1"] != m["b0"]).sum()),
}
print(json.dumps(out, indent=2))
(METRICS / "round4_geodesic_vs_eight.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

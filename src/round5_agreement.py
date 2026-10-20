import json
import numpy as np
import pandas as pd
from .paths import METRICS, TABLES

stocks = pd.read_csv(TABLES / "round4_stocks.csv")
stocks["profile_id"] = stocks["profile_id"].astype(str)
base = pd.read_csv(TABLES / "oof_spatial_stocks.csv")
base["profile_id"] = base["profile_id"].astype(str)
add = pd.read_csv(TABLES / "integral_additivity.csv")
add["profile_id"] = add["profile_id"].astype(str)
base = base.merge(add[["profile_id", "fold", "s30_8", "s30_100_8", "s100_8"]], on=["profile_id", "fold"])
low = stocks.loc[stocks["model"] == "lowfreq", ["profile_id", "fold", "p30", "p100", "gap"]].rename(
    columns={"p30": "p30_low", "p100": "p100_low", "gap": "gap_low"}
)
cum = stocks.loc[stocks["model"] == "cumulative", ["profile_id", "fold", "p30", "p100", "gap"]].rename(
    columns={"p30": "p30_cum", "p100": "p100_cum", "gap": "gap_cum"}
)
m = base.merge(low, on=["profile_id", "fold"]).merge(cum, on=["profile_id", "fold"])
m["s70_pri"] = m["s30_100_8"]
m["s70_low"] = m["p100_low"] - m["p30_low"] - m["gap_low"]
m["s70_cum"] = m["p100_cum"] - m["p30_cum"] - m["gap_cum"]
m["s30_pri"] = m["s30_8"]
m["s100_pri"] = m["s100_8"]
m["s30_low"] = m["p30_low"]
m["s100_low"] = m["p100_low"]
m["s30_cum"] = m["p30_cum"]
m["s100_cum"] = m["p100_cum"]

def stat(a, b):
    ok = np.isfinite(m[a]) & np.isfinite(m[b])
    d = (m.loc[ok, a] - m.loc[ok, b]).abs()
    return {"n": int(ok.sum()), "median_abs": float(d.median()), "p95_abs": float(d.quantile(0.95)), "corr": float(np.corrcoef(m.loc[ok, a], m.loc[ok, b])[0, 1])}

out = {}
for piece in ("s30", "s70", "s100"):
    out[piece] = {
        "high_vs_low": stat(f"{piece}_pri", f"{piece}_low"),
        "high_vs_cum": stat(f"{piece}_pri", f"{piece}_cum"),
        "low_vs_cum": stat(f"{piece}_low", f"{piece}_cum"),
    }
(METRICS / "round5_stock_agreement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))

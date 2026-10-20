"""MCU-SoilNet: mass-consistent continuous-depth SOCD mapping for China.

Re-run from this directory:

    python -m src.qc
    python -m src.gee_extract --profiles
    python -m src.gee_extract --map
    python -m src.experiment
    python -m src.mapping
    python -m src.figures

Earth Engine project: hallowed-chain-458005-g9.
Every map uses 中国标准行政区划数据 GS（2024）0650号
(province polygons plus the 境界线 layer). No other national boundary is used.
"""

from __future__ import annotations

import json

import pandas as pd

from .experiment import run as run_experiment
from .figures import make_available
from .gee_extract import download_map_rasters, extract_profiles
from .mapping import run_national_map
from .qc import build_qc


def main() -> None:
    audit = build_qc()
    (pd.Series(audit)).to_json("outputs/tables/v10_audit.json", force_ascii=False)
    extract_profiles(batch_size=250)
    download_map_rasters(degree=0.1)
    run_experiment()
    run_national_map(degree=0.1)
    make_available()
    print(json.dumps({"status": "complete", "profiles": audit["v10_unique_profiles"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

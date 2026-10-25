"""Result figures revised after review.

Geographic panels use the GS (2024) 0650 province polygons and boundary lines.
The map review number is not drawn on the figure.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
from pyproj import Transformer

from .boundaries import (
    add_north_arrow,
    add_scale_bar,
    draw_official_boundaries,
    load_province_lines,
    load_provinces,
    set_official_extent,
    use_map_font,
    xinjiang_polygon,
)
from .paths import CHINA_ALBERS, FIGURES, METRICS, PROCESSED, RASTERS, TABLES

TRANSFORMER = Transformer.from_crs(4326, CHINA_ALBERS, always_xy=True)


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _mesh(transform, data):
    height, width = data.shape
    lon = transform.c + (np.arange(width) + 0.5) * transform.a
    lat = transform.f + (np.arange(height) + 0.5) * transform.e
    lon2d, lat2d = np.meshgrid(lon, lat)
    x, y = TRANSFORMER.transform(lon2d, lat2d)
    return x, y


def gpd_geom(geom):
    import geopandas as gpd

    return gpd.GeoSeries([geom], crs=4326).to_crs(CHINA_ALBERS)


def _box(ax, x, y, w, h, text, face="#f7f7f7", size=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02", facecolor=face, edgecolor="#222222", lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size)


def _arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "#222222", "shrinkA": 0, "shrinkB": 0})


def figure_profiles() -> None:
    use_map_font()
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    depth = profiles["depth_max_cm"].to_numpy()
    fig = plt.figure(figsize=(9.4, 6.6))
    ax = fig.add_axes([0.005, 0.04, 0.58, 0.90])
    cax = fig.add_axes([0.595, 0.50, 0.012, 0.36])
    axh = fig.add_axes([0.74, 0.60, 0.24, 0.32])
    axb = fig.add_axes([0.74, 0.05, 0.24, 0.40])
    provinces = load_provinces().to_crs(CHINA_ALBERS)
    provinces.plot(ax=ax, color="#f7f7f7", edgecolor="none")
    x, y = TRANSFORMER.transform(profiles["longitude"].to_numpy(), profiles["latitude"].to_numpy())
    sc = ax.scatter(x, y, c=np.clip(depth, 0, 200), s=7, cmap="cividis", linewidths=0, alpha=0.45, vmin=0, vmax=200, zorder=3)
    draw_official_boundaries(ax, provinces=load_provinces(), linewidth=0.45)
    set_official_extent(ax)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.14 * (y1 - y0))
    add_scale_bar(ax)
    add_north_arrow(ax, x_frac=0.96, y_frac=0.90, dy_frac=0.045)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Maximum depth (cm)\ncapped at 200", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("(a)", fontsize=12, loc="left")
    order = np.sort(depth)
    axh.plot(order, np.arange(1, len(order) + 1), color="#3b4cc0", lw=1.6)
    axh.axvline(200, color="black", lw=0.8, ls="--")
    axh.set_xlabel("Maximum depth (cm)", fontsize=13, labelpad=2)
    axh.set_ylabel("Cumulative profiles", fontsize=13)
    axh.tick_params(labelsize=12)
    axh.set_title("(b)", fontsize=12, loc="left")
    coverage = TABLES / "round11_coverage_profiles.csv"
    if coverage.exists():
        flags = pd.read_csv(coverage)
        n30 = int(((flags["depth_cm"] == 30) & flags["status"].isin(["exact_boundary", "prorated"])).sum())
        n100 = int(((flags["depth_cm"] == 100) & flags["status"].isin(["exact_boundary", "prorated"])).sum())
    else:
        n30 = int(profiles["cover_0_30"].sum())
        n100 = int(profiles["cover_0_100"].sum())
    deep_path = METRICS / "round12_depth200.json"
    if deep_path.exists():
        deep = json.loads(deep_path.read_text(encoding="utf-8"))
        n200 = int(deep["union_cover_0_200"])
        deep_label = "0–200 cm cover"
    else:
        n200 = int(profiles["cover_100_200"].sum())
        deep_label = "100–200 cm, archived"
    labels = ["All profiles", "0–30 cm cover", "0–100 cm cover", deep_label]
    values = [len(profiles), n30, n100, n200]
    axb.barh(labels[::-1], values[::-1], color="#8c8c8c")
    axb.set_xlabel("Profiles", fontsize=13)
    axb.tick_params(labelsize=12)
    axb.set_title("(c)", fontsize=12, loc="left", pad=2)
    for i, val in enumerate(values[::-1]):
        axb.text(val + 40, i, str(val), va="center", fontsize=12)
    _save(fig, "fig01_profiles")


def figure_workflow() -> None:
    use_map_font()
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, 0.03, 0.78, 0.28, 0.16, "V10 table\n23,103 layers\n7,955 profile IDs", "#d9ead3")
    _box(ax, 0.36, 0.78, 0.28, 0.16, "QC\noverlap, gaps, reliability\nno coarse fragments", "#d9ead3")
    _box(ax, 0.69, 0.78, 0.28, 0.16, "Labels\n0–30: 4,948 profiles\n0–100: 3,212 profiles", "#d9ead3")
    _arrow(ax, 0.31, 0.86, 0.36, 0.86)
    _arrow(ax, 0.64, 0.86, 0.69, 0.86)
    _box(ax, 0.03, 0.54, 0.44, 0.16, "Earth Engine covariates, 2015–2022\nSRTM 1 km, ERA5-Land/CHIRPS ~4 km\nMODIS NDVI 1 km, WorldCover 2021", "#cfe2f3")
    _box(ax, 0.53, 0.54, 0.44, 0.16, "Not used as predictors\nV10 pH, texture, DEM, NDVI\nclimate and land-cover fields", "#f4cccc")
    _box(ax, 0.03, 0.30, 0.28, 0.16, "Outer test, untouched\n2° blocks, 5 folds\nno profile on both sides", "#fff2cc")
    _box(ax, 0.36, 0.30, 0.28, 0.16, "Inside training blocks\nfit / early-stop / calibration\ncalibration does not update weights", "#fff2cc")
    _box(ax, 0.69, 0.30, 0.28, 0.16, "Separate refits\nsource holdout\nXinjiang zero-shot", "#fff2cc")
    _box(ax, 0.08, 0.06, 0.38, 0.16, "Primary model is the integral\nextra penalties are ablations\npaired block bootstrap of saved errors", "#fce5cd")
    _box(ax, 0.54, 0.06, 0.38, 0.16, "Residual quantiles, not a national interval\n0.10° × 0.11° fixed-depth sum\nno thickness or gravel scenario", "#fce5cd")
    _arrow(ax, 0.47, 0.62, 0.53, 0.62)
    ax.set_axis_off()
    _save(fig, "fig02_workflow")


def figure_architecture() -> None:
    use_map_font()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.6))
    panels = [
        (
            "Eight frequencies",
            "k = 0…7\nperiod down to 3.125 cm",
            "Separate 8-point\nintegrals of q(z)",
            "Not strictly additive\nmedian gap 0.19 kg m$^{-2}$\nmedian 63 peaks",
            "#f4cccc",
        ),
        (
            "Four frequencies",
            "k = 0…3\nlongest periods only",
            "Separate 8-point\nintegrals of q(z)",
            "Approximately additive\nmedian gap 0.0009 kg m$^{-2}$\nmedian 3 peaks",
            "#fff2cc",
        ),
        (
            "Cumulative stock",
            "k = 0…3\nF(0) = 0, increments ≥ 0",
            "Layer stock is\nF(b) − F(a)",
            "Additive at float precision\nmedian gap 2×10$^{-7}$\nmedian 3 peaks",
            "#d9ead3",
        ),
    ]
    for ax, (title, depth, stock, result, color) in zip(axes, panels):
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        _box(ax, 0.12, 0.78, 0.76, 0.16, "Same covariates and encoder", "#cfe2f3", size=8)
        _box(ax, 0.12, 0.54, 0.76, 0.16, depth, "#fff2cc", size=8)
        _box(ax, 0.12, 0.30, 0.76, 0.16, stock, color, size=8)
        _box(ax, 0.12, 0.05, 0.76, 0.18, result, color, size=8)
        _arrow(ax, 0.50, 0.78, 0.50, 0.70)
        _arrow(ax, 0.50, 0.54, 0.50, 0.46)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig03_architecture")


def figure_profiles_curves() -> None:
    oof = pd.read_parquet(METRICS / "oof_layers.parquet")
    oof = oof.loc[(oof["model"] == "mcu") & (oof["split"] == "spatial_block")].copy()
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    counts = oof.groupby("profile_id").size()
    meta = profiles.loc[profiles["profile_id"].isin(counts[counts >= 4].index)].copy()
    picks = []
    for province in ("Xinjiang", "Heilongjiang", "Guangdong", "Xizang"):
        hit = meta.loc[meta["province_en"] == province]
        if len(hit):
            picks.append(hit.iloc[0]["profile_id"])
    use_map_font()
    fig, axes = plt.subplots(1, len(picks), figsize=(2.3 * len(picks), 4.6), sharey=True)
    if len(picks) == 1:
        axes = [axes]
    for ax, pid in zip(axes, picks):
        g = oof.loc[oof["profile_id"] == pid].sort_values("top_cm")
        thick = (g["bottom_cm"] - g["top_cm"]).clip(lower=0.5)
        q_obs = g["socd_kgm2"] / thick
        q_hat = g["pred"] / thick
        for top, bot, q in zip(g["top_cm"], g["bottom_cm"], q_obs):
            ax.add_patch(Rectangle((0, top), q, bot - top, facecolor="#f4a582", edgecolor="#b2182b", lw=0.6, alpha=0.9))
        for top, bot, q in zip(g["top_cm"], g["bottom_cm"], q_hat):
            ax.plot([q, q], [top, bot], color="#2166ac", lw=1.4)
        prov = profiles.loc[profiles["profile_id"] == pid, "province_en"]
        ax.set_title(str(prov.iloc[0]) if len(prov) else str(pid), fontsize=11)
        ax.set_xlabel("Layer-mean q\n(kg m$^{-2}$ cm$^{-1}$)", fontsize=11)
        ax.tick_params(labelsize=10)
    deepest = max(float(oof.loc[oof["profile_id"].isin(picks), "bottom_cm"].max()), 1.0)
    for ax in axes:
        ax.set_ylim(deepest, 0)
    axes[0].set_ylabel("Depth (cm)", fontsize=11)
    axes[0].plot([], [], color="#2166ac", lw=1.4, label="Predicted layer mean")
    axes[0].plot([], [], color="#b2182b", lw=4, label="Observed layer mean", solid_capstyle="butt")
    axes[0].legend(fontsize=10, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, "fig04_profile_curves")


def _bootstrap_rows():
    payload = json.loads((METRICS / "revision_stats.json").read_text(encoding="utf-8"))
    return payload["paired_block_bootstrap"]


def figure_model_comparison() -> None:
    corrected_folds = json.loads((METRICS / "round11_coverage.json").read_text(encoding="utf-8"))
    use_map_font()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.6), gridspec_kw={"width_ratios": [1.0, 1.35]})
    models = ["rf", "mlp_integral", "mcu"]
    colors = {"rf": "#4d4d4d", "mlp_integral": "#74add1", "mcu": "#2166ac"}
    names = {"rf": "Random forest", "mlp_integral": "Integral MLP", "mcu": "Primary integral"}
    folds = [row["fold"] for row in corrected_folds["scores"]["100"]["mcu"]["folds"]]
    for model in models:
        part = pd.DataFrame(corrected_folds["scores"]["100"][model]["folds"]).set_index("fold").loc[folds]
        axes[0].plot(folds, part["mae"], marker="o", color=colors[model], label=names[model], lw=1.2, ms=5)
    axes[0].set_xticks(folds)
    axes[0].set_xlabel("Fold index as stored (0–4)", fontsize=12)
    axes[0].set_ylabel("0–100 cm MAE (kg m$^{-2}$)", fontsize=12)
    axes[0].tick_params(labelsize=11)
    axes[0].legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    wanted = [
        ("stock_0_100", "xgb", "0–100 cm, XGBoost"),
        ("stock_0_100", "hgb", "0–100 cm, hist. boosting"),
        ("stock_0_100", "rf", "0–100 cm, random forest"),
        ("stock_0_30", "rf", "0–30 cm, random forest"),
        ("stock_0_30", "mlp_integral", "0–30 cm, integral MLP"),
        ("layer", "rf", "Layer, random forest"),
        ("stock_0_100", "mlp_integral", "0–100 cm, integral MLP"),
        ("layer", "mlp_integral", "Layer, integral MLP"),
    ]
    lookup = {(r["target"], r["baseline"]): r for r in _bootstrap_rows()}
    corrected = json.loads((METRICS / "round11_coverage.json").read_text(encoding="utf-8"))
    for depth, target in (("30", "stock_0_30"), ("100", "stock_0_100")):
        for baseline, row in corrected["scores"][depth]["paired_other_minus_primary"].items():
            lookup[(target, baseline)] = {
                "mean_baseline_minus_mcu": row["mean"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
            }
    labels, means, los, his = [], [], [], []
    for target, baseline, label in wanted:
        row = lookup[(target, baseline)]
        labels.append(label)
        means.append(row["mean_baseline_minus_mcu"])
        los.append(row["mean_baseline_minus_mcu"] - row["ci95_low"])
        his.append(row["ci95_high"] - row["mean_baseline_minus_mcu"])
    tree_pairs = json.loads((METRICS / "round4_tree_pairs.json").read_text(encoding="utf-8"))
    tuned = next(row for row in tree_pairs if row["model"] == "tuned_rf")["versus_primary_positive_means_tuned_worse"]
    labels.append("0–100 cm, tuned forest, exploratory")
    plot_mean = -tuned["mean"]
    paper_low = -tuned["ci95_high"]
    paper_high = -tuned["ci95_low"]
    means.append(plot_mean)
    los.append(plot_mean - paper_low)
    his.append(paper_high - plot_mean)
    y = np.arange(len(labels))
    planned = np.array([label == "0–100 cm, random forest" for label in labels])
    axes[1].errorbar(np.array(means)[~planned], y[~planned], xerr=[np.array(los)[~planned], np.array(his)[~planned]], fmt="o", color="#4d4d4d", ecolor="#666666", capsize=2, ms=4, label="Exploratory")
    axes[1].errorbar(np.array(means)[planned], y[planned], xerr=[np.array(los)[planned], np.array(his)[planned]], fmt="s", color="#4d4d4d", ecolor="#666666", capsize=2, ms=6, label="Planned")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=13)
    axes[1].tick_params(labelsize=11)
    axes[1].set_xlabel("Named MAE − primary MAE (kg m$^{-2}$)", fontsize=12)
    axes[1].legend(frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.55, 1.02), ncol=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24, top=0.86)
    _save(fig, "fig06_model_comparison")


def figure_ablation_calibration() -> None:
    use_map_font()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 4.4))
    lookup = {(r["target"], r["baseline"]): r for r in _bootstrap_rows()}
    keys = [
        ("stock_0_100", "mlp_integral"),
        ("stock_0_100", "mcu_equal_weight"),
        ("stock_0_100", "mcu_closure"),
        ("stock_0_100", "mcu_nors"),
        ("stock_0_30", "mlp_integral"),
        ("layer", "mlp_integral"),
    ]
    pretty = {
        ("stock_0_100", "mlp_integral"): "0–100 cm, integral MLP",
        ("stock_0_100", "mcu_equal_weight"): "0–100 cm, equal weights",
        ("stock_0_100", "mcu_closure"): "0–100 cm, product penalty",
        ("stock_0_100", "mcu_nors"): "0–100 cm, no NDVI",
        ("stock_0_30", "mlp_integral"): "0–30 cm, integral MLP",
        ("layer", "mlp_integral"): "Layer, integral MLP",
    }
    labels, means, los, his, colors = [], [], [], [], []
    for key in keys:
        row = lookup[key]
        labels.append(pretty[key])
        means.append(row["mean_baseline_minus_mcu"])
        los.append(row["mean_baseline_minus_mcu"] - row["ci95_low"])
        his.append(row["ci95_high"] - row["mean_baseline_minus_mcu"])
        excludes = row["ci95_low"] > 0 or row["ci95_high"] < 0
        colors.append(excludes)
    y = np.arange(len(labels))
    for i in range(len(labels)):
        axes[0].errorbar(
            means[i],
            y[i],
            xerr=[[los[i]], [his[i]]],
            fmt="o" if colors[i] else "o",
            markerfacecolor="#4d4d4d" if colors[i] else "none",
            markeredgecolor="#4d4d4d",
            color="#4d4d4d",
            ecolor="#4d4d4d",
            capsize=2,
            ms=6,
        )
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=11)
    axes[0].tick_params(labelsize=11)
    axes[0].set_xlabel("Other MAE − primary MAE\n(kg m$^{-2}$)", fontsize=11)
    axes[0].set_title("(a)", fontsize=12, loc="left")
    rebuilt = METRICS / "round12_intervals.json"
    if rebuilt.exists():
        payload = json.loads(rebuilt.read_text(encoding="utf-8"))
        pred = pd.DataFrame(payload["models"]["mcu"]["scores"]["predicted_decile"])
        obs = pd.DataFrame(payload["models"]["mcu"]["scores"]["observed_decile"])
        axes[1].plot(pred["decile"], pred["picp"], marker="o", color="#2166ac", label="Prediction")
        axes[1].plot(obs["decile"], obs["picp"], marker="s", color="#b2182b", label="Target, post hoc")
        axes[1].set_xticks(pred["decile"])
        axes[1].set_xticklabels([str(int(d)) for d in pred["decile"]], fontsize=12)
        axes[1].set_xlabel("Decile (n = 321). 1 is lowest.", fontsize=12)
        axes[1].set_title("(b) Coverage", fontsize=12, loc="left")
        axes[2].plot(pred["decile"], pred["mpiw"], marker="o", color="#2166ac", label="Prediction")
        axes[2].plot(obs["decile"], obs["mpiw"], marker="s", color="#b2182b", label="Target, post hoc")
        axes[2].set_xticks(pred["decile"])
        axes[2].set_xticklabels([str(int(d)) for d in pred["decile"]], fontsize=12)
        axes[2].set_xlabel("Decile (n = 321). 1 is lowest.", fontsize=12)
        axes[2].set_title("(c) Width", fontsize=12, loc="left")
    else:
        payload = json.loads((METRICS / "round6_scaled_intervals.json").read_text(encoding="utf-8"))
        dec = pd.DataFrame(payload["deciles"])
        axes[1].plot(dec["predicted_decile"], dec["picp_constant_residual"], marker="o", color="#2166ac", label="Constant-width coverage")
        axes[1].plot(dec["predicted_decile"], dec["picp_scaled_residual"], marker="s", color="#b2182b", label="Scaled-width coverage")
        axes[2].plot(dec["predicted_decile"], dec["mean_width_constant"], color="#2166ac", lw=1.0, ls="--", label="Constant width")
        axes[2].plot(dec["predicted_decile"], dec["mean_width_scaled"], color="#b2182b", lw=1.0, ls="--", label="Scaled width")
        axes[1].set_xticks(dec["predicted_decile"])
        axes[1].set_xticklabels([f"{int(d)}\nn={int(n)}" for d, n in zip(dec["predicted_decile"], dec["n"])], fontsize=8)
        axes[1].set_xlabel("Decile of the predicted 0–100 cm stock", fontsize=8)
        axes[1].set_title("(b)", fontsize=12, loc="left")
    axes[1].axhline(0.90, color="black", lw=0.8, ls=":", label="Nominal 90%")
    axes[1].set_ylim(0.45, 1.02)
    axes[1].set_ylabel("Empirical coverage", fontsize=12)
    axes[2].set_ylabel("Mean interval width (kg m$^{-2}$)", fontsize=12)
    axes[1].tick_params(labelsize=11)
    axes[2].tick_params(labelsize=11)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")
    axes[2].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    _save(fig, "fig07_ablation_calibration")


def _quantile_edges(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    edges = np.quantile(finite, [0, 0.2, 0.4, 0.6, 0.8, 0.98])
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-3
    edges[-1] = max(edges[-1], float(np.nanmax(finite)))
    return edges


def figure_national_maps() -> None:
    path30 = RASTERS / "socd_0_30_mean.tif"
    path100 = RASTERS / "socd_0_100_mean.tif"
    path_aoa = RASTERS / "aoa_mask.tif"
    if not path30.exists() or not path100.exists() or not path_aoa.exists():
        return
    use_map_font()
    with rasterio.open(path30) as src:
        data30 = src.read(1)
        transform = src.transform
    with rasterio.open(path100) as src:
        data100 = src.read(1)
    with rasterio.open(path_aoa) as src:
        aoa = src.read(1)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    cmap = ListedColormap(["#0c2c84", "#225ea8", "#7fcdbb", "#edf8b1", "#fec44f", "#d7301f"])
    x, y = _mesh(transform, data100)
    breaks = {
        "0–30 cm": (data30, [0, 2, 4, 6, 10, 16, 30]),
        "0–100 cm": (data100, [0, 4, 8, 12, 20, 30, 60]),
    }
    for ax, (title, (data, edges)) in zip(axes, breaks.items()):
        land = np.isfinite(data)
        outside = land & (aoa != 1)
        inside = np.where(land & (aoa == 1), data, np.nan)
        grey = np.where(outside, 1.0, np.nan)
        ax.pcolormesh(x, y, np.ma.masked_invalid(grey), cmap=ListedColormap(["#bdbdbd"]), shading="auto")
        norm = BoundaryNorm(edges, cmap.N)
        mesh = ax.pcolormesh(x, y, np.ma.masked_invalid(inside), cmap=cmap, norm=norm, shading="auto")
        draw_official_boundaries(ax)
        set_official_extent(ax)
        ax.set_title(title, fontsize=11, loc="left")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02, ticks=edges)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("kg m$^{-2}$", fontsize=8)
    add_scale_bar(axes[1])
    add_north_arrow(axes[1])
    fig.tight_layout()
    _save(fig, "fig05_national_socd")


def figure_xinjiang() -> None:
    stocks = pd.read_parquet(METRICS / "oof_stocks.parquet")
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    mcu = stocks.loc[(stocks["model"] == "mcu") & (stocks["split"] == "xinjiang_holdout")].copy()
    rf = stocks.loc[(stocks["model"] == "rf") & (stocks["split"] == "xinjiang_holdout"), ["profile_id", "p100"]]
    mcu = mcu.merge(rf, on="profile_id", suffixes=("", "_rf"))
    mcu = mcu.merge(profiles[["profile_id", "longitude", "latitude"]], on="profile_id", how="left")
    use_map_font()
    fig = plt.figure(figsize=(7.6, 6.5))
    ax_map = fig.add_axes([0.005, 0.12, 0.46, 0.80])
    cax = fig.add_axes([0.07, 0.045, 0.34, 0.018])
    ax_full = fig.add_axes([0.58, 0.55, 0.38, 0.38])
    ax_zoom = fig.add_axes([0.58, 0.08, 0.38, 0.36])
    xj = xinjiang_polygon()
    provinces = load_provinces().to_crs(CHINA_ALBERS)
    lines = load_province_lines().to_crs(CHINA_ALBERS)
    xj_albers = gpd_geom(xj)
    xj_albers.plot(ax=ax_map, color="#f7f7f7", edgecolor="black", linewidth=0.6, zorder=1)
    lines.plot(ax=ax_map, color="black", linewidth=0.35, zorder=2)
    xx, yy = TRANSFORMER.transform(mcu["longitude"].to_numpy(), mcu["latitude"].to_numpy())
    err = (mcu["p100"] - mcu["y100"]).to_numpy()
    finite_err = err[np.isfinite(err)]
    cap = float(np.percentile(np.abs(finite_err), 95))
    sc = ax_map.scatter(xx, yy, c=err, cmap="coolwarm", s=22, linewidths=0, vmin=-cap, vmax=cap, zorder=3)
    high_ids = {"3637", "3607", "3632", "3627", "3631", "3674", "3679"}
    extreme = mcu["profile_id"].astype(str).isin(high_ids).to_numpy()
    ax_map.scatter(np.asarray(xx)[extreme], np.asarray(yy)[extreme], s=36, facecolors="none", edgecolors="black", linewidths=0.8, zorder=4)
    minx, miny, maxx, maxy = xj_albers.total_bounds
    pad_x, pad_y = 0.03 * (maxx - minx), 0.04 * (maxy - miny)
    ax_map.set_xlim(minx - pad_x, maxx + pad_x)
    ax_map.set_ylim(miny - 3.2 * pad_y, maxy + pad_y)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("(a)", fontsize=12, loc="left")
    add_scale_bar(ax_map, length_m=400_000, label="400 km", y_frac=0.025)
    add_north_arrow(ax_map, loc="upper right")
    inset = ax_map.inset_axes([0.02, 0.58, 0.34, 0.38])
    provinces.plot(ax=inset, color="#f4f4f4", edgecolor="#4d4d4d", linewidth=0.45)
    lines.plot(ax=inset, color="black", linewidth=0.55)
    xj_albers.boundary.plot(ax=inset, color="#b2182b", linewidth=0.9)
    set_official_extent(inset, pad_frac=0.01)
    inset.set_title(" ", fontsize=1)
    cbar = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cbar.set_label("Predicted − observed (kg m$^{-2}$)", fontsize=11)
    cbar.ax.tick_params(labelsize=11)
    finite_all = err[np.isfinite(err)]
    fig.text(
        0.24,
        0.100,
        f"Residuals {finite_all.min():.1f} to {finite_all.max():.1f}; display clipped at ±{cap:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    finite = mcu.dropna(subset=["y100", "p100", "p100_rf"])
    lim = float(finite[["y100", "p100", "p100_rf"]].to_numpy().max()) * 1.04
    ax_full.scatter(finite["y100"], finite["p100_rf"], s=18, c="#bdbdbd", label="Random forest", zorder=2)
    ax_full.scatter(finite["y100"], finite["p100"], s=18, c="#2166ac", label="Eight-frequency integral", zorder=3)
    high = finite.loc[finite["profile_id"].astype(str).isin(high_ids)]
    ax_full.scatter(high["y100"], high["p100"], s=42, facecolors="none", edgecolors="black", linewidths=0.9, zorder=4)
    ax_full.plot([0, lim], [0, lim], color="black", lw=0.7)
    ax_full.set_xlim(0, lim)
    ax_full.set_ylim(0, lim)
    ax_full.set_xlabel("0–100 cm target (kg m$^{-2}$)", fontsize=11)
    ax_full.set_ylabel("Predicted (kg m$^{-2}$)", fontsize=11)
    ax_full.set_title("(b)", fontsize=12, loc="left")
    ax_full.tick_params(labelsize=9)
    ax_full.legend(frameon=False, fontsize=10)
    peak = finite.loc[finite["y100"].idxmax()]
    ax_full.annotate(
        f"{peak['y100']:.1f}",
        xy=(peak["y100"], peak["p100"]),
        xytext=(0.62, 0.18),
        textcoords="axes fraction",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "lw": 0.6, "color": "black"},
    )
    zoom = finite.loc[finite["y100"] <= 40]
    pred_hi = float(zoom[["p100", "p100_rf"]].to_numpy().max())
    ax_zoom.scatter(zoom["y100"], zoom["p100_rf"], s=28, c="#bdbdbd", label="Random forest", zorder=2)
    ax_zoom.scatter(zoom["y100"], zoom["p100"], s=28, c="#2166ac", label="Eight-frequency integral", zorder=3)
    upper = max(40.0, pred_hi) * 1.05
    ax_zoom.plot([0, 40], [0, 40], color="black", lw=0.7)
    ax_zoom.set_xlim(0, 40)
    ax_zoom.set_ylim(0, upper)
    ax_zoom.set_xlabel("0–100 cm target (kg m$^{-2}$)", fontsize=11)
    ax_zoom.set_ylabel("Predicted (kg m$^{-2}$)", fontsize=11)
    ax_zoom.set_title("(c)", fontsize=12, loc="left")
    ax_zoom.tick_params(labelsize=9)
    ax_zoom.legend(frameon=False, fontsize=10)
    _save(fig, "fig08_xinjiang")


def _draw_synthetic(part, pid, layer_edges, truths, designs, figsize, filename: str) -> None:
    use_map_font()
    fig, axes = plt.subplots(2, len(truths), figsize=figsize, sharey=True)
    if len(truths) == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for row, design in enumerate(designs):
        for col, truth in enumerate(truths):
            ax = axes[row, col]
            block = part.loc[(part["truth"] == truth) & (part["design"] == design)].sort_values("depth_cm")
            if block.empty:
                ax.set_axis_off()
                continue
            block = block.loc[block["depth_cm"] <= 100]
            known = block.loc[block["model"] == "lowfreq"]
            ax.plot(known["q_true"], known["depth_cm"], color="black", lw=2.4, label="Known q", zorder=3)
            for model, colour, label in (
                ("highfreq", "#b2182b", "k = 0…7"),
                ("lowfreq", "#2166ac", "k = 0…3"),
            ):
                one = block.loc[block["model"] == model]
                ax.plot(one["q_pred"], one["depth_cm"], color=colour, lw=1.5, ls="-" if model == "highfreq" else "--", label=label, zorder=2)
            if design == "irregular_layers":
                for edge in layer_edges:
                    if 0 < float(edge) < 100:
                        ax.axhline(float(edge), color="#737373", lw=0.8, ls="--", zorder=1)
            ax.set_ylim(100, 0)
            ax.tick_params(labelsize=12)
            low = block.loc[block["model"] == "lowfreq"]
            if not low.empty:
                z = low["depth_cm"].to_numpy(float)
                q_known = low["q_true"].to_numpy(float)
                pred = low["q_pred"].to_numpy(float)
                rel = float(np.sqrt(np.mean((pred - q_known) ** 2)) / max(np.sqrt(np.mean(q_known**2)), 1e-8))
                peak_err = abs(float(z[int(np.argmax(pred))] - z[int(np.argmax(q_known))]))
                ax.set_xlabel(f"rel. L2 {rel:.2f}\npeak error {peak_err:.0f} cm", fontsize=12)
            if row == 0:
                ax.set_title({"decay": "Decay", "bulge": "Bulge", "horizons": "Three pieces", "deep_peak": "Deep peak"}[truth], fontsize=11)
            if col == 0:
                name = "Irregular layers" if design == "irregular_layers" else "One 0–100 cm label"
                ax.set_ylabel(f"Depth (cm)\n{name}", fontsize=10)
    for col, truth in enumerate(truths):
        xmax = 0.0
        for design in designs:
            block = part.loc[(part["truth"] == truth) & (part["design"] == design) & (part["depth_cm"] <= 100)]
            if block.empty:
                continue
            xmax = max(xmax, float(block[["q_true", "q_pred"]].max().max()))
        for row in range(2):
            axes[row, col].set_xlim(0, xmax * 1.08 if xmax > 0 else 1)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], texts[:3], loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, filename)


def figure_synthetic() -> None:
    path = TABLES / "round6_synthetic_curves.csv"
    if not path.exists():
        return
    curves = pd.read_csv(path)
    curves["profile_id"] = curves["profile_id"].astype(str)
    pid = str(curves["profile_id"].iloc[0])
    part = curves.loc[(curves["profile_id"] == pid) & (curves["noise"] == 0.15) & (curves["seed"] == 42)]
    designs = ["irregular_layers", "one_slab_0_100"]
    layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
    layer_edges = layers.loc[layers["profile_id"].astype(str) == pid, "bottom_cm"]
    _draw_synthetic(part, pid, layer_edges, ["decay", "deep_peak"], designs, (7.6, 6.4), "fig14_synthetic")
    _draw_synthetic(part, pid, layer_edges, ["bulge", "horizons"], designs, (7.6, 6.4), "fig18_synthetic_rest")


def figure_linked() -> None:
    path = TABLES / "round7_linked_curves.csv"
    if not path.exists():
        return
    curves = pd.read_csv(path)
    curves["profile_id"] = curves["profile_id"].astype(str)
    pid = str(curves["profile_id"].iloc[0])
    part = curves.loc[(curves["profile_id"] == pid) & (curves["noise"] == 0.15) & (curves["seed"] == 42)]
    truths = ["decay", "deep_peak"]
    designs = ["irregular_layers", "one_slab_0_100"]
    use_map_font()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0), sharey=True)
    for row, design in enumerate(designs):
        for col, truth in enumerate(truths):
            ax = axes[row, col]
            block = part.loc[(part["truth"] == truth) & (part["design"] == design) & (part["depth_cm"] <= 100)].sort_values("depth_cm")
            if block.empty:
                ax.set_axis_off()
                continue
            known = block.loc[block["model"] == "lowfreq"]
            ax.plot(known["q_true"], known["depth_cm"], color="black", lw=1.6, label="Known q", zorder=3)
            for model, colour, label in (("highfreq", "#b2182b", "k = 0…7"), ("lowfreq", "#2166ac", "k = 0…3")):
                one = block.loc[block["model"] == model]
                ax.plot(one["q_pred"], one["depth_cm"], color=colour, lw=1.3, label=label, zorder=2)
            ax.set_ylim(100, 0)
            ax.tick_params(labelsize=11)
            if row == 0:
                ax.set_title("Decay" if truth == "decay" else "Deep peak", fontsize=12)
            if col == 0:
                name = "Irregular layers" if design == "irregular_layers" else "One 0–100 cm label"
                ax.set_ylabel(f"Depth (cm)\n{name}", fontsize=12)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[:3], texts[:3], loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "fig15_linked")


def _figure_gap(audit: dict) -> None:
    use_map_font()
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    orders = sorted(int(k) for k in audit["orders"])
    xpos = np.arange(len(orders))
    median = [audit["orders"][str(n)]["additivity_0_30_100"]["median_abs"] for n in orders]
    p95 = [audit["orders"][str(n)]["additivity_0_30_100"]["p95_abs"] for n in orders]
    ax.plot(xpos, median, marker="o", color="#2166ac", label="Median")
    ax.plot(xpos, p95, marker="s", color="#b2182b", label="P95")
    ax.set_yscale("log")
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(n) for n in orders])
    ax.set_xlabel("Gauss–Legendre points")
    ax.set_ylabel("|S(0,100) − S(0,30) − S(30,100)| (kg m$^{-2}$)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, "fig12_quadrature_gap")


def figure_nullspace() -> None:
    use_map_font()
    z = np.linspace(0, 100, 801)
    q = 0.18 * np.exp(-z / 28.0)
    bounds = ((0.0, 18.0), (18.0, 47.0), (47.0, 100.0))
    h = np.zeros_like(z)
    for top, bottom in bounds:
        mask = (z >= top) & (z <= bottom)
        floor = float(np.min(q[mask]))
        h[mask] = 0.65 * floor * np.sin(2.0 * np.pi * (z[mask] - top) / (bottom - top))
    q2 = q + h
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 4.2), sharey=True)
    panels = (
        (q, "Known q(z)", "#2166ac"),
        (h, "Perturbation with zero layer integrals", "#762a83"),
        (q2, "q + perturbation, same labels", "#b2182b"),
    )
    for ax, (curve, title, colour) in zip(axes, panels):
        ax.plot(curve, z, color=colour, lw=1.4)
        for top, bottom in bounds:
            ax.axhline(bottom, color="#4d4d4d", lw=1.15)
        ax.set_ylim(100, 0)
        ax.set_title(("(a)", "(b)", "(c)")[list(axes).index(ax)], fontsize=12, loc="left")
        ax.set_xlabel("kg m$^{-2}$ cm$^{-1}$", fontsize=10)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Depth (cm)")
    fig.tight_layout()
    _save(fig, "fig13_nullspace")


def figure_quadrature() -> None:
    audit = json.loads((METRICS / "integral_audit.json").read_text(encoding="utf-8"))
    high_path = METRICS / "round3_quadrature.json"
    if high_path.exists():
        high = json.loads(high_path.read_text(encoding="utf-8"))
        audit["orders"].update(high["orders"])
    round5 = TABLES / "round5_q_examples.csv"
    use_map_font()
    if round5.exists():
        extra = pd.read_csv(round5)
        extra["profile_id"] = extra["profile_id"].astype(str)
        layers = pd.read_parquet(PROCESSED / "layers_qc.parquet")
        layers["profile_id"] = layers["profile_id"].astype(str)
        available = list(dict.fromkeys(extra["profile_id"]))
        preferred = [pid for pid in ("3878", "3880", "3303", "3653") if pid in available]
        picks = preferred or available[:4]
        fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.2))
        colours = {"highfreq": "#b2182b", "lowfreq": "#2166ac", "cumulative": "#4d9221"}
        labels = {"highfreq": "k = 0…7", "lowfreq": "k = 0…3", "cumulative": "cumulative F"}
        sampled_style = {"highfreq": "-", "lowfreq": "-.", "cumulative": ":"}
        for ax, pid in zip(axes.ravel(), picks):
            part = layers.loc[layers["profile_id"] == pid]
            z_obs = float(part["bottom_cm"].max()) if len(part) else 100.0
            thick = (part["bottom_cm"] - part["top_cm"]).clip(lower=0.5)
            q_obs = part["socd_kgm2"] / thick
            for top, bot, q in zip(part["top_cm"], part["bottom_cm"], q_obs):
                ax.add_patch(Rectangle((0, top), float(q), max(float(bot) - float(top), 0.2), facecolor="#fddbc7", edgecolor="#f4a582", lw=0.25, alpha=0.28, zorder=0))
            for model, colour in colours.items():
                curve = extra.loc[(extra["profile_id"] == pid) & (extra["model"] == model)].sort_values("depth_cm")
                if curve.empty:
                    continue
                above = curve.loc[curve["depth_cm"] <= z_obs]
                below = curve.loc[curve["depth_cm"] >= z_obs]
                ax.plot(above["q"], above["depth_cm"], color=colour, lw=2.2, ls=sampled_style[model], label=labels[model], zorder=3)
                ax.plot(below["q"], below["depth_cm"], color=colour, lw=1.3, ls="--", alpha=0.7, zorder=3)
            ax.axhline(z_obs, color="#666666", lw=0.8, ls="-")
            ax.set_ylim(200, 0)
            ax.set_xlim(left=0)
            ax.set_title(f"Profile {pid}", fontsize=11)
            ax.tick_params(labelsize=10)
            ax.set_xlabel("kg m$^{-2}$ cm$^{-1}$", fontsize=11)
        axes[0, 0].set_ylabel("Depth (cm)", fontsize=11)
        axes[1, 0].set_ylabel("Depth (cm)", fontsize=11)
        handles, texts = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, texts, loc="lower center", ncol=3, frameon=False, fontsize=11)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        _save(fig, "fig11_quadrature")
        _figure_gap(audit)
        return
    fine = TABLES / "q_curves_0p25cm.csv"
    curves = pd.read_csv(fine if fine.exists() else TABLES / "q_curves_examples.csv")
    extra_path = TABLES / "round4_q_examples.csv"
    use_map_font()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    primary = curves.loc[curves["profile_id"].astype(str) == "3878"].sort_values("depth_cm")
    if "fold" in primary.columns:
        primary = primary.loc[primary["fold"] == primary["fold"].min()]
    axes[0].plot(primary["q"], primary["depth_cm"], color="#b2182b", lw=0.8, label="k = 0…7")
    if extra_path.exists():
        extra = pd.read_csv(extra_path)
        extra["profile_id"] = extra["profile_id"].astype(str)
        colours = {"lowfreq": "#2166ac", "cumulative": "#4d9221", "fine_smooth": "#762a83"}
        for model, colour in colours.items():
            part = extra.loc[(extra["model"] == model) & (extra["profile_id"] == "3878")].sort_values("depth_cm")
            if part.empty:
                continue
            axes[0].plot(part["q"], part["depth_cm"], color=colour, lw=1.1, label=model.replace("_", " "))
    axes[0].invert_yaxis()
    axes[0].set_xlabel("q(z) (kg m$^{-2}$ cm$^{-1}$)", fontsize=8)
    axes[0].set_ylabel("Depth (cm)", fontsize=8)
    axes[0].tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Profile 3878, sampled every 0.25 cm", fontsize=9)
    orders = sorted(int(k) for k in audit["orders"])
    xpos = np.arange(len(orders))
    median = [audit["orders"][str(n)]["additivity_0_30_100"]["median_abs"] for n in orders]
    p95 = [audit["orders"][str(n)]["additivity_0_30_100"]["p95_abs"] for n in orders]
    axes[1].plot(xpos, median, marker="o", color="#2166ac", label="Median |residual|")
    axes[1].plot(xpos, p95, marker="s", color="#b2182b", label="P95 |residual|")
    axes[1].set_yscale("log")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([str(n) for n in orders], fontsize=8)
    axes[1].set_xlabel("Gauss–Legendre points on each interval", fontsize=8)
    axes[1].set_ylabel("|S(0,100) − S(0,30) − S(30,100)| (kg m$^{-2}$)", fontsize=8)
    axes[1].tick_params(labelsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Same q, not a retrained model", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig11_quadrature")


def figure_distance() -> None:
    payload = json.loads((METRICS / "round2_stats.json").read_text(encoding="utf-8"))
    rows = [r for r in payload["distance_bins"] if r["target"] == "y100"]
    paired = {r["bin"]: r for r in payload["distance_bin_paired"]}
    use_map_font()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    order = ["0–25 km", "25–50 km", "50–100 km", ">100 km"]
    xpos = np.arange(len(order))
    for model, color, shift in (("mcu", "#2166ac", -0.08), ("rf", "#4d4d4d", 0.08)):
        part = {r["bin"]: r for r in rows if r["model"] == model}
        mae = [part[b]["mae"] for b in order]
        axes[0].plot(xpos + shift, mae, marker="o", color=color, lw=1.3, label="Primary integral" if model == "mcu" else "Random forest")
    counts = [next(r["n"] for r in rows if r["model"] == "mcu" and r["bin"] == b) for b in order]
    axes[0].set_xticks(xpos)
    axes[0].set_xticklabels([f"{b}\nn={n}" for b, n in zip(order, counts)], fontsize=11)
    axes[0].set_ylabel("0–100 cm MAE (kg m$^{-2}$)", fontsize=12)
    axes[0].tick_params(labelsize=11)
    axes[0].set_ylim(3.8, 6.3)
    axes[0].legend(frameon=False, fontsize=11)
    axes[0].set_title("(a)", fontsize=12, loc="left")
    means = [paired[b]["mean_rf_minus_mcu"] for b in order]
    lo = [paired[b]["mean_rf_minus_mcu"] - paired[b]["ci95_low"] for b in order]
    hi = [paired[b]["ci95_high"] - paired[b]["mean_rf_minus_mcu"] for b in order]
    axes[1].errorbar(xpos, means, yerr=[lo, hi], fmt="o", color="#2166ac", ecolor="#666666", capsize=3, ms=5)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(order, fontsize=11)
    axes[1].set_ylabel("Random forest MAE − primary MAE", fontsize=12)
    axes[1].tick_params(labelsize=11)
    axes[1].set_title("(b)", fontsize=12, loc="left")
    fig.tight_layout()
    _save(fig, "fig09_distance")


def figure_profile_density() -> None:
    profiles = pd.read_parquet(PROCESSED / "profiles_qc.parquet")
    use_map_font()
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    provinces = load_provinces().to_crs(CHINA_ALBERS)
    provinces.plot(ax=ax, color="#f7f7f7", edgecolor="none")
    px, py = TRANSFORMER.transform(profiles["longitude"].to_numpy(), profiles["latitude"].to_numpy())
    minx, miny, maxx, maxy = provinces.total_bounds
    cell = 100_000.0
    ix = np.floor((px - minx) / cell).astype(int)
    iy = np.floor((py - miny) / cell).astype(int)
    nx = int(np.floor((maxx - minx) / cell)) + 1
    ny = int(np.floor((maxy - miny) / cell)) + 1
    grid = np.full((ny, nx), np.nan)
    for col, row in zip(ix, iy):
        if 0 <= col < nx and 0 <= row < ny:
            grid[row, col] = 0.0 if not np.isfinite(grid[row, col]) else grid[row, col]
            grid[row, col] += 1.0
    extent = [minx, minx + nx * cell, miny, miny + ny * cell]
    edges = np.array([1, 5, 10, 20, 40, 80, np.nanmax(grid)])
    norm = BoundaryNorm(edges, 6)
    mesh = ax.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#edf8fb", "#b3cde3", "#8c96c6", "#8856a7", "#810f7c", "#4d004b"]),
        norm=norm,
        interpolation="nearest",
        zorder=3,
    )
    draw_official_boundaries(ax)
    set_official_extent(ax)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.01, ticks=edges)
    cbar.set_label("Profiles per 10$^{4}$ km$^{2}$", fontsize=10)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    _save(fig, "fig10_profile_density")


def figure_layer_recovery() -> None:
    path = TABLES / "round9_layer_profiles.csv"
    if not path.exists():
        return
    detail = pd.read_csv(path)
    detail = detail.loc[
        (detail["parameter_source"] == "deterministic")
        & (detail["truth"] == "deep_peak")
        & (detail["noise"] == 0.15)
        & (detail["design"].astype(str).str.startswith("equal_"))
    ].copy()
    if detail.empty:
        return
    detail["n_layers"] = detail["design"].str.replace("equal_", "", regex=False).astype(int)
    detail["s30_abs"] = (detail["s30_true"] - detail["s30_pred"]).abs()
    detail["s30_60_abs"] = (detail["s30_60_true"] - detail["s30_60_pred"]).abs()
    metrics = {
        "rel_l2_0_100": "Relative L2 on 0–100 cm",
        "peak_abs_err_cm": "Peak-depth error (cm)",
        "s30_abs": "0–30 cm absolute error",
        "s30_60_abs": "30–60 cm absolute error",
    }
    use_map_font()
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    colours = {"highfreq": "#b2182b", "lowfreq": "#2166ac"}
    names = {"highfreq": "k = 0…7", "lowfreq": "k = 0…3"}
    styles = {"highfreq": "-", "lowfreq": "--"}
    n_test = int(detail["profile_id"].nunique())
    n_seeds = int(detail["seed"].nunique())
    for ax, (column, ylabel) in zip(axes.ravel(), metrics.items()):
        for model, colour in colours.items():
            one = detail.loc[detail["model"] == model]
            by_seed = one.groupby(["n_layers", "seed"])[column].median().reset_index()
            summary = by_seed.groupby("n_layers")[column].agg(["mean", "min", "max"]).reset_index()
            ax.plot(summary["n_layers"], summary["mean"], color=colour, ls=styles[model], marker="o", label=names[model])
            ax.fill_between(summary["n_layers"], summary["min"], summary["max"], color=colour, alpha=0.15)
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.tick_params(labelsize=9)
        ax.set_xlabel("Equal layers on 0–100 cm", fontsize=9)
        unit = "" if "Relative" in ylabel or "Peak" in ylabel else " (kg m$^{-2}$)"
        ax.set_ylabel(ylabel + unit, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save(fig, "fig16_layer_recovery")


def figure_peak_geometry() -> None:
    path = TABLES / "round10_peak_summary.csv"
    if not path.exists():
        return
    summary = pd.read_csv(path)
    summary["n_layers"] = summary["design"].str.replace("equal_", "", regex=False).astype(int)
    labels = {
        "narrow_w6": "width 6 cm",
        "baseline_extra_seeds": "width 12 cm, seeds 47–49",
        "wide_w24": "width 24 cm",
        "centre_40": "centre 40 cm",
        "centre_90": "centre 90 cm",
    }
    colours = {
        "narrow_w6": "#b2182b",
        "baseline_extra_seeds": "#4d4d4d",
        "wide_w24": "#2166ac",
        "centre_40": "#e66101",
        "centre_90": "#5e3c99",
    }
    use_map_font()
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), sharex=True)
    panels = (
        (0, "highfreq", "rel_l2_mean_of_seed_medians", "Relative L2, k = 0…7"),
        (1, "lowfreq", "rel_l2_mean_of_seed_medians", "Relative L2, k = 0…3"),
        (2, "highfreq", "frac_peak_le_20", "Fraction with peak error ≤ 20 cm, k = 0…7"),
        (3, "lowfreq", "frac_peak_le_20", "Fraction with peak error ≤ 20 cm, k = 0…3"),
    )
    for index, model, column, title in panels:
        ax = axes.ravel()[index]
        for variant, colour in colours.items():
            one = summary.loc[(summary["variant"] == variant) & (summary["model"] == model)].sort_values("n_layers")
            ax.plot(
                one["n_layers"],
                one[column],
                color=colour,
                marker="o",
                lw=1.4,
                label=labels[variant],
            )
        ax.set_xticks([1, 2, 3, 5, 10])
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=13)
        ax.set_xlabel("Equal layers on 0–100 cm", fontsize=13)
    axes[1, 0].set_ylim(-0.02, 1.05)
    axes[1, 1].set_ylim(-0.02, 1.05)
    handles, texts = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, texts, loc="lower center", ncol=3, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig17_peak_geometry")


def make_available() -> None:
    figure_profiles()
    figure_workflow()
    figure_architecture()
    figure_profiles_curves()
    figure_model_comparison()
    figure_ablation_calibration()
    figure_national_maps()
    figure_xinjiang()
    figure_distance()
    figure_quadrature()
    figure_nullspace()
    figure_profile_density()


if __name__ == "__main__":
    make_available()

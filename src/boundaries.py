"""Official GS（2024）0650 administrative boundaries.

Polygons supply land area, province assignment, and fill masks.
The companion line layer (境界线) is drawn on every national map so the
approved boundary, including the South China Sea segments, is visible.
"""

from __future__ import annotations

import functools

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import box

from .paths import (
    CHINA_ALBERS,
    CITY_SHP,
    COUNTY_SHP,
    MAP_REVIEW_LABEL,
    PROVINCE_LINE_SHP,
    PROVINCE_SHP,
)

ENGLISH_NAME = {
    "北京市": "Beijing",
    "天津市": "Tianjin",
    "河北省": "Hebei",
    "山西省": "Shanxi",
    "内蒙古自治区": "Inner Mongolia",
    "辽宁省": "Liaoning",
    "吉林省": "Jilin",
    "黑龙江省": "Heilongjiang",
    "上海市": "Shanghai",
    "江苏省": "Jiangsu",
    "浙江省": "Zhejiang",
    "安徽省": "Anhui",
    "福建省": "Fujian",
    "江西省": "Jiangxi",
    "山东省": "Shandong",
    "河南省": "Henan",
    "湖北省": "Hubei",
    "湖南省": "Hunan",
    "广东省": "Guangdong",
    "广西壮族自治区": "Guangxi",
    "海南省": "Hainan",
    "重庆市": "Chongqing",
    "四川省": "Sichuan",
    "贵州省": "Guizhou",
    "云南省": "Yunnan",
    "西藏自治区": "Xizang",
    "陕西省": "Shaanxi",
    "甘肃省": "Gansu",
    "青海省": "Qinghai",
    "宁夏回族自治区": "Ningxia",
    "新疆维吾尔自治区": "Xinjiang",
    "香港特别行政区": "Hong Kong",
    "澳门特别行政区": "Macao",
    "台湾省": "Taiwan",
}

XINJIANG_NAME = "新疆维吾尔自治区"


def use_map_font() -> None:
    font_path = r"C:\Windows\Fonts\msyh.ttc"
    try:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = "Microsoft YaHei"
    except Exception:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False


@functools.lru_cache(maxsize=1)
def load_provinces() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PROVINCE_SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326, allow_override=True)
    gdf["name_en"] = gdf["name"].map(lambda n: ENGLISH_NAME.get(n, n))
    missing = sorted(set(gdf["name"]) - set(ENGLISH_NAME))
    if missing:
        raise RuntimeError(f"Untranslated official province names: {missing}")
    return gdf


@functools.lru_cache(maxsize=1)
def load_province_lines() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(PROVINCE_LINE_SHP)
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.set_crs(4326, allow_override=True)
    return gdf


@functools.lru_cache(maxsize=1)
def load_cities() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(CITY_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


@functools.lru_cache(maxsize=1)
def load_counties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(COUNTY_SHP)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    return gdf


def official_bounds():
    """Longitude/latitude bounds covering polygons and 境界线."""
    poly = load_provinces().total_bounds
    line = load_province_lines().total_bounds
    return (
        float(min(poly[0], line[0])),
        float(min(poly[1], line[1])),
        float(max(poly[2], line[2])),
        float(max(poly[3], line[3])),
    )


def xinjiang_polygon():
    gdf = load_provinces()
    hit = gdf.loc[gdf["name"] == XINJIANG_NAME]
    if hit.empty:
        raise RuntimeError("Xinjiang polygon is missing from the official province layer.")
    return hit.geometry.union_all()


def assign_province(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign each point to an official province polygon.

    Points that fall just outside a vertex are assigned to the nearest
    province only when that province is within 5 km. All others stay
    unassigned and are excluded from land-masked products.
    """
    provinces = load_provinces()[["name", "name_en", "gb", "geometry"]]
    pts = points.copy()
    if pts.crs is None:
        pts = pts.set_crs(4326)
    joined = gpd.sjoin(pts, provinces, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]
    missing = joined["name"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            pts.loc[missing, ["geometry"]].to_crs(CHINA_ALBERS),
            provinces.to_crs(CHINA_ALBERS),
            how="left",
            max_distance=5000,
            distance_col="snap_m",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        for col in ("name", "name_en", "gb"):
            joined.loc[nearest.index, col] = nearest[col]
    joined["in_official_land"] = joined["name"].notna()
    return joined.drop(columns=[c for c in joined.columns if c.startswith("index_")], errors="ignore")


def add_review_number(ax, x: float = 0.01, y: float = 0.01) -> None:
    use_map_font()
    ax.text(
        x,
        y,
        MAP_REVIEW_LABEL,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        zorder=10,
    )


def add_scale_bar(ax, length_m: float = 500_000, label: str = "500 km") -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x0 + 0.06 * (x1 - x0)
    y = y0 + 0.06 * (y1 - y0)
    ax.plot([x, x + length_m], [y, y], color="black", lw=1.4, solid_capstyle="butt", zorder=10)
    ax.plot([x, x], [y - 8_000, y + 8_000], color="black", lw=1.0, zorder=10)
    ax.plot([x + length_m, x + length_m], [y - 8_000, y + 8_000], color="black", lw=1.0, zorder=10)
    ax.text(x + length_m / 2, y + 18_000, label, ha="center", va="bottom", fontsize=7, zorder=10)


def add_north_arrow(ax) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x = x1 - 0.08 * (x1 - x0)
    y = y0 + 0.16 * (y1 - y0)
    dy = 0.07 * (y1 - y0)
    ax.add_patch(
        FancyArrowPatch(
            (x, y),
            (x, y + dy),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color="black",
            zorder=10,
        )
    )
    ax.text(x, y + dy + 0.012 * (y1 - y0), "N", ha="center", va="bottom", fontsize=8, zorder=10)


def draw_official_boundaries(ax, provinces=None, lines=None, edgecolor="#222222", linewidth=0.25) -> None:
    """Draw official province fill edges and the 境界线 layer."""
    use_map_font()
    if provinces is None:
        provinces = load_provinces()
    if lines is None:
        lines = load_province_lines()
    provinces.to_crs(CHINA_ALBERS).boundary.plot(ax=ax, color=edgecolor, linewidth=linewidth, zorder=4)
    lines.to_crs(CHINA_ALBERS).plot(ax=ax, color="black", linewidth=0.55, zorder=5)


def set_official_extent(ax, extra: gpd.GeoDataFrame | None = None, pad_frac: float = 0.02) -> None:
    """Extent must include both polygons and 境界线."""
    parts = [load_provinces().to_crs(CHINA_ALBERS), load_province_lines().to_crs(CHINA_ALBERS)]
    if extra is not None and len(extra):
        layer = extra if extra.crs is not None and extra.crs.to_string() == CHINA_ALBERS else extra.to_crs(CHINA_ALBERS)
        parts.append(layer)
    minx = min(g.total_bounds[0] for g in parts)
    miny = min(g.total_bounds[1] for g in parts)
    maxx = max(g.total_bounds[2] for g in parts)
    maxy = max(g.total_bounds[3] for g in parts)
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - pad_frac * dx, maxx + pad_frac * dx)
    ax.set_ylim(miny - pad_frac * dy, maxy + pad_frac * dy)
    ax.set_aspect("equal")
    ax.set_axis_off()


def locator_inset(fig, bounds_geom=None, rect=(0.62, 0.14, 0.22, 0.22)):
    """Small national locator that still shows 境界线 and the review number."""
    ax = fig.add_axes(list(rect))
    provinces = load_provinces().to_crs(CHINA_ALBERS)
    lines = load_province_lines().to_crs(CHINA_ALBERS)
    provinces.plot(ax=ax, color="#f4f4f4", edgecolor="#666666", linewidth=0.15)
    lines.plot(ax=ax, color="black", linewidth=0.35)
    if bounds_geom is not None:
        gpd.GeoSeries([bounds_geom], crs=4326).to_crs(CHINA_ALBERS).boundary.plot(
            ax=ax, color="#b2182b", linewidth=0.8
        )
    set_official_extent(ax, pad_frac=0.01)
    add_review_number(ax, x=0.02, y=0.02)
    ax.set_title(" ", fontsize=1)
    return ax


def province_lookup() -> dict[str, str]:
    gdf = load_provinces()
    return dict(zip(gdf["name"], gdf["name_en"]))


def land_rectangle_wgs84():
    minx, miny, maxx, maxy = official_bounds()
    return box(minx, miny, maxx, maxy)

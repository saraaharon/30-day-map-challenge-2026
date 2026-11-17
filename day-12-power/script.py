"""
#30DayMapChallenge | Day 12: Power
Romania's Power Grid: plants as glowing nodes sized by MW,
220 kV and 400 kV lines as neon veins.

Data:  OpenStreetMap via Overpass API (cached locally after first run)
Bound: USDOS LSIB via Google Earth Engine
Run:   python day12_power_grid.py            -> day12_power.png
       python day12_power_grid.py --gif      -> also day12_power.gif (pulsing nodes)
Needs: pip install requests geopandas shapely matplotlib numpy earthengine-api imageio
"""
import argparse
import io
import json
import os
import re
import time

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests
from shapely.geometry import LineString, Point, shape

# ----------------------------------------------------------------- config
GEE_PROJECT = "advance-vector-464019-f8"
AUTHOR = "Sara AHARON"
CRS_RO = 3844  # Stereo 70

BG = "#0b0b1e"
LAND = "#12122e"
LAND_EDGE = "#2c2c66"
TITLE_SUB = "#ff2bd6"
FOOT = "#8a93b0"

SOURCE_COLORS = {
    "nuclear": "#ffe14d",
    "hydro": "#00e5ff",
    "wind": "#39ff88",
    "solar": "#ff9f1c",
    "coal": "#ff3b5c",
    "gas": "#b967ff",
    "other": "#8a93b0",
}
SOURCE_LABELS = {
    "nuclear": "Nuclear", "hydro": "Hydro", "wind": "Wind", "solar": "Solar",
    "coal": "Coal", "gas": "Gas", "other": "Other",
}
KV400, KV220 = "#ff2bd6", "#7b5cff"

OVERPASS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
Q_PLANTS = """
[out:json][timeout:240];
area["ISO3166-1"="RO"][admin_level=2]->.ro;
nwr["power"="plant"](area.ro);
out center tags;
"""
Q_LINES = """
[out:json][timeout:240];
area["ISO3166-1"="RO"][admin_level=2]->.ro;
way["power"="line"]["voltage"~"400000|220000"](area.ro);
out geom tags;
"""


# ----------------------------------------------------------------- data
def overpass(query, cache):
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    headers = {"User-Agent": "30DayMapChallenge-Day12/1.0 (student map project)"}
    errors = []
    for attempt in range(4):
        for url in OVERPASS:
            try:
                r = requests.post(url, data={"data": query}, headers=headers, timeout=300)
                if r.status_code in (429, 504):
                    errors.append(f"{url}: {r.status_code}")
                    continue
                r.raise_for_status()
                data = r.json()
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                time.sleep(5)  # be polite before the next query
                return data
            except Exception as e:  # try next mirror
                errors.append(f"{url}: {e}")
        wait = 20 * (attempt + 1)
        print(f"Overpass busy, waiting {wait}s (attempt {attempt + 1}/4)...")
        time.sleep(wait)
    raise RuntimeError("Overpass failed:\n" + "\n".join(errors[-6:]))


def parse_mw(txt):
    """OSM output tag -> MW. No unit means watts (OSM convention)."""
    if not txt:
        return None
    m = re.search(r"([\d]+(?:[.,]\d+)?)\s*([a-zA-Z]*)", txt)
    if not m:
        return None
    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower()
    if unit.startswith("gw"):
        return val * 1000
    if unit.startswith("mw"):
        return val
    if unit.startswith("kw"):
        return val / 1000
    if unit in ("w", ""):
        return val / 1e6
    return None


def group_source(tags):
    s = (tags.get("plant:source") or tags.get("generator:source") or "").lower()
    if "nuclear" in s:
        return "nuclear"
    if "hydro" in s or "water" in s:
        return "hydro"
    if "wind" in s:
        return "wind"
    if "solar" in s:
        return "solar"
    if any(k in s for k in ("coal", "lignite")):
        return "coal"
    if "gas" in s:
        return "gas"
    return "other"


def load_plants():
    data = overpass(Q_PLANTS, "osm_ro_plants.json")
    rows = []
    for el in data["elements"]:
        tags = el.get("tags", {})
        if el["type"] == "node":
            lat, lon = el["lat"], el["lon"]
        else:
            c = el.get("center")
            if not c:
                continue
            lat, lon = c["lat"], c["lon"]
        mw = parse_mw(tags.get("plant:output:electricity"))
        rows.append({
            "name": tags.get("name", ""),
            "src": group_source(tags),
            "mw": mw,
            "geometry": Point(lon, lat),
        })
    g = gpd.GeoDataFrame(rows, crs=4326).to_crs(CRS_RO)
    g["mw_plot"] = g["mw"].fillna(3.0).clip(lower=1.0)  # untagged capacity -> small dot
    return g.sort_values("mw_plot", ascending=False).reset_index(drop=True)


def load_lines():
    data = overpass(Q_LINES, "osm_ro_lines.json")
    rows = []
    for el in data["elements"]:
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        volt = el.get("tags", {}).get("voltage", "")
        kv = 400 if "400000" in volt else 220
        rows.append({"kv": kv, "geometry": LineString([(p["lon"], p["lat"]) for p in geom])})
    return gpd.GeoDataFrame(rows, crs=4326).to_crs(CRS_RO)


def load_boundary():
    ee.Initialize(project=GEE_PROJECT)
    fc = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq("country_na", "Romania"))
    geom = shape(fc.geometry().getInfo())
    return gpd.GeoDataFrame(geometry=[geom], crs=4326).to_crs(CRS_RO)


# ----------------------------------------------------------------- drawing
def size_of(mw):
    return 8 + 6 * np.sqrt(mw)


def build(plants, lines, ro):
    fig = plt.figure(figsize=(10.8, 13.5), facecolor=BG)
    ax = fig.add_axes([0.03, 0.16, 0.94, 0.70], facecolor=BG)
    ax.set_aspect("equal")
    ax.axis("off")

    ro.plot(ax=ax, color=LAND, edgecolor=LAND_EDGE, linewidth=1.2, zorder=1)

    # lines: soft glow layers + bright core
    l220, l400 = lines[lines.kv == 220], lines[lines.kv == 400]
    for lw, a in ((2.6, 0.05), (1.2, 0.10), (0.45, 0.65)):
        l220.plot(ax=ax, color=KV220, linewidth=lw, alpha=a, zorder=2)
    for lw, a in ((5.0, 0.06), (2.6, 0.12), (1.0, 0.95)):
        l400.plot(ax=ax, color=KV400, linewidth=lw, alpha=a, zorder=3)

    # plants: two glow layers + core, one set per source
    rng = np.random.default_rng(12)
    pulse = []  # (collection, base_sizes, phases)
    for src, col in SOURCE_COLORS.items():
        sub = plants[plants.src == src]
        if sub.empty:
            continue
        x, y = sub.geometry.x.values, sub.geometry.y.values
        base = size_of(sub.mw_plot.values)
        phases = rng.random(len(sub))
        for mult, a in ((7.0, 0.07), (3.2, 0.16)):
            c = ax.scatter(x, y, s=base * mult, color=col, alpha=a, linewidths=0, zorder=4)
            pulse.append((c, base * mult, phases))
        ax.scatter(x, y, s=base, color=col, alpha=0.95, linewidths=0.4,
                   edgecolors="white", zorder=5)

    minx, miny, maxx, maxy = ro.total_bounds
    pad = 0.03 * max(maxx - minx, maxy - miny)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)

    # header (project layout standard)
    km = lines.length.sum() / 1000
    gw = plants.mw.dropna().sum() / 1000
    fig.text(0.06, 0.93, "Romania's Power Grid", color="white", fontsize=26,
             fontweight="bold")
    fig.text(0.06, 0.895, "Plants as stars, 400 kV and 220 kV lines as neon veins",
             color=TITLE_SUB, fontsize=14)
    fig.text(0.06, 0.870,
             f"{len(plants):,} mapped plants  |  {gw:,.1f} GW with a capacity tag  |  "
             f"~{km:,.0f} km of 220-400 kV lines",
             color=FOOT, fontsize=11)

    # legend strip
    lg = fig.add_axes([0.06, 0.105, 0.88, 0.085], facecolor="none")
    lg.set_xlim(0, 1)
    lg.set_ylim(0, 1)
    lg.axis("off")
    present = [s for s in SOURCE_COLORS if (plants.src == s).any()]
    step = 1 / len(present)
    for i, s in enumerate(present):
        xx = i * step + 0.015
        lg.scatter([xx], [0.75], s=90, color=SOURCE_COLORS[s], edgecolors="white",
                   linewidths=0.4)
        lg.text(xx + 0.02, 0.75, SOURCE_LABELS[s], color="white", fontsize=12,
                va="center")
    lg.plot([0.0, 0.05], [0.25, 0.25], color=KV400, lw=2.2)
    lg.text(0.06, 0.25, "400 kV", color="white", fontsize=12, va="center")
    lg.plot([0.19, 0.24], [0.25, 0.25], color=KV220, lw=1.4)
    lg.text(0.25, 0.25, "220 kV", color="white", fontsize=12, va="center")
    for j, mw in enumerate((10, 100, 700)):
        xx = 0.52 + j * 0.15
        lg.scatter([xx], [0.25], s=size_of(mw), color="#8a93b0", edgecolors="white",
                   linewidths=0.4)
        lg.text(xx + 0.025, 0.25, f"{mw} MW", color="white", fontsize=12, va="center")

    # footer
    fig.text(0.06, 0.075, f"#30DayMapChallenge | Day 12: Power | {AUTHOR}",
             color=FOOT, fontsize=11)
    fig.text(0.06, 0.052,
             "Data: © OpenStreetMap contributors (power=plant, power=line); plants without "
             "a capacity tag are drawn as small dots.",
             color=FOOT, fontsize=10)
    fig.text(0.06, 0.034, "Boundary: USDOS LSIB. Projection: Stereo 70.",
             color=FOOT, fontsize=10)
    return fig, pulse


def make_gif(fig, pulse, path, frames=36, dpi=70):
    import imageio.v2 as imageio
    imgs = []
    for k in range(frames):
        t = k / frames
        for coll, base, ph in pulse:
            coll.set_sizes(base * (1 + 0.35 * np.sin(2 * np.pi * (t + ph))))
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=BG)
        buf.seek(0)
        imgs.append(imageio.imread(buf))
    imageio.mimsave(path, imgs, duration=1 / 15, loop=0)


# ----------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true", help="also write a pulsing GIF")
    args = ap.parse_args()

    plants, lines, ro = load_plants(), load_lines(), load_boundary()
    print(f"{len(plants)} plants, {len(lines)} line segments")
    fig, pulse = build(plants, lines, ro)
    fig.savefig("day12_power.png", dpi=100, facecolor=BG)
    print("saved day12_power.png")
    if args.gif:
        make_gif(fig, pulse, "day12_power.gif")
        print("saved day12_power.gif")

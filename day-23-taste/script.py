#!/usr/bin/env python3
"""
Day 23 | #30DayMapChallenge | Senses: Taste
The Taste Wave: expanding from Focșani along ALL Romanian roads simultaneously.
Map by Sara AHARON.

Data : local Geofabrik extract, (c) OpenStreetMap contributors + GADM 4.1 boundary
Needs: pip install pyrosm geopandas networkx matplotlib shapely numpy scipy imageio imageio-ffmpeg
"""
import os
import re
import json
import math
import zipfile
import urllib.request
import unicodedata
from collections import Counter

import geopandas as gpd
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import patheffects as pe
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from pyrosm import OSM
from shapely.geometry import LineString, Point

# ----------------------------------------------------------------- CONFIG
PBF = r"C:\Users\LENOVO\Desktop\30day\_archive\14\romania-latest.osm.pbf"
BOUNDARY_PATH = None   # optional path to gadm41_ROU_0.json; auto-downloads if missing
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_0.json.zip"
OUT = "day23_mici_focsani.mp4"
CRS = 3844             # Stereo 70 (metres)

ORIGIN = ("Focșani", 45.6967, 27.1833)

# Reference cities shown on map (matching Day 17)
CITIES = [
    ("Sulina",      29.6667, 45.1583, (-10000,  14000), "right"),
    ("Constanța",   28.6348, 44.1598, ( -8000, -15000), "right"),
    ("Iași",        27.5766, 47.1585, ( -8000,  10000), "right"),
    ("București",   26.1025, 44.4268, (     0, -17000), "center"),
    ("Cluj-Napoca", 23.5914, 46.7712, (     0,  11000), "center"),
    ("Timișoara",   21.2272, 45.7489, (     0, -17000), "center"),
]

FPS, MOVE, HOLD = 30, 390, 90   # 13s expansion + 3s hold
RING = 20                       # frames a discovery pulse lasts
SNAP_MAX = 4000                 # m: max distance from a venue to the road network

# Day 17 layout palette + requested dish colors
BG      = "#0b0b1a"
BOX     = "#0e0e22"
GREY    = "#8a93b0"
OUTLINE = "#b8c0ff"
AMBER   = "#ffb000"   # Mici
MAGENTA = "#ff2e97"   # Sarmale
CYAN    = "#00f5d4"   # Papanași
WAVE_C  = "#ffffff"

# All major + regional + local connecting Romanian roads
ROAD_W = {
    "motorway": 1.00, "motorway_link": 1.15,
    "trunk": 1.15,    "trunk_link": 1.30,
    "primary": 1.35,  "primary_link": 1.50,
    "secondary": 1.65,"secondary_link": 1.80,
    "tertiary": 2.10, "tertiary_link": 2.25,
    "unclassified": 2.80,
}

# Base road styling (dim background network)
BASE_DRAW = {
    "motorway":     (1.15, 0.34),
    "trunk":        (0.85, 0.28),
    "primary":      (0.60, 0.22),
    "secondary":    (0.40, 0.16),
    "tertiary":     (0.25, 0.11),
    "unclassified": (0.16, 0.07),
}

# Illuminated road styling once the wave passes
LIT_DRAW = {
    "motorway":     (1.65, 0.85),
    "trunk":        (1.25, 0.72),
    "primary":      (0.90, 0.58),
    "secondary":    (0.60, 0.44),
    "tertiary":     (0.38, 0.30),
    "unclassified": (0.22, 0.18),
}


# ------------------------------------------------------------ BOUNDARY
def rdp(pts, eps):
    n = len(pts)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = np.hypot(*ab)
        seg = pts[i + 1:j] - a
        d = np.hypot(seg[:, 0], seg[:, 1]) if L == 0 else np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def load_boundary_3844(path=None):
    if path is None:
        for cand in ("gadm41_ROU_0.json", "gadm41_ROU_0.geojson", "romania.geojson", "ROU.geojson"):
            if os.path.exists(cand):
                path = cand
                break
    if path is None:
        print("Downloading Romania boundary from GADM...")
        urllib.request.urlretrieve(GADM_URL, "gadm41_ROU_0.json.zip")
        with zipfile.ZipFile("gadm41_ROU_0.json.zip") as z:
            z.extract("gadm41_ROU_0.json")
        path = "gadm41_ROU_0.json"

    gj = json.load(open(path, encoding="utf-8"))
    geoms = [f["geometry"] for f in gj["features"]] if gj["type"] == "FeatureCollection" else [gj.get("geometry", gj)]
    rings = []
    for g in geoms:
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        for poly in polys:
            r = np.array(poly[0])[:, :2]
            if np.ptp(r[:, 0]) > 0.03 or np.ptp(r[:, 1]) > 0.03:
                r_simp = rdp(r, 0.002)
                gs = gpd.GeoSeries([LineString(r_simp)], crs=4326).to_crs(CRS).iloc[0]
                rings.append(np.array(gs.coords))
    return rings


# ------------------------------------------------------------ DISH MATCHING
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


PAT_MICI     = re.compile(r"\bmici\b|\bmititei\b|gratar|grill|\bbbq\b|barbecue|rotiser|ceaun")
PAT_SARMALE  = re.compile(r"sarmal|sarmalut|varza a la cluj|cuptor|traditional|romanesc|casa romaneasca|hanul|han\b|crama|taverna|popas|coliba|stana|ospat")
PAT_PAPANASI = re.compile(r"papanas|clatit|placint|desert|cofetari|patiser|gogoas|dulciur|strudel|savarin")

CUIS_MICI     = {"barbecue", "grill", "bbq", "rotisserie"}
CUIS_SARMALE  = {"romanian", "traditional", "regional", "moldovan", "transylvanian", "balkan"}
CUIS_PAPANASI = {"dessert", "pastry", "crepe", "pancake", "bakery", "cake"}


def classify_dish(row):
    text = " ".join(norm(row.get(k, "")) for k in ("name", "cuisine", "description", "dish"))
    cuis = {c.strip() for c in norm(row.get("cuisine", "")).split(";") if c.strip()}

    has_papanasi = bool(PAT_PAPANASI.search(text) or (cuis & CUIS_PAPANASI))
    has_sarmale  = bool(PAT_SARMALE.search(text)  or (cuis & CUIS_SARMALE))
    has_mici     = bool(PAT_MICI.search(text)     or (cuis & CUIS_MICI))

    # Specific dish priority prevents over-counting one venue across multiple categories
    if has_papanasi:
        return "papanași"
    if has_sarmale:
        return "sarmale"
    if has_mici:
        return "mici"
    return None


def line_parts(geom):
    t = geom.geom_type
    if t == "LineString":
        yield geom
    elif t == "MultiLineString":
        yield from geom.geoms
    elif t == "Polygon":
        yield LineString(geom.exterior.coords)
    elif t == "MultiPolygon":
        for p in geom.geoms:
            yield LineString(p.exterior.coords)


# --------------------------------------------------------------- LOAD DATA
print("Reading PBF for all Romanian roads & food venues...")
osm = OSM(PBF)

roads = osm.get_data_by_custom_criteria(
    custom_filter={"highway": list(ROAD_W)},
    osm_keys_to_keep=["highway"],
    filter_type="keep", keep_nodes=False, keep_ways=True, keep_relations=False,
).to_crs(CRS)
roads = roads[roads["highway"].isin(ROAD_W)]
print(f"{len(roads):,} road ways loaded")

venues = osm.get_data_by_custom_criteria(
    custom_filter={"amenity": ["restaurant", "fast_food", "pub", "bar", "cafe", "food_court"]},
    osm_keys_to_keep=["amenity", "name", "cuisine", "description", "dish"],
    filter_type="keep", keep_nodes=True, keep_ways=True, keep_relations=False,
).to_crs(CRS)

for col in ("name", "cuisine", "description", "dish"):
    if col not in venues:
        venues[col] = ""

venues["geometry"] = venues.geometry.representative_point()
venues["dish_cat"] = venues.apply(classify_dish, axis=1)
venues = venues[venues["dish_cat"].notna()].copy()

# Deduplicate nearby OSM node+way pairs for the same venue
venues["key"] = (
    (venues.geometry.x // 120).astype(int).astype(str) + "_" +
    (venues.geometry.y // 120).astype(int).astype(str) + "_" +
    venues["dish_cat"] + "_" +
    venues["name"].fillna("").astype(str).str[:18]
)
venues = venues.drop_duplicates("key").reset_index(drop=True)
print("Matched venues:", venues["dish_cat"].value_counts().to_dict())

# ----------------------------------------- FULL ROMANIAN ROAD GRAPH
ways = []
for hw, geom in zip(roads["highway"], roads.geometry):
    base = hw.replace("_link", "")
    for part in line_parts(geom):
        cs = list(part.coords)
        if len(cs) >= 2:
            ways.append((hw, base, cs))
del roads

cnt = Counter()
for _, _, cs in ways:
    for c in set(cs):
        cnt[c] += 1

print("Building full connected Romanian road graph...")
G = nx.Graph()
for hw, base, cs in ways:
    arr = np.array(cs)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(arr, axis=0).T))])
    start, last = 0, len(cs) - 1
    for i in range(1, len(cs)):
        if i == last or cnt[cs[i]] > 1:
            a, b = cs[start], cs[i]
            if a != b:
                phys_len = float(cum[i] - cum[start])
                w = phys_len / ROAD_W[hw]
                if not G.has_edge(a, b) or G[a][b]["w"] > w:
                    G.add_edge(a, b, w=w, base=base, coords=cs[start:i + 1])
            start = i
del ways, cnt

# Keep main connected component of Romania's road network
comp = max(nx.connected_components(G), key=len)
G = G.subgraph(comp).copy()
nodes_list = list(G.nodes())
node_xy = np.array(nodes_list)

# Snap Focșani to closest road node
o_pt = gpd.GeoSeries([Point(ORIGIN[2], ORIGIN[1])], crs=4326).to_crs(CRS).iloc[0]
o_idx = int(np.argmin(np.hypot(node_xy[:, 0] - o_pt.x, node_xy[:, 1] - o_pt.y)))
o_node = nodes_list[o_idx]
o_xy = (float(node_xy[o_idx, 0]), float(node_xy[o_idx, 1]))

print("Computing shortest-path wave across ALL Romanian roads from Focșani...")
dist_map = nx.single_source_dijkstra_path_length(G, o_node, weight="w")

# Pre-organize road segments by class and activation time (weighted distance from Focșani)
base_segs = {k: [] for k in BASE_DRAW}
lit_segs  = {k: [] for k in LIT_DRAW}
lit_times = {k: [] for k in LIT_DRAW}

for u, v, data in G.edges(data=True):
    du = dist_map.get(u, np.inf)
    dv = dist_map.get(v, np.inf)
    if not (np.isfinite(du) and np.isfinite(dv)):
        continue
    arr = np.array(data["coords"])
    b = data["base"]
    if b in base_segs:
        base_segs[b].append(arr)
    if b in lit_segs:
        lit_segs[b].append(arr)
        lit_times[b].append(max(du, dv))

for k in lit_times:
    lit_times[k] = np.array(lit_times[k], dtype=float)

# Snap venues to nearest road node to get their exact arrival time along the road wave
vx = venues.geometry.x.values
vy = venues.geometry.y.values
v_dist = np.full(len(venues), np.inf)
node_dists_arr = np.array([dist_map.get(n, np.inf) for n in nodes_list])

# Chunked nearest-node lookup (fast & low memory)
CHUNK = 2000
for i0 in range(0, len(venues), CHUNK):
    i1 = min(i0 + CHUNK, len(venues))
    dx = node_xy[:, 0:1] - vx[i0:i1][None, :]
    dy = node_xy[:, 1:2] - vy[i0:i1][None, :]
    d2 = dx * dx + dy * dy
    nn = np.argmin(d2, axis=0)
    min_d = np.sqrt(d2[nn, np.arange(i1 - i0)])
    valid = min_d <= SNAP_MAX
    v_dist[i0:i1] = np.where(valid, node_dists_arr[nn] + min_d, np.inf)

del G

# Keep reachable venues
reach = np.isfinite(v_dist)
venues = venues[reach].reset_index(drop=True)
v_dist = v_dist[reach]

# Timeline mapping: smooth wave expansion across the entire country
max_wave = float(np.percentile(v_dist, 99.6))
all_edge_max = max(float(arr.max()) for arr in lit_times.values() if len(arr))
W_END = max(max_wave, all_edge_max)

TOTAL = MOVE + HOLD
t = np.clip(np.arange(TOTAL) / MOVE, 0.0, 1.0)
ease = 0.35 * t + 0.65 * (t * t * (3.0 - 2.0 * t))
wave_r = W_END * ease

trig_f = np.searchsorted(wave_r, v_dist)
v_xy = np.column_stack([venues.geometry.x.values, venues.geometry.y.values])
v_cat = venues["dish_cat"].values

DISHES = [
    ("mici",     "Mici",     AMBER),
    ("sarmale",  "Sarmale",  MAGENTA),
    ("papanași", "Papanași", CYAN),
]

dish_data = {}
for key, label, col in DISHES:
    m = (v_cat == key)
    dish_data[key] = {
        "label": label,
        "color": col,
        "rgb": to_rgb(col),
        "xy": v_xy[m],
        "trig": trig_f[m],
        "total": int(m.sum()),
    }
    print(f"  {label:9s}: {dish_data[key]['total']:,} stops")

total_road_km = sum(
    np.hypot(*np.diff(seg, axis=0).T).sum()
    for segs in base_segs.values() for seg in segs
) / 1000.0
print(f"Total connected road network: {total_road_km:,.0f} km")

# --------------------------------------------------------------- FIGURE (Day 17 Layout)
rings = load_boundary_3844(BOUNDARY_PATH)
allp = np.vstack(rings)
PAD = 22_000
X0, X1 = allp[:, 0].min() - PAD, allp[:, 0].max() + PAD
Y0, Y1 = allp[:, 1].min() - PAD, allp[:, 1].max() + PAD

fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
canvas = FigureCanvasAgg(fig)

# Main map box — exact Day 17 proportions
ax = fig.add_axes([0.06, 0.395, 0.88, 0.465], facecolor=BOX)
ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_aspect("equal", adjustable="box")
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_color("#2a2a4a")
    sp.set_linewidth(1)

# Subtle Stereo 70 coordinate grid (matching Day 17's subtle grid lines)
for gx in range(int(math.ceil(X0 / 100_000)) * 100_000, int(X1), 100_000):
    ax.axvline(gx, color="#171738", lw=0.6, zorder=0)
for gy in range(int(math.ceil(Y0 / 100_000)) * 100_000, int(Y1), 100_000):
    ax.axhline(gy, color="#171738", lw=0.6, zorder=0)

# Romania glowing outline (matching Day 17)
for r in rings:
    for lw, al in ((9, 0.05), (5, 0.10), (1.6, 0.90)):
        ax.plot(r[:, 0], r[:, 1], color=OUTLINE, lw=lw, alpha=al, zorder=6, solid_joinstyle="round")

# Base dim road network
for hw, (lw, al) in BASE_DRAW.items():
    if base_segs[hw]:
        ax.add_collection(LineCollection(
            base_segs[hw], colors="#232856", linewidths=lw, alpha=al, zorder=1
        ))

# Dynamic illuminated road collections (updated per frame by alpha mask)
lit_cols = {}
for hw, (lw, al) in LIT_DRAW.items():
    if lit_segs[hw]:
        n_s = len(lit_segs[hw])
        rgba = np.zeros((n_s, 4), dtype=float)
        rgba[:, :3] = to_rgb("#7b8cff")
        lc = LineCollection(lit_segs[hw], colors=rgba, linewidths=lw, zorder=2)
        ax.add_collection(lc)
        lit_cols[hw] = (lc, rgba, al, lit_times[hw])

# Faint background dots for all venues before the wave hits
ax.scatter(v_xy[:, 0], v_xy[:, 1], s=1.5, color="#4f588a", alpha=0.18, lw=0, zorder=2)

# Origin (Focșani) + reference cities
glowfx = [pe.withStroke(linewidth=2.5, foreground=BG)]
ax.scatter(*o_xy, s=55, color="white", edgecolor=AMBER, linewidth=1.8, zorder=11)
ax.text(o_xy[0] + 10_000, o_xy[1] - 11_000, "FOCȘANI", color="white", fontsize=9.5,
        fontweight="bold", path_effects=glowfx, zorder=12)
origin_ring = ax.scatter([], [], s=[], facecolors="none", edgecolors=[AMBER], linewidths=1.5, zorder=10)

for cname, clo, cla, (dx, dy), ha in CITIES:
    cpt = gpd.GeoSeries([Point(clo, cla)], crs=4326).to_crs(CRS).iloc[0]
    ax.plot(cpt.x, cpt.y, "o", color="white", ms=4.2, mec=BG, mew=1.2, zorder=9)
    ax.text(cpt.x + dx, cpt.y + dy, cname, color="white", fontsize=8.2,
            ha=ha, va="center", path_effects=glowfx, zorder=9)

# Per-dish scatter layers (persistent glow + core + pulse ring)
dish_artists = {}
for key, label, col in DISHES:
    glow = ax.scatter([], [], s=55, color=col, alpha=0.16, lw=0, zorder=4)
    core = ax.scatter([], [], s=7,  color=col, alpha=0.92, lw=0, zorder=5)
    ring = ax.scatter([], [], s=[], facecolors="none", edgecolors=[col], linewidths=1.4, zorder=7)
    dish_artists[key] = (glow, core, ring)

# Header — exact Day 17 positions & typography
fig.text(0.06, 0.93, "The Romanian Taste Wave", color="white", fontsize=26, fontweight="bold")
fig.text(0.06, 0.895, "Mici, sarmale & papanași reached along all roads from Focșani", color=CYAN, fontsize=14)

# Legend strip directly below map box (matching Day 17's legend strip at y=0.366)
leg_x = [0.06, 0.22, 0.39]
for lx, (key, label, col) in zip(leg_x, DISHES):
    fig.text(lx, 0.366, "●", color=col, fontsize=11, va="center")
    fig.text(lx + 0.022, 0.366, label, color="white", fontsize=9.5, fontweight="bold", va="center")
fig.text(0.94, 0.366, f"{total_road_km:,.0f} km of mapped roads", color=GREY, fontsize=8.5, ha="right", va="center")

# Bottom-left telemetry panel (matching Day 17's DATE / THE RACE panel)
fig.text(0.06, 0.322, "ROAD WAVE REACH", color=GREY, fontsize=10, fontweight="bold")
t_km = fig.text(0.06, 0.282, "0 km", color="white", fontsize=32, fontweight="bold")
t_sub = fig.text(0.06, 0.255, "expanding outward from Focșani", color=MAGENTA, fontsize=10.5)

fig.text(0.06, 0.215, "CULINARY STOPS", color=GREY, fontsize=10, fontweight="bold")
stat_rows = {}
for i, (key, label, col) in enumerate(DISHES):
    y = 0.185 - i * 0.031
    fig.text(0.06, y, "●", color=col, fontsize=11, va="center")
    fig.text(0.085, y, label, color="white", fontsize=12.5, va="center")
    stat_rows[key] = fig.text(0.38, y, "0", color=col, fontsize=15, fontweight="bold",
                              ha="right", va="center", family="monospace")

t_lead = fig.text(0.06, 0.092, "", color=AMBER, fontsize=10.5, fontweight="bold")

# Bottom-right: Running Leader Bar Chart (matching Day 17's chart box at [0.52, 0.085, 0.42, 0.215])
fig.text(0.52, 0.322, "RUNNING LEADER", color=GREY, fontsize=10, fontweight="bold")
bx = fig.add_axes([0.52, 0.085, 0.42, 0.215], facecolor=BOX)
for sp in bx.spines.values():
    sp.set_color("#2a2a4a")
    sp.set_linewidth(1)
bx.set_ylim(-0.55, 2.55)
bx.set_yticks([0, 1, 2])
bx.set_yticklabels(["3rd", "2nd", "1st"], color=GREY, fontsize=8.5, fontweight="bold")
bx.tick_params(axis="y", length=0)
bx.tick_params(axis="x", colors=GREY, labelsize=8, length=2)
bx.grid(axis="x", color="#171738", lw=0.6)
max_cat = max(d["total"] for d in dish_data.values())
bx.set_xlim(0, max(10, int(max_cat * 1.24)))

bars = bx.barh([2, 1, 0], [0, 0, 0], height=0.52, color=[AMBER, MAGENTA, CYAN], edgecolor="none", zorder=3)
bar_labels = [
    bx.text(0, y, "", color="white", fontsize=9.5, fontweight="bold", va="center", ha="left", zorder=4)
    for y in (2, 1, 0)
]

# Credits — bottom-left aligned cleanly with the 0.06 left margin (Day 17 exact style)
fig.text(0.06, 0.045, "#30DayMapChallenge | Day 23: Senses: Taste | Map: Sara AHARON", color=GREY, fontsize=8.5)
fig.text(0.06, 0.030, "Data: © OpenStreetMap contributors (Geofabrik) · Boundary: GADM 4.1 · CRS: Stereo 70 (EPSG:3844)",
         color=GREY, fontsize=7.5)


# --------------------------------------------------------------- FRAME RENDER
def render_frame(f):
    wr = wave_r[f]

    # Illuminate road network up to wave radius
    for hw, (lc, rgba, max_al, t_arr) in lit_cols.items():
        # Smooth fade-in across the wavefront edge
        alpha = np.clip((wr - t_arr + 12_000) / 12_000, 0.0, 1.0) * max_al
        rgba[:, 3] = alpha
        lc.set_colors(rgba)

    # Update dish points, pulse rings & counts
    counts = []
    for key, label, col in DISHES:
        d = dish_data[key]
        glow, core, ring = dish_artists[key]
        active = d["trig"] <= f
        n = int(active.sum())
        pts = d["xy"][active] if n else np.empty((0, 2))
        glow.set_offsets(pts)
        core.set_offsets(pts)
        stat_rows[key].set_text(f"{n:,}")
        counts.append((n, key, label, col))

        age = f - d["trig"]
        live = (age >= 0) & (age < RING)
        if live.any():
            pr = age[live] / RING
            ring.set_offsets(d["xy"][live])
            ring.set_sizes(25 + 680 * (pr ** 0.7))
            ring.set_edgecolors([(*d["rgb"], float(a)) for a in (1.0 - pr) ** 1.5])
        else:
            ring.set_offsets(np.empty((0, 2)))

    # Origin pulse while wave is expanding
    if f < MOVE:
        ph = (f % 45) / 45.0
        origin_ring.set_offsets([o_xy])
        origin_ring.set_sizes([55 + 1600 * ph])
        origin_ring.set_edgecolors([(*to_rgb(AMBER), 1.0 - ph)])
    else:
        origin_ring.set_offsets(np.empty((0, 2)))

    # Update telemetry text
    eff_km = int(round(wr / 1000.0))
    t_km.set_text(f"{eff_km:,} km")
    pct_roads = min(100, int(round(100.0 * wr / max(W_END, 1.0))))
    t_sub.set_text(f"{pct_roads}% of road network reached" if f < MOVE else "all Romanian roads reached")

    # Sort dishes by count for the running leader bar chart
    counts.sort(key=lambda item: (-item[0], item[1]))
    top_n, _, top_label, top_col = counts[0]
    if top_n > 0:
        t_lead.set_text(f"{top_label} leads with {top_n:,} stops")
        t_lead.set_color(top_col)
    else:
        t_lead.set_text("Departing Focșani...")
        t_lead.set_color(GREY)

    pad_txt = max_cat * 0.025
    for idx, (n, key, label, col) in enumerate(counts):
        bar = bars[idx]
        bar.set_width(n)
        bar.set_color(col)
        lbl = bar_labels[idx]
        lbl.set_x(n + pad_txt)
        lbl.set_text(f"{label} ({n:,})")
        lbl.set_color(col)

    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()


if __name__ == "__main__":
    print(f"Rendering {TOTAL} frames at 1080x1350...")
    writer = imageio.get_writer(
        OUT, fps=FPS, codec="libx264", macro_block_size=None,
        pixelformat="yuv420p", ffmpeg_params=["-crf", "18"]
    )
    for f in range(TOTAL):
        writer.append_data(render_frame(f))
        if f % 30 == 0:
            print(f"  frame {f}/{TOTAL}", flush=True)
    writer.close()
    print("DONE ->", OUT)

"""
#30DayMapChallenge | Day 21: OpenStreetMap
Focșani's Hidden Layer - the OSM tags normal maps never render.

Run:   python day21_hidden_layer.py            (downloads OSM data, cached to osm_focsani.json)
Test:  python day21_hidden_layer.py --demo     (synthetic data, no internet)
Out:   day21_hidden_layer.mp4  (1080x1350)

Deps: pip install numpy matplotlib imageio imageio-ffmpeg shapely requests
"""
import sys, json, os, time
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import linemerge, polygonize, unary_union

AUTHOR = "Sara AHARON"
DAY, THEME = 21, "OpenStreetMap"
CACHE = "osm_focsani.json"
OUT = "day21_hidden_layer.mp4"
W, H, DPI, FPS = 1080, 1350, 100, 30
BBOX = (45.655, 27.130, 45.735, 27.240)  # S, W, N, E (fallback + query box)
LAT0 = 45.6967
KX = np.cos(np.radians(LAT0))

BG = "#0b0d1a"
LAYERS = [  # key, label, color, size
    ("tree",     "trees",         "#39ff88", 5),
    ("bench",    "benches",       "#00e5ff", 14),
    ("waste",    "waste bins",    "#ff3df2", 14),
    ("crossing", "crossings",     "#ffd23d", 8),
    ("lamp",     "street lamps",  "#ff9a3c", 7),
]

MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def queries():
    s, w, n, e = BBOX
    b = f"({s},{w},{n},{e})"
    nodes = f"""[out:json][timeout:120];
(
 node["natural"="tree"]{b};
 node["amenity"="bench"]{b};
 node["amenity"="waste_basket"]{b};
 node["highway"="crossing"]{b};
 node["highway"="street_lamp"]{b};
);
out body;"""
    ways = f"""[out:json][timeout:120];
way["highway"]["highway"!~"^(footway|path|steps|service|track|cycleway)$"]{b};
out geom;"""
    bound = """[out:json][timeout:60];
rel["boundary"="administrative"]["admin_level"="8"]["name"="Focșani"];
out geom;"""
    return [nodes, ways, bound]


HEADERS = {
    "User-Agent": "30DayMapChallenge-Day21/1.0 (student project; Focsani OSM map)",
    "Accept": "*/*",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}


def run_query(q, name):
    part = f"osm_{name}.json"
    if os.path.exists(part):
        return json.load(open(part, encoding="utf-8"))
    for rnd in range(3):
        for url in MIRRORS:
            try:
                print(f"[{name}] {url}")
                r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=200)
                if r.status_code == 429:
                    print("   429, waiting 25s"); time.sleep(25); continue
                r.raise_for_status()
                data = r.json()
                json.dump(data, open(part, "w", encoding="utf-8"))
                return data
            except Exception as ex:
                print("   failed:", ex)
                time.sleep(5)
        print("round failed, waiting 40s"); time.sleep(40)
    return None


def fetch():
    els = []
    for q, name in zip(queries(), ["nodes", "roads", "boundary"]):
        d = run_query(q, name)
        if d is None:
            if name == "nodes":
                sys.exit("Nodes query failed on all mirrors. Wait a few min and rerun (finished parts are cached).")
            print(f"WARNING: {name} skipped")
            continue
        els += d["elements"]
        time.sleep(3)
    return {"elements": els}


def proj(lon, lat):
    return (np.asarray(lon) - 27.1865) * KX, np.asarray(lat) - LAT0


def parse(data):
    pts = {k: [] for k, *_ in LAYERS}
    roads, boundary_lines = [], []
    for el in data["elements"]:
        t = el.get("tags", {})
        if el["type"] == "node":
            key = ("tree" if t.get("natural") == "tree" else
                   "bench" if t.get("amenity") == "bench" else
                   "waste" if t.get("amenity") == "waste_basket" else
                   "crossing" if t.get("highway") == "crossing" else
                   "lamp" if t.get("highway") == "street_lamp" else None)
            if key:
                pts[key].append((el["lon"], el["lat"]))
        elif el["type"] == "way" and "geometry" in el:
            roads.append([(g["lon"], g["lat"]) for g in el["geometry"]])
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    boundary_lines.append(LineString([(g["lon"], g["lat"]) for g in m["geometry"]]))
    poly = None
    if boundary_lines:
        polys = list(polygonize(unary_union(linemerge(boundary_lines))))
        if polys:
            poly = max(polys, key=lambda p: p.area)
    return pts, roads, poly


def demo_data():
    rng = np.random.default_rng(7)
    pts = {}
    for k, _, _, _ in LAYERS:
        n = {"tree": 2500, "bench": 300, "waste": 250, "crossing": 900, "lamp": 1800}[k]
        lon = rng.normal(27.1865, 0.022, n)
        lat = rng.normal(LAT0, 0.016, n)
        pts[k] = list(zip(lon, lat))
    roads = []
    for _ in range(500):
        x, y = rng.normal(27.1865, 0.025), rng.normal(LAT0, 0.018)
        a = rng.uniform(0, np.pi)
        roads.append([(x, y), (x + 0.01 * np.cos(a), y + 0.007 * np.sin(a))])
    poly = Point(27.1865, LAT0).buffer(0.035).intersection(
        Polygon([(27.14, 45.66), (27.23, 45.66), (27.23, 45.73), (27.14, 45.73)]))
    return pts, roads, poly


def main():
    demo = "--demo" in sys.argv
    pts, roads, poly = demo_data() if demo else parse(fetch())

    # clip points to city boundary if we have it
    if poly is not None:
        for k in pts:
            pts[k] = [p for p in pts[k] if poly.contains(Point(p))]
    rng = np.random.default_rng(21)
    P = {}
    for k in pts:
        a = np.array(pts[k]) if len(pts[k]) else np.zeros((0, 2))
        if len(a):
            x, y = proj(a[:, 0], a[:, 1])
            a = np.c_[x, y]
            a = a[rng.permutation(len(a))]
        P[k] = a
    counts = {k: len(P[k]) for k in P}
    print(counts)

    # roads as segments (clipped by bbox only; very dim)
    segs = []
    for r in roads:
        if poly is not None:
            clipped = LineString(r).intersection(poly)
            parts = [clipped] if clipped.geom_type == "LineString" else list(getattr(clipped, "geoms", []))
            parts = [p for p in parts if p.geom_type == "LineString" and not p.is_empty]
        else:
            parts = [LineString(r)]
        for p in parts:
            xs, ys = p.xy
            x, y = proj(list(xs), list(ys))
            segs.append(np.c_[x, y])
    if poly is not None:
        bx, by = proj(*poly.exterior.xy)
        outline = np.c_[bx, by]
        minx, maxx, miny, maxy = bx.min(), bx.max(), by.min(), by.max()
    else:
        allp = np.vstack([P[k] for k in P if len(P[k])])
        minx, miny = allp.min(0); maxx, maxy = allp.max(0)
        outline = None
    pad = 0.03 * max(maxx - minx, maxy - miny)

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.72], facecolor=BG)
    ax.set_aspect("equal"); ax.axis("off")
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    half = max(maxx - minx, maxy - miny) / 2 + pad
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half)

    # base
    road_lc = LineCollection(segs, colors="#2a3052", linewidths=0.35, alpha=0.0)
    ax.add_collection(road_lc)
    if outline is not None:
        out_line, = ax.plot(outline[:, 0], outline[:, 1], color="#3b4478", lw=0.8, alpha=0.0)

    # glow layers: 3 passes each
    art = {}
    for k, label, col, s in LAYERS:
        passes = []
        for mult, al in ((9, 0.05), (3.5, 0.14), (1, 0.95)):
            sc = ax.scatter([], [], s=s * mult, c=col, alpha=al, linewidths=0, zorder=3)
            passes.append((sc, al))
        art[k] = passes

    # text (Day 12 layout standard)
    fig.text(0.06, 0.93, "Focșani's Hidden Layer", color="white", size=26, weight="bold")
    sub = fig.text(0.06, 0.895, "What OpenStreetMap knows that maps never show", color="#00e5ff", size=14)
    fig.text(0.06, 0.04, f"#30DayMapChallenge | Day {DAY}: {THEME} | {AUTHOR}",
             color="#8a93b0", size=8)
    fig.text(0.06, 0.022, "Data: © OpenStreetMap contributors (ODbL) via Overpass API",
             color="#8a93b0", size=8)

    # legend with live counters
    leg = {}
    for i, (k, label, col, s) in enumerate(LAYERS):
        yy = 0.105 - 0.0 * i
        xx = 0.06 + i * 0.185
        fig.patches.append(plt.Circle((xx, 0.088), 0.0, transform=fig.transFigure))
        fig.text(xx, 0.098, "●", color=col, size=13, alpha=0.9)
        leg[k] = fig.text(xx + 0.022, 0.0985, "", color="#c9d0ea", size=10)
        fig.text(xx + 0.022, 0.0815, label, color="#6f7899", size=8)

    # timeline
    F_BASE = 30
    F_LAYER = 70
    F_HOLD = 90
    total = F_BASE + F_LAYER * len(LAYERS) + F_HOLD
    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1, pixelformat="yuv420p")
    ease = lambda t: 1 - (1 - min(max(t, 0), 1)) ** 3

    for f in range(total):
        # base fade-in
        b = ease(f / F_BASE)
        road_lc.set_alpha(0.9 * b)
        if outline is not None:
            out_line.set_alpha(b)

        for i, (k, label, col, s) in enumerate(LAYERS):
            start = F_BASE + i * F_LAYER
            t = (f - start) / (F_LAYER * 0.8)
            frac = ease(t)
            n = int(len(P[k]) * frac)
            for sc, al in art[k]:
                sc.set_offsets(P[k][:n] if n else np.zeros((0, 2)))
                # gentle breathing once all on
                breathe = 1.0
                if f >= F_BASE + F_LAYER * len(LAYERS):
                    breathe = 0.85 + 0.15 * np.sin((f / 18.0) + i)
                sc.set_alpha(min(1, al * breathe))
            leg[k].set_text(f"{n:,}" if f >= start else "")

        # subtitle swaps as layers appear
        cur = min(max((f - F_BASE) // F_LAYER, 0), len(LAYERS) - 1)
        if f >= F_BASE + F_LAYER * len(LAYERS):
            sub.set_text(f"{sum(counts.values()):,} invisible objects, mapped by volunteers")
            sub.set_color("#ff3df2")
        elif f >= F_BASE:
            sub.set_text(f"Layer {cur + 1}/{len(LAYERS)}: {LAYERS[cur][1]}")
            sub.set_color(LAYERS[cur][2])

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(frame)
    writer.close()
    print("saved", OUT)


if __name__ == "__main__":
    main()

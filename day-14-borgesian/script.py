"""
30 Day Map Challenge - Day 14: Borgesian Map (VIDEO)
"The map rotting in the territory": Borges ends "On Exactitude in Science" with the Tattered Ruins
of the map still lying in the desert. Here the map of Romania is a grid of 8 km tiles. As OpenStreetMap
places that are ruined, abandoned or disused appear (historic=ruins, ruins:*, abandoned[:*], disused:*),
the tiles around them fray and fade, until the territory has outlived its own map.
Bottom: live ranking of the counties with the most recorded places.

Install once:  pip install geopandas matplotlib shapely rasterio imageio imageio-ffmpeg numpy
First run downloads the OSM data from the Overpass API in 20 tiles (a few minutes, cached in ruins_cache/).
Tip: set PREVIEW_ONLY = True first -> renders ONE frame (day14_preview.png) in seconds.
"""
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from rasterio.features import rasterize
from rasterio.transform import from_origin

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
CELL_KM = 8                  # size of one "map tile" (km)
DECAY_K = 2.0                # a tile with this many places is ~37% intact (lower = faster decay)
EASE = 1.8                   # >1 = places appear slowly at first, then faster
TOP_N = 7                    # counties shown in the ranking
SECONDS = 20                 # length of the reveal
FPS = 30
HOLD_START, HOLD_END = 1.5, 4.0   # seconds frozen at the start (intact map) / end (the ruins)
SIZE_W, SIZE_H = 1080, 1350  # 4:5 portrait, same as Day 5
PREVIEW_ONLY = False         # True = one frame only -> day14_preview.png
OUT_MP4 = "day14_borgesian.mp4"
OUT_PNG = "day14_borgesian_final.png"
CACHE_DIR = Path("ruins_cache")
BBOX = (43.6, 20.2, 48.3, 29.8)   # S, W, N, E around Romania
TILES = (5, 4)                    # Overpass download split: columns x rows
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
CRS_RO = "EPSG:3844"
CAT_NAME = ["historic ruins", "abandoned", "disused"]
CAT_COL = ["#ffb347", "#ff2e97", "#00e5ff"]
MAP_CMAP = LinearSegmentedColormap.from_list("integrity", [BG, "#1d2540", "#33406a"])


# ---------- OSM download ----------
def overpass_query(s, w, n, e):
    bb = f"{s:.4f},{w:.4f},{n:.4f},{e:.4f}"
    return (
        "[out:json][timeout:180];\n"
        "(\n"
        f'  nwr["historic"="ruins"]({bb});\n'
        f'  nwr[~"^(abandoned|disused|ruins)(:.*)?$"~"."]({bb});\n'
        ");\n"
        "out center tags;"
    )


def fetch_tile(i, s, w, n, e):
    path = CACHE_DIR / f"tile_{i:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    CACHE_DIR.mkdir(exist_ok=True)
    data = urllib.parse.urlencode({"data": overpass_query(s, w, n, e)}).encode()
    for attempt in range(6):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            print(f"  tile {i + 1}: downloading (try {attempt + 1}) ...")
            req = urllib.request.Request(
                url, data=data, headers={"User-Agent": "30DayMapChallenge-day14 (student project)"})
            with urllib.request.urlopen(req, timeout=240) as r:
                payload = json.loads(r.read().decode("utf-8"))
            remark = str(payload.get("remark", "")).lower()
            if "error" in remark or "timed out" in remark or "out of memory" in remark:
                raise ValueError(remark)          # partial answer -> retry
            path.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(2)                          # be polite to the public server
            return payload
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            wait = 10 * (attempt + 1)
            print(f"    failed ({exc}); retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Could not download tile {i + 1} from Overpass. Try again later.")


def classify(tags):
    """0 = ruins, 1 = abandoned, 2 = disused (first match wins). None = not interesting."""
    keys = list(tags)
    if tags.get("historic") == "ruins" or any(k == "ruins" or k.startswith("ruins:") for k in keys):
        return 0
    if any((k == "abandoned" and tags[k] != "no") or k.startswith("abandoned:") for k in keys):
        return 1
    if any(k.startswith("disused:") for k in keys):
        return 2
    return None


def load_places():
    s, w, n, e = BBOX
    nx_t, ny_t = TILES
    lats = np.linspace(s, n, ny_t + 1)
    lons = np.linspace(w, e, nx_t + 1)
    seen, t = {}, 0
    for a in range(ny_t):
        for b in range(nx_t):
            payload = fetch_tile(t, lats[a], lons[b], lats[a + 1], lons[b + 1])
            t += 1
            for el in payload.get("elements", []):
                key = (el.get("type"), el.get("id"))
                if key in seen:
                    continue
                c = el.get("center", {})
                la, lo = el.get("lat", c.get("lat")), el.get("lon", c.get("lon"))
                cat = classify(el.get("tags", {}))
                if la is None or lo is None or cat is None:
                    continue
                seen[key] = (lo, la, cat)
    arr = np.array(list(seen.values()), dtype="float64")
    print(f"OSM places downloaded: {len(arr):,}")
    return arr[:, 0], arr[:, 1], arr[:, 2].astype(int)


# 1. Boundaries --------------------------------------------------------
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties_full = gpd.read_file(f).to_crs(CRS_RO)
country = counties_full.dissolve().geometry.iloc[0]
counties_d = counties_full.geometry.simplify(150)       # lighter geometry = faster frames
country_line = gpd.GeoSeries([country], crs=CRS_RO).simplify(150).boundary

# 2. Places: OSM -> inside Romania -> county -> random reveal order -----------
lon, lat, cat = load_places()
pts = gpd.GeoDataFrame({"cat": cat}, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(CRS_RO)
joined = gpd.sjoin(pts, counties_full[["NAME_1", "geometry"]], predicate="within", how="inner")
joined = joined[~joined.index.duplicated()]
joined = joined.iloc[np.random.default_rng(14).permutation(len(joined))]   # fixed seed = same video every run
X = joined.geometry.x.to_numpy()
Y = joined.geometry.y.to_numpy()
XY = np.column_stack([X, Y])
CAT = joined["cat"].to_numpy().astype(int)
county_names, county_code = np.unique(joined["NAME_1"].to_numpy(), return_inverse=True)
nc = len(county_names)
N = len(X)
print(f"Places inside Romania: {N:,}  ({', '.join(f'{CAT_NAME[c]}: {(CAT == c).sum():,}' for c in range(3))})")
base_rgb = np.array([to_rgb(c) for c in CAT_COL])[CAT]

# 3. The "map": a grid of tiles ------------------------------------------------
minx, miny, maxx, maxy = country.bounds
cell = CELL_KM * 1000
gx = int(math.ceil((maxx - minx) / cell))
gy = int(math.ceil((maxy - miny) / cell))
xe = minx + cell * np.arange(gx + 1)
ye = miny + cell * np.arange(gy + 1)
tf = from_origin(minx, miny + gy * cell, cell, cell)
in_cells = rasterize([(country, 1)], out_shape=(gy, gx), transform=tf, fill=0,
                     dtype="uint8", all_touched=True).astype(bool)[::-1]   # row 0 = south
n_cells = int(in_cells.sum())
col = np.clip(((X - minx) // cell).astype(int), 0, gx - 1)
row = np.clip(((Y - miny) // cell).astype(int), 0, gy - 1)
cell_id = row * gx + col

# 4. Timeline ---------------------------------------------------------------------
n_ramp = int(SECONDS * FPS)
ramp = np.round(N * np.linspace(0, 1, n_ramp) ** EASE).astype(int)
m_sched = np.concatenate([np.zeros(int(HOLD_START * FPS), dtype=int), ramp,
                          np.full(int(HOLD_END * FPS), N, dtype=int)])
appear = np.searchsorted(m_sched, np.arange(1, N + 1), side="left")   # frame at which each place shows up
final_tot = np.bincount(county_code, minlength=nc)
xmax = final_tot.max() * 1.18

# 5. Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(SIZE_W / 100, SIZE_H / 100), dpi=100)
fig.patch.set_facecolor(BG)

ax = fig.add_axes([0.125, 0.325, 0.75, 0.60])          # 810 x 810 px square
ax.set_facecolor(BG)
mesh = ax.pcolormesh(xe, ye, np.ma.masked_where(~in_cells, np.ones((gy, gx))), cmap=MAP_CMAP,
                     vmin=0, vmax=1, edgecolors=BG, linewidth=0.3, zorder=2)
counties_d.boundary.plot(ax=ax, color=LINE, linewidth=0.5, alpha=0.7, zorder=3)
country_line.plot(ax=ax, color="#4a5578", linewidth=1.0, zorder=3)
halo = ax.scatter([], [], s=[], linewidths=0, zorder=4)
core = ax.scatter([], [], s=[], linewidths=0, zorder=5)
cx, cy, half = (minx + maxx) / 2, (miny + maxy) / 2, max(maxx - minx, maxy - miny) / 2 * 1.04
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.set_aspect("equal")
ax.set_axis_off()

axp = fig.add_axes([0.19, 0.105, 0.75, 0.165])
axp.set_facecolor(BG)
axp.set_xlim(0, xmax)
axp.set_ylim(-0.6, TOP_N - 0.4)
for sp in axp.spines.values():
    sp.set_visible(False)
axp.set_xticks([])
axp.tick_params(axis="y", length=0, colors="#c7cde0", labelsize=10)
bar_sets = [axp.barh(np.arange(TOP_N), np.zeros(TOP_N), height=0.62, color=CAT_COL[c], left=0)
            for c in range(3)]
cnt_txt = [axp.text(0, y, "", color="#c7cde0", fontsize=10, va="center") for y in range(TOP_N)]

fig.text(0.06, 0.968, "The map rotting in the territory", color="white",
         fontsize=22, fontweight="bold", ha="left", va="center")
fig.text(0.06, 0.943, "Ruined, abandoned and disused places in Romania, as recorded in OpenStreetMap",
         color="#00e5ff", fontsize=13, ha="left", va="center")
txt_phase = fig.text(0.06, 0.303, "", color="white", fontsize=17, fontweight="bold",
                     ha="left", va="center")
txt_count = fig.text(0.94, 0.303, "", color="#00e5ff", fontsize=17, ha="right", va="center")
fig.text(0.06, 0.278, "Counties with the most recorded places", color="#8a93b0", fontsize=9,
         ha="left", va="bottom")
for x_leg, c in zip((0.06, 0.28, 0.46), range(3)):
    fig.text(x_leg, 0.082, "●", color=CAT_COL[c], fontsize=10, ha="left", va="center")
    fig.text(x_leg + 0.022, 0.082, CAT_NAME[c], color="#c7cde0", fontsize=9, ha="left", va="center")
fig.text(0.06, 0.012,
         f"#30DayMapChallenge | Day 14: Borgesian Map | {AUTHOR}\n"
         "Data: © OpenStreetMap contributors (Overpass API), GADM 4.1 | tags: historic=ruins, ruins:*, abandoned[:*], disused:*\n"
         f"Lines and areas are shown at their centre point. A {CELL_KM} km tile frays as OSM records places inside it.\n"
         "After Borges, “On Exactitude in Science”: the map outlives the Empire and rots in the desert.",
         color="#8a93b0", fontsize=8, ha="left", va="bottom", linespacing=1.4)


def draw_frame(f):
    """f = index into the timeline (m_sched[f] places are on the map)."""
    n = int(m_sched[f])

    # the map: tiles fade as places accumulate inside them
    cell_cnt = np.bincount(cell_id[:n], minlength=gx * gy)
    integrity = np.exp(-cell_cnt / DECAY_K).reshape(gy, gx)
    mesh.set_array(np.ma.masked_where(~in_cells, integrity))
    frayed = 100 * ((cell_cnt.reshape(gy, gx) > 0) & in_cells).sum() / n_cells

    # the ruins: each place flashes when it appears, then settles to an ember
    if n == 0:
        core.set_visible(False)
        halo.set_visible(False)
    else:
        core.set_visible(True)
        halo.set_visible(True)
        age = (f - appear[:n]).astype(float)
        fl = np.exp(-age / (0.45 * FPS))
        rgb = base_rgb[:n] + (1 - base_rgb[:n]) * 0.75 * fl[:, None]
        for coll, size0, size1, a0, a1 in ((halo, 12, 140, 0.07, 0.28), (core, 2.6, 38, 0.78, 0.22)):
            coll.set_offsets(XY[:n])
            coll.set_sizes(size0 + size1 * fl)
            coll.set_facecolors(np.column_stack([rgb, a0 + a1 * fl]))

    # ranking of counties
    cnt = np.bincount(county_code[:n] * 3 + CAT[:n], minlength=nc * 3).reshape(nc, 3)
    tot = cnt.sum(1)
    top = np.argsort(-tot)[:TOP_N]
    left = np.zeros(TOP_N)
    for c in range(3):
        for r, ci in enumerate(top):
            rect = bar_sets[c].patches[TOP_N - 1 - r]
            rect.set_x(left[r])
            rect.set_width(cnt[ci, c])
        left += cnt[top, c]
    labels = []
    for y in range(TOP_N):
        ci = top[TOP_N - 1 - y]
        has = tot[ci] > 0
        labels.append(county_names[ci] if has else "")
        cnt_txt[y].set_text(f"{tot[ci]:,}" if has else "")
        cnt_txt[y].set_position((tot[ci] + xmax * 0.012, y))
    axp.set_yticks(range(TOP_N))
    axp.set_yticklabels(labels)

    frac = n / N
    txt_phase.set_text("The map is whole" if n == 0 else
                       "The map begins to fray" if frac < 0.35 else
                       "The territory outlives its map" if frac < 0.999 else
                       "Tattered ruins")
    txt_count.set_text(f"{n:,} places  ·  {frayed:.0f}% of the map frayed")


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# 6. Render ---------------------------------------------------------------
if PREVIEW_ONLY:
    draw_frame(int(np.searchsorted(m_sched, 0.6 * N)))
    fig.savefig("day14_preview.png", facecolor=BG, dpi=100)
    print("Saved day14_preview.png  (set PREVIEW_ONLY = False for the full video)")
    raise SystemExit

with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as w:
    for f in range(len(m_sched)):
        draw_frame(f)
        w.append_data(grab())
        if f % 30 == 0:
            print(f"frame {f + 1}/{len(m_sched)}")
fig.savefig(OUT_PNG, facecolor=BG, dpi=100)
print(f"Saved {OUT_MP4} and {OUT_PNG}")

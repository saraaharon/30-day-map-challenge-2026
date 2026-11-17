"""
30 Day Map Challenge - Day 3: Polygons
STEP 2 of 2: animate driving time to the nearest airport (time zones = polygons).

Needs (same folder): grid_times.csv, airports.csv  (from day3_compute.py)
Install once:  pip install scipy imageio imageio-ffmpeg
Outputs:       day3_airports.mp4  (for LinkedIn)  +  day3_airports_final.png
"""
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
MAX_MIN = 180          # animation runs from 0 to this many minutes
BAND = 15              # color band width (minutes)
STEP = 1.5             # minutes per frame -> 120 frames
FPS = 30
HOLD_SECONDS = 3       # freeze on the last frame
SIZE_PX = 1080         # square video (LinkedIn-friendly)
FINE_KM = 2            # resolution of the smooth surface
SMOOTH = 2.0           # blur in grid cells (removes road-snapping noise; 0 = off)
MAKE_GIF = False       # True = also write a (big) GIF
OUT_MP4, OUT_PNG, OUT_GIF = "day3_airports.mp4", "day3_airports_final.png", "day3_airports.gif"
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
CRS_RO = "EPSG:3844"
CMAP = LinearSegmentedColormap.from_list(
    "reach", ["#00e5ff", "#3d8bff", "#8a4dff", "#ff2e97", "#ffb347"])

# 1. Data --------------------------------------------------------------
grid = pd.read_csv("grid_times.csv")
airports = pd.read_csv("airports.csv")
tcols = [c for c in grid.columns if c.startswith("t_")]
grid["tmin"] = grid[tcols].min(axis=1)           # minutes to nearest airport
grid = grid.dropna(subset=["tmin"])

gpts = gpd.GeoSeries(gpd.points_from_xy(grid.lon, grid.lat), crs="EPSG:4326").to_crs(CRS_RO)
gx, gy = gpts.x.values, gpts.y.values
apts = gpd.GeoSeries(gpd.points_from_xy(airports.lon, airports.lat),
                     crs="EPSG:4326").to_crs(CRS_RO)

zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f).to_crs(CRS_RO)
country = counties.dissolve().geometry.iloc[0]

# 2. Smooth surface of "minutes to nearest airport" -------------------
x0, y0, x1, y1 = country.bounds
step = FINE_KM * 1000
xs = np.arange(x0, x1, step)
ys = np.arange(y0, y1, step)
X, Y = np.meshgrid(xs, ys)
Z = griddata((gx, gy), grid.tmin.values, (X, Y), method="linear")
Zn = griddata((gx, gy), grid.tmin.values, (X, Y), method="nearest")
Z = np.where(np.isnan(Z), Zn, Z)                  # fill edges / unreachable gaps
if SMOOTH:
    Z = gaussian_filter(Z, sigma=SMOOTH)
inside = shapely.contains_xy(country, X, Y)
Z = np.where(inside, Z, np.nan)
n_inside = inside.sum()

# 3. Figure ------------------------------------------------------------
dpi = 100
fig = plt.figure(figsize=(SIZE_PX / dpi, SIZE_PX / dpi), dpi=dpi)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.03, 0.17, 0.94, 0.62])
ax.set_facecolor(BG)
counties.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=1)
ax.set_xlim(x0 - 15000, x1 + 15000)
ax.set_ylim(y0 - 15000, y1 + 15000)
ax.set_aspect("equal")
ax.set_axis_off()

# borders + airports drawn on top of the colored zones
counties.boundary.plot(ax=ax, color="#0b0f1a", linewidth=0.35, alpha=0.55, zorder=5)
counties.dissolve().boundary.plot(ax=ax, color="#8a93b0", linewidth=0.9, zorder=6)
for size, alpha in [(260, 0.05), (110, 0.10)]:
    apts.plot(ax=ax, color="white", markersize=size, alpha=alpha, linewidth=0, zorder=7)
apts.plot(ax=ax, color="white", markersize=14, edgecolor="#0b0f1a", linewidth=0.6, zorder=8)

# text
fig.text(0.05, 0.945, "How far is the nearest airport?", color="white",
         fontsize=26, fontweight="bold", ha="left", va="center")
fig.text(0.05, 0.905, f"Driving time to the nearest of {len(airports)} airports in Romania",
         color="#00e5ff", fontsize=14, ha="left", va="center")
counter = fig.text(0.95, 0.135, "", color="white", fontsize=30, fontweight="bold",
                   ha="right", va="center")
share = fig.text(0.95, 0.098, "", color="#c7cde0", fontsize=14, ha="right", va="center")
fig.text(0.05, 0.075,
         f"#30DayMapChallenge | Day 3: Polygons | {AUTHOR}\n"
         "All airports with an IATA code in OpenStreetMap (including small regional ones).\n"
         "Data: OpenStreetMap contributors, GADM 4.1 | Driving times: OpenRouteService",
         color="#8a93b0", fontsize=8.5, ha="left", va="bottom", linespacing=1.5)

# color legend (bottom-left under the map)
cax = fig.add_axes([0.05, 0.145, 0.45, 0.014])
cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CMAP)
cax.set_yticks([])
ticks = np.arange(0, MAX_MIN + 1, 60)
cax.set_xticks(ticks / MAX_MIN * 255)
cax.set_xticklabels([f"{t} min" for t in ticks], color="#c7cde0", fontsize=8)
cax.tick_params(length=0, pad=2)
for sp in cax.spines.values():
    sp.set_visible(False)

state = {"cs": None}


def draw_frame(T):
    if state["cs"] is not None:
        try:
            state["cs"].remove()
        except Exception:
            for c in state["cs"].collections:
                c.remove()
    full = np.arange(0, T, BAND)[1:] if T > BAND else np.array([])
    levels = np.unique(np.concatenate([[0], full, [T]]))
    if len(levels) >= 2 and T > 0:
        colors = [CMAP(min(lv / MAX_MIN, 1.0)) for lv in levels[1:]]
        state["cs"] = ax.contourf(X, Y, Z, levels=levels, colors=colors, zorder=3)
    counter.set_text(f"within {int(round(T))} min")
    pct = 100 * np.nansum(Z <= T) / n_inside
    share.set_text(f"{pct:.0f}% of Romania's area")


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# 4. Render ------------------------------------------------------------
times = list(np.arange(0, MAX_MIN + STEP, STEP))
frames = []
for i, T in enumerate(times):
    draw_frame(min(T, MAX_MIN))
    frames.append(grab())
    if i % 20 == 0:
        print(f"frame {i + 1}/{len(times)}")
last = frames[-1]
frames += [last] * int(HOLD_SECONDS * FPS)

fig.savefig(OUT_PNG, facecolor=BG, dpi=dpi)
print(f"Saved {OUT_PNG}")

with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as w:
    for fr in frames:
        w.append_data(fr)
print(f"Saved {OUT_MP4}")

if MAKE_GIF:
    from PIL import Image
    small = [Image.fromarray(f).resize((720, 720)) for f in frames[::2]]
    small[0].save(OUT_GIF, save_all=True, append_images=small[1:], duration=1000 // (FPS // 2),
                  loop=0, optimize=True)
    print(f"Saved {OUT_GIF}")

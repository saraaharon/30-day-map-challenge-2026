"""
30 Day Map Challenge - Day 6: Vintage (80s Neon Synthwave Edition)
Marea Neagra: an 80s neon-synthwave tourism poster of the Romanian Black Sea coast.

Install once:  pip install geopandas shapely pyproj matplotlib numpy pillow
Output:        day6_80s_neon_poster.png  (4:5 portrait, LinkedIn-friendly)
First run downloads Natural Earth country borders (~13 MB) and caches them.
"""
import logging
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import unary_union

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
OUT_PNG = Path("day6_80s_neon_poster.png").resolve()  # Absolute path avoids Windows OSError 22
DPI = 200  # 10.8 x 13.5 in  ->  2160 x 2700 px
TITLE_FONT = ["Impact", "Arial Narrow", "DejaVu Sans"]
BODY_FONT = ["Georgia", "Palatino Linotype", "DejaVu Serif"]
SUN_LON, SUN_LAT, SUN_R = 30.10, 45.05, 21000  # sunrise over the sea (radius in meters)
N_RAYS = 26
# ----------------------------

# 80s Synthwave Palette
NAVY, SEA, SEA_RAY = "#0a0a23", "#120836", "#1a0f4a"
CREAM, SAND, LAND2 = "#00ffff", "#ff007f", "#2b0a45"
ORANGE, RED, SUN, SUN_IN = "#ff007f", "#ff0055", "#ffb800", "#ffee00"

NE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_admin_0_countries.geojson"
)
NE_FILE = Path("ne_10m_admin_0_countries.geojson")
if not NE_FILE.exists():
  print("Downloading Natural Earth borders (once)...")
  urllib.request.urlretrieve(NE_URL, NE_FILE)

# local Transverse Mercator centered on the coast -> north is straight up
CRS = "+proj=tmerc +lat_0=44 +lon_0=29 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)

# 1. Data --------------------------------------------------------------
world = gpd.read_file(NE_FILE).cx[26.0:32.0, 42.0:47.0].to_crs(CRS)

# poster view: 4:5 portrait around the coast
_, y0 = tr.transform(29.05, 43.10)
_, y1 = tr.transform(29.05, 45.60)
cx, _ = tr.transform(29.05, 44.35)
H = y1 - y0
W = H * 0.8
xmin, xmax = cx - W / 2, cx + W / 2
pad = 70000
world = gpd.clip(world, box(xmin - pad, y0 - pad, xmax + pad, y1 + pad))

ro = world[world["ADMIN"] == "Romania"]
neigh = world[world["ADMIN"] != "Romania"]
land_all = unary_union(world.geometry).simplify(300)

# 2. Figure ------------------------------------------------------------
fig = plt.figure(figsize=(10.8, 13.5), dpi=DPI)
fig.patch.set_facecolor(SEA)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(SEA)
ax.set_xlim(xmin, xmax)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")

# sunburst rays over the sea
sx, sy = tr.transform(SUN_LON, SUN_LAT)
ang = np.linspace(0, 2 * np.pi, 2 * N_RAYS + 1)
R = 1.5e6
for k in range(N_RAYS):
  a0, a1 = ang[2 * k], ang[2 * k + 1]
  ax.add_patch(
      Polygon(
          [
              (sx, sy),
              (sx + R * np.cos(a0), sy + R * np.sin(a0)),
              (sx + R * np.cos(a1), sy + R * np.sin(a1)),
          ],
          closed=True,
          fc=SEA_RAY,
          ec="none",
          zorder=1,
      )
  )

# neon sun
for r, al in [(2.2, 0.15), (1.7, 0.25), (1.3, 0.35)]:
  ax.add_patch(
      Circle((sx, sy), SUN_R * r, fc=ORANGE, ec="none", alpha=al, zorder=2)
  )
ax.add_patch(Circle((sx, sy), SUN_R, fc=SUN, ec=CREAM, lw=3.0, zorder=3))
ax.add_patch(Circle((sx, sy), SUN_R * 0.62, fc=SUN_IN, ec="none", zorder=3))

# wave ripples that follow the coastline (Neon Cyan glow)
for d, lw, al in [
    (4000, 2.2, 0.90),
    (9000, 1.9, 0.70),
    (15000, 1.6, 0.50),
    (22000, 1.3, 0.32),
    (31000, 1.0, 0.18),
]:
  gpd.GeoSeries([land_all.buffer(d).boundary], crs=CRS).plot(
      ax=ax, color=CREAM, linewidth=lw, alpha=al, zorder=4
  )

# land (Synthwave dark purple & neon pink accents)
neigh.plot(ax=ax, color=LAND2, edgecolor="none", zorder=5)
ro.plot(ax=ax, color=SAND, edgecolor="none", zorder=5)  # Restored neon pink fill for Romania

# Thin dark blue contour for Romania to make it pop distinctively
ro.boundary.plot(
    ax=ax,
    color=NAVY,
    linewidth=1.2,
    zorder=5.5,
)

neigh.boundary.plot(
    ax=ax,
    color="#ff00ff",
    linewidth=0.7,
    linestyle=(0, (4, 3)),
    alpha=0.6,
    zorder=6,
)
ro.boundary.plot(
    ax=ax,
    color="#00ffff",
    linewidth=0.9,
    linestyle=(0, (4, 3)),
    alpha=0.8,
    zorder=6,
)
gpd.GeoSeries([land_all.boundary], crs=CRS).plot(
    ax=ax, color=CREAM, linewidth=3.2, zorder=7
)

# 3. Labels ------------------------------------------------------------
sea_fx = [pe.withStroke(linewidth=3.5, foreground=NAVY)]
land_fx = [pe.withStroke(linewidth=3.0, foreground="#120836")]


def pin(lat, lon, name, dx=10, dy=0, ha="left", size=13):
  x, y = tr.transform(lon, lat)
  ax.plot(x, y, "o", ms=8, mfc=RED, mec=CREAM, mew=1.8, zorder=9)
  ax.annotate(
      name,
      (x, y),
      xytext=(dx, dy),
      textcoords="offset points",
      color=CREAM,
      fontsize=size,
      fontfamily=BODY_FONT,
      fontweight="bold",
      ha=ha,
      va="center",
      path_effects=sea_fx,
      zorder=10,
  )


pin(45.155, 29.655, "SULINA", dx=-11, ha="right")
pin(44.265, 28.62, "MAMAIA", dx=11)
pin(44.16, 28.635, "CONSTANȚA", dx=11)
pin(44.05, 28.635, "EFORIE", dx=11)
pin(43.945, 28.635, "COSTINEȘTI", dx=11)
pin(43.82, 28.585, "MANGALIA", dx=11)
pin(43.75, 28.575, "VAMA VECHE", dx=11)
pin(44.55, 28.77, "HISTRIA", dx=11)

dx_, dy_ = tr.transform(29.38, 45.0)
ax.text(
    dx_,
    dy_,
    "DANUBE\nDELTA",
    color=CREAM,
    fontsize=17,
    fontfamily=TITLE_FONT,
    ha="center",
    va="center",
    linespacing=1.05,
    path_effects=land_fx,
    zorder=10,
)

# 4. Poster furniture --------------------------------------------------
banner_h = 0.185
ax.add_patch(
    Rectangle(
        (0, 0),
        1,
        banner_h,
        transform=ax.transAxes,
        fc="#0a0a23",
        ec="none",
        zorder=20,
    )
)
ax.add_patch(
    Rectangle(
        (0, banner_h),
        1,
        0.008,
        transform=ax.transAxes,
        fc=ORANGE,
        ec="none",
        zorder=20,
    )
)
ax.add_patch(
    Rectangle(
        (0, banner_h + 0.008),
        1,
        0.003,
        transform=ax.transAxes,
        fc=CREAM,
        ec="none",
        zorder=20,
    )
)

ax.text(
    0.5,
    0.105,
    "THE BLACK SEA",
    transform=ax.transAxes,
    color=CREAM,
    fontsize=52,
    fontfamily=TITLE_FONT,
    ha="center",
    va="center",
    zorder=21,
)
ax.text(
    0.5,
    0.035,
    f"#30DayMapChallenge | Day 6: Vintage (80s Neon Edition) | {AUTHOR}\n"
    "Data: Natural Earth. Style inspired by 1980s synthwave aesthetics.",
    transform=ax.transAxes,
    color="#a7a9be",
    fontsize=8.5,
    fontfamily=BODY_FONT,
    ha="center",
    va="center",
    linespacing=1.5,
    zorder=21,
)

ax.add_patch(
    Rectangle(
        (0.018, 0.012),
        0.964,
        0.976,
        transform=ax.transAxes,
        fc="none",
        ec=CREAM,
        lw=2.6,
        zorder=30,
    )
)

fig.savefig(OUT_PNG, facecolor=SEA, dpi=DPI)
print(f"Saved {OUT_PNG}")

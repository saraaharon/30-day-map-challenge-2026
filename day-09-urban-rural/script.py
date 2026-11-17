"""
30DayMapChallenge - Focsani sprawl animation
"Fields turning into housing": GHSL built-up epochs 1990 -> 2020

Each frame shows where the city has grown. Colors = WHEN a pixel became
built-up (cyan = already there in 1990 ... yellow = built by 2020). New
growth in each step flashes white before settling into its color.
The dark teal wash underneath is open land (fields / grass) that is
still waiting to be built on.

Setup (once):
    pip install earthengine-api numpy matplotlib pillow
    earthengine authenticate

Run:
    python focsani_sprawl_animation.py
Output: GIF + a PNG of the final frame.
"""

import math

import ee
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from PIL import Image

# ----------------------------------------------------------------- CONFIG ---
GEE_PROJECT = "advance-vector-464019-f8"

CENTER = (45.6965, 27.1865)      # Focsani center (lat, lon)
HALF_KM = 12                     # map covers 24 x 24 km
CRS = "EPSG:32635"               # UTM 35N (Focsani)
EPOCHS = [1990, 1995, 2000, 2005, 2010, 2015, 2020]   # GHSL observed epochs
BUILT_THRESHOLD = 5              # % of a 100 m cell that counts as "built"

DAY = 10
THEME = "Urban-rural"
AUTHOR = "Sara AHARON"
SOURCE = ("Data: GHSL built-up surface 1990-2020 (JRC GHS-BUILT-S R2023A), "
          "ESA WorldCover 2021 via Google Earth Engine")

FADE_STEPS = 5                   # frames per 5-year step (more = smoother)
DPI = 80                         # 80 -> 864x1080 GIF (lower = smaller file)
OUT_GIF = "focsani_sprawl_1990_2020.gif"
OUT_PNG = "focsani_sprawl_final_frame.png"

# Synthwave palette
BG = "#0d0221"
PINK = "#ff2a6d"
CYAN = "#05d9e8"
PURPLE = "#7b2ff7"
YELLOW = "#ffd319"
MUTED = "#8a93b0"
RED = PINK
FIELD = "#0b3d4f"                # open land wash

RAMP = LinearSegmentedColormap.from_list("age", [CYAN, PURPLE, PINK, YELLOW])

# --------------------------------------------------------- EARTH ENGINE -----
ee.Initialize(project=GEE_PROJECT)

# Square region in UTM, snapped to the 100 m grid
cx, cy = ee.Geometry.Point([CENTER[1], CENTER[0]]).transform(CRS, 1).coordinates().getInfo()
cx, cy = round(cx / 100) * 100, round(cy / 100) * 100
h = HALF_KM * 1000
region = ee.Geometry.Rectangle([cx - h, cy - h, cx + h, cy + h], proj=CRS, geodesic=False)


def fetch(img, band):
    """Pull one band over the region as a numpy array (row 0 = north)."""
    arr = img.select(band).sampleRectangle(region=region, defaultValue=0).get(band).getInfo()
    return np.array(arr, dtype=float)


print("Downloading GHSL epochs...")
built_raw = []
for y in EPOCHS:
    img = (ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{y}").select("built_surface")
           .resample("bilinear").reproject(crs=CRS, scale=100))
    built_raw.append(fetch(img, "built_surface"))     # m2 of built surface per 100 m cell

print("Downloading open-land mask...")
wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
open_land = (wc.eq(30).Or(wc.eq(40)).Or(wc.eq(50))          # grass, cropland, built
             .rename("open").toFloat()
             .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024)
             .reproject(crs=CRS, scale=100))
open_share = fetch(open_land, "open")

# Trim everything to a common shape
Hh = min([a.shape[0] for a in built_raw] + [open_share.shape[0]])
Ww = min([a.shape[1] for a in built_raw] + [open_share.shape[1]])
built_raw = [a[:Hh, :Ww] for a in built_raw]
open_share = np.clip(open_share[:Hh, :Ww], 0, 1)

n = len(EPOCHS)
built_pct = [np.clip(a / 100.0, 0, 100) for a in built_raw]
area_km2 = [float(np.nansum(a)) / 1e6 for a in built_raw]
print("Built-up area (km2):", [round(a, 1) for a in area_km2])

# First epoch in which each pixel becomes "built" (99 = never)
first = np.full((Hh, Ww), 99, dtype=int)
for i in reversed(range(n)):
    first[built_pct[i] >= BUILT_THRESHOLD] = i

# ------------------------------------------------------------------ PLOT ----
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "white"})

ext_x = Ww * 0.1 / 2   # km (100 m pixels)
ext_y = Hh * 0.1 / 2
extent = [-ext_x, ext_x, -ext_y, ext_y]

bg_rgb, field_rgb = np.array(to_rgb(BG)), np.array(to_rgb(FIELD))
base = bg_rgb[None, None, :] * (1 - open_share[..., None]) + field_rgb[None, None, :] * open_share[..., None]

fig = plt.figure(figsize=(10.8, 13.5), facecolor=BG, dpi=DPI)

# Title block (same layout as the other maps)
fig.text(0.06, 0.93, "Focșani - Fields to Façades",
         color="white", fontsize=26, fontweight="bold", ha="left")
fig.text(0.06, 0.895,
         f"Fields to housing: built-up growth {EPOCHS[0]}–{EPOCHS[-1]}, colored by when it was built",
         color=RED, fontsize=14, ha="left")

# Map
axm = fig.add_axes([0.06, 0.16, 0.88, 0.70], facecolor=BG)
axm.imshow(base, extent=extent, origin="upper", interpolation="bilinear", zorder=0)
layer_im = axm.imshow(np.zeros((Hh, Ww, 4)), extent=extent, origin="upper",
                      interpolation="nearest", zorder=2)
axm.set_xlim(-ext_x, ext_x)
axm.set_ylim(-ext_y, ext_y)
axm.axis("off")

glow = [pe.withStroke(linewidth=5, foreground=BG)]
stat_txt = axm.text(0.97, 0.985, "", transform=axm.transAxes, ha="right", va="top",
                    fontsize=13, fontweight="normal", color=YELLOW, path_effects=glow,
                    zorder=6, linespacing=1.5)

# Scale bar (5 km) - bottom left
sx0, sx1, sy = -ext_x + 1.5, -ext_x + 6.5, -ext_y + 0.9
axm.plot([sx0, sx1], [sy, sy], color="white", lw=2.5, zorder=6, solid_capstyle="butt")
axm.text((sx0 + sx1) / 2, sy + 0.5, "5 km", color="white", fontsize=10, ha="center", zorder=6,
         path_effects=glow)

# Legend: when it was built
axl = fig.add_axes([0.72, 0.185, 0.20, 0.010])
axl.imshow(np.linspace(0, 1, 256)[None, :], cmap=RAMP, aspect="auto")
axl.axis("off")
fig.text(0.72, 0.205, "WHEN IT WAS BUILT", color="white", fontsize=9, fontweight="bold", ha="left")
fig.text(0.72, 0.172, f"≤ {EPOCHS[0]}", color=CYAN, fontsize=9, ha="left")
fig.text(0.92, 0.172, str(EPOCHS[-1]), color=YELLOW, fontsize=9, ha="right")

# Timeline bars (built-up area per epoch)
axt = fig.add_axes([0.06, 0.092, 0.88, 0.052], facecolor=BG)
bar_colors = [RAMP(i / (n - 1)) for i in range(n)]
bars = axt.bar(range(n), np.zeros(n), color=bar_colors, width=0.62)
axt.set_xticks(range(n))
axt.set_xticklabels(EPOCHS, fontsize=9, color=MUTED)
axt.set_yticks([])
axt.set_ylim(0, max(area_km2) * 1.08)
axt.set_xlim(-0.6, n - 0.4)
for sp in ["top", "right", "left"]:
    axt.spines[sp].set_visible(False)
axt.spines["bottom"].set_color("#2a1f4d")
axt.tick_params(length=0)

# Footer (same layout as the other maps)
fig.text(0.06, 0.04, f"#30DayMapChallenge | Day {DAY}: {THEME} | {AUTHOR}\n{SOURCE}",
         color=MUTED, fontsize=8, ha="left", va="bottom")

white = np.array([1.0, 1.0, 1.0])


def render(k, t):
    """Frame for epoch index k; t in (0,1] = progress of the new growth appearing."""
    layer = np.zeros((Hh, Ww, 4))
    settled = (first < k) if k > 0 else (first == 0)
    new = (first == k) if k > 0 else np.zeros_like(settled)

    dens = 0.55 + 0.45 * np.clip(built_pct[k] / 50.0, 0, 1)
    idx = np.clip(first, 0, n - 1) / (n - 1)
    rgb = RAMP(idx)[..., :3]
    layer[settled, :3] = rgb[settled]
    layer[settled, 3] = dens[settled]

    if new.any():
        col = rgb[new]
        mix = t ** 0.7
        layer[new, :3] = white * (1 - mix) + col * mix
        layer[new, 3] = np.clip(0.25 + 0.75 * t, 0, 1) * dens[new]
    layer_im.set_data(layer)

    # Counters (interpolated during the fade)
    prev = area_km2[k - 1] if k > 0 else area_km2[0]
    area_now = prev + (area_km2[k] - prev) * t
    growth = (area_now / area_km2[0] - 1) * 100
    stat_txt.set_text(f"{EPOCHS[k]}  ·  {area_now:.1f} km² built-up\n"
                      + (f"+{growth:.0f}% since {EPOCHS[0]}" if k > 0 else "starting point"))

    for i, b in enumerate(bars):
        if i < k:
            b.set_height(area_km2[i])
        elif i == k:
            b.set_height(prev + (area_km2[k] - prev) * t if k > 0 else area_km2[0])
        else:
            b.set_height(0)

    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


# ----------------------------------------------------------- ANIMATION ------
frames, durations = [], []
frames.append(render(0, 1.0)); durations.append(1500)
for k in range(1, n):
    for s in range(1, FADE_STEPS + 1):
        t = s / FADE_STEPS
        frames.append(render(k, t))
        durations.append(900 if s == FADE_STEPS else 110)
durations[-1] = 4000   # long hold on the final frame

# Final-frame poster (high res)
render(n - 1, 1.0)
fig.savefig(OUT_PNG, dpi=200, facecolor=BG)

pal = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
       for f in frames]
pal[0].save(OUT_GIF, save_all=True, append_images=pal[1:], duration=durations,
            loop=0, optimize=False, disposal=2)
print("Saved", OUT_GIF, "and", OUT_PNG)

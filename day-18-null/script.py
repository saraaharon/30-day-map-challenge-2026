"""
#30DayMapChallenge | Day 18: NULL  (animated)
"Map of Nothing": valid Sentinel-2 observations per pixel over Romania,
month by month through the year (Magma palette, dark seamless background, black NULL spots).

pip import/install: rasterio numpy matplotlib imageio imageio-ffmpeg
"""
import calendar
import glob
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------- CONFIG ---------------------------------
YEAR = 2025
AUTHOR = "Sara AHARON"
OUT = "day18_null_magma_romania.mp4"
CACHE = "day18_cache"   # Directory where your .npz monthly rasters are saved

FPS, TRANS, HOLD, END_HOLD = 30, 14, 16, 36

# Palette design: dark seamless background, Magma for valid obs, black for NULL (0 valid)
BG = "#14141f"           # Figure surround & map background for seamless blending
MAP_BG = BG              # Matches figure background to eliminate the country box
NULL_COLOR = "#000000"   # Deep black for 0 valid observations (NULL)
GREY = "#949cb8"
PINK = "#ff2e97"         # Contour glow for NULL patches

# ----------------------------- LOAD LOCAL DATA ------------------------
print(f"Loading cached monthly rasters from '{CACHE}'...")
months = []
bounds = None

for m in range(1, 13):
    pattern = os.path.join(CACHE, f"{YEAR}-{m:02d}.npz")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"Could not find cached file for month {m:02d} matching {pattern}. "
            "Make sure CACHE points to the folder with your downloaded .npz files."
        )
    d = np.load(files[0])
    arr = d["arr"].astype("float32")
    if bounds is None:
        bounds = tuple(d["bounds"])
    months.append(arr)
    print(f"  {calendar.month_abbr[m]} {YEAR} loaded")

inside = months[0] >= 0
# Outside country = NaN, NULL (0 valid) = 0.0, Valid obs > 0
A = []
for a in months:
    a_masked = np.where(inside, a, np.nan)
    A.append(a_masked)

pct = np.array([100 * ((a == 0) & inside).sum() / inside.sum() for a in months])
print("NULL % per month across Romania:", np.round(pct, 1))

allpos = np.concatenate([a[a > 0] for a in A if a[a > 0].size > 0])
vmax = max(2, np.percentile(allpos, 98)) if allpos.size else 2
l, b, r_, t = bounds
extent = (l, r_, b, t)

# ----------------------------- FIGURE & COLORMAP ----------------------
# Magma colormap for valid observation counts (dark purple/brown to bright yellow/orange)
base_magma = plt.get_cmap("magma")
colors_list = base_magma(np.linspace(0.15, 1.0, 256))
# Set the very bottom / zero value to pure black for NULL observations
colors_list[0] = [0.0, 0.0, 0.0, 1.0]

cmap = LinearSegmentedColormap.from_list("magma_null_black", colors_list)
cmap.set_bad(MAP_BG)     # Background outside Romania country boundary matches frame
cmap.set_under("#000000")

fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
ax = fig.add_axes([0.06, 0.235, 0.88, 0.61], facecolor=MAP_BG)

im = ax.imshow(np.ma.masked_invalid(A[0]), cmap=cmap, vmin=0, vmax=vmax,
               extent=extent, origin="upper", interpolation="nearest")
ax.set_axis_off()
ax.set_aspect("equal")

# Mini chart: % NULL per month
ax2 = fig.add_axes([0.06, 0.095, 0.88, 0.09], facecolor=BG)
ax2.bar(range(12), pct, color="#5a189a", width=0.7)
ax2.set_xlim(-0.6, 11.6)
ax2.set_ylim(0, max(5, pct.max() * 1.2))
ax2.set_xticks(range(12))
ax2.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)],
                    color=GREY, size=8)
ax2.tick_params(axis="x", length=0)
ax2.set_yticks([])
for s in ax2.spines.values():
    s.set_visible(False)
marker = ax2.axvline(0, color=PINK, lw=2)

# ---- Layout & Typography Standards ----
fig.text(0.06, 0.93, "Map of Nothing", color="white", size=26, weight="bold")
fig.text(0.06, 0.895,
         f"Where Sentinel-2 saw nothing usable across Romania, month by month, {YEAR}",
         color="#ffb703", size=13)
fig.text(0.06, 0.04, f"#30DayMapChallenge | Day 18: NULL | {AUTHOR}",
         color=GREY, size=8)
fig.text(0.06, 0.022,
         "Data: Copernicus Sentinel-2 L2A (SCL cloud mask) | Palette: Magma (bright = high valid obs, black = NULL) | CRS: Stereo 70 (EPSG:3844)",
         color=GREY, size=8)

month_txt = fig.text(0.94, 0.93, "", color=PINK, size=26, weight="bold", ha="right")
pct_txt = fig.text(0.94, 0.205, "", color=PINK, size=11, ha="right")
fig.text(0.06, 0.205, "■ solid black spots + pink glow = 0 valid observations (NULL)",
         color=GREY, size=9)

# ----------------------------- ANIMATE --------------------------------
pos = [0.0] * HOLD
for i in range(11):
    for k in range(1, TRANS + 1):
        s = k / TRANS
        pos.append(i + s * s * (3 - 2 * s))              # smoothstep
    pos += [float(i + 1)] * HOLD
pos += [11.0] * END_HOLD

writer = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1, pixelformat="yuv420p")
contours = []
print(f"Rendering {len(pos)} frames...")
for n, p in enumerate(pos):
    i = min(int(np.floor(p)), 10)
    f = p - i
    arr = (1 - f) * A[i] + f * A[i + 1]
    im.set_data(np.ma.masked_invalid(arr))

    for cs in contours:
        cs.remove()
    contours = []
    z = np.nan_to_num(arr, nan=99.0)
    # Contour outlining strict NULL patches (values strictly equal to 0 inside country)
    null_mask = ((arr >= 0) & (arr < 0.5)).astype(float)
    if null_mask.any():
        for lw, a in [(7, 0.12), (1.0, 0.9)]:
            contours.append(ax.contour(null_mask, levels=[0.5], colors=PINK,
                                       linewidths=lw, alpha=a,
                                       extent=extent, origin="upper"))

    cur_pct = (1 - f) * pct[i] + f * pct[i + 1]
    month_txt.set_text(f"{calendar.month_abbr[int(round(p)) + 1].upper()} {YEAR}")
    pct_txt.set_text(f"NULL: {cur_pct:.1f}% of Romania")
    marker.set_xdata([p, p])

    fig.canvas.draw()
    writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    if n % 50 == 0:
        print(f"  frame {n}/{len(pos)}")

writer.close()
print("Saved", OUT)

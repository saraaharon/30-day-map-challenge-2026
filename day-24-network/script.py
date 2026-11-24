"""
30 Day Map Challenge - Day 24: Network (Beijing Adaptation)
Beijing's Human Movement Network, hour by hour, from Microsoft GeoLife GPS Trajectories.
"""
import glob
import os
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pyproj import Transformer
from scipy.ndimage import gaussian_filter

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
BBOX = (116.25, 39.75, 116.65, 40.05)    # left, bottom, right, top (lon/lat) = Beijing urban area
GEOLIFE_DIR = "Data"                    # Root folder containing user directories (000, 001, etc.)
CSV = "beijing_gps_points.csv"          # Download cache / compiled CSV
CELL_M = 35                             # Finer grid cells (35m) for crispness
SMOOTH = 1.0                            # Reduced blur for sharper network lines
WALK_MIN, WALK_MAX = 0.6, 2.0           # m/s (approx 2 to 7 km/h)
FPS = 30
SECONDS_PER_DAY = 12                    # Length of one 24h loop in seconds
SIZE_PX = 1080
OUT_MP4, OUT_PNG = "day24_network_beijing.mp4", "day24_network_beijing_final.png"
# ----------------------------

BG = "#0b0f1a"

# 1. Parse Microsoft GeoLife Dataset -----------------------------------
def parse_geolife():
    rows = []
    seg_id = 0
    user_folders = sorted(glob.glob(os.path.join(GEOLIFE_DIR, "*")))
    if not user_folders:
        raise SystemExit(
            f"Error: GeoLife 'Data' folder not found at '{GEOLIFE_DIR}/'. "
            "Please download and extract the Microsoft GeoLife dataset here: "
            "https://www.microsoft.com/en-us/download/details.aspx?id=52367"
        )
    
    print("Parsing GeoLife trajectory network files (.plt)...")
    for user_folder in user_folders:
        plt_files = glob.glob(os.path.join(user_folder, "Trajectory", "*.plt"))
        for plt_file in plt_files:
            seg_id += 1
            try:
                with open(plt_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                for line in lines[6:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 7:
                        lat, lon = float(parts[0]), float(parts[1])
                        if BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]:
                            date_str, time_str = parts[5], parts[6]
                            rows.append((seg_id, lat, lon, f"{date_str} {time_str}"))
            except Exception:
                continue
                
    df = pd.DataFrame(rows, columns=["seg", "lat", "lon", "time"])
    df.to_csv(CSV, index=False)
    print(f"Compiled and saved {len(df):,} network points within Beijing bounds to {CSV}")

if not Path(CSV).exists():
    parse_geolife()

# 2. Process & Filter Walking Network Data ----------------------------
df = pd.read_csv(CSV)
if df.empty:
    raise SystemExit("GeoLife dataset CSV is empty or points fell outside the bounding box.")

to_m = Transformer.from_crs("EPSG:4326", "EPSG:32650", always_xy=True)
df["x"], df["y"] = to_m.transform(df["lon"].values, df["lat"].values)
df["t"] = pd.to_datetime(df["time"], errors="coerce")

timed = df[df["t"].notna()].sort_values(["seg", "t"], kind="stable").copy()
g = timed.groupby("seg")
dist = np.hypot(g["x"].shift(-1) - timed["x"], g["y"].shift(-1) - timed["y"])
dt = (g["t"].shift(-1) - timed["t"]).dt.total_seconds()
timed["speed"] = (dist / dt).where((dt > 0) & (dt <= 30))
timed["speed_s"] = timed.groupby("seg")["speed"].transform(
    lambda s: s.rolling(5, center=True, min_periods=2).median())
walk = timed[timed["speed_s"].between(WALK_MIN, WALK_MAX)].copy()

print(f"{len(df):,} points in total | {len(timed):,} with timestamps | "
      f"{len(walk):,} walking points across {walk['seg'].nunique():,} network segments")

if len(walk) < 500:
    raise SystemExit("Too few timestamped network points found in this region.")

local_t = walk["t"].dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
walk["hour"] = local_t.dt.hour

# 3. Network Density Fields & Grids ----------------------------------
x0, y0 = to_m.transform(BBOX[0], BBOX[1])
x1, y1 = to_m.transform(BBOX[2], BBOX[3])
nx, ny = int((x1 - x0) / CELL_M), int((y1 - y0) / CELL_M)
xe, ye = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)

def hist(d):
    h, _, _ = np.histogram2d(d["x"], d["y"], bins=[xe, ye])
    return h.T

base = gaussian_filter(hist(df), 0.8)
base = np.log1p(base)
base = np.clip(base / np.percentile(base[base > 0], 99.5), 0, 1)

raw = np.stack([hist(walk[walk["hour"] == h]) for h in range(24)])
hourly_counts = raw.sum(axis=(1, 2))
mix = 0.25 * np.roll(raw, 1, 0) + 0.5 * raw + 0.25 * np.roll(raw, -1, 0)
fields = np.stack([gaussian_filter(m, SMOOTH) for m in mix])
vmax = np.percentile(fields[fields > 0], 99.5) if np.any(fields > 0) else 1.0

def alpha_cmap(name, colors, power):
    cols = LinearSegmentedColormap.from_list(name, colors)(np.linspace(0, 1, 256))
    cols[:, 3] = np.linspace(0, 1, 256) ** power
    return ListedColormap(cols)

BASE_CMAP = alpha_cmap("base", ["#102a45", "#0088cc", "#00e5ff"], 0.85)
PULSE_CMAP = alpha_cmap("pulse", ["#ff2e97", "#ff7a3d", "#ffd166", "#fff3c4"], 0.65)

def pulse_img(f):
    return np.clip(f / vmax, 0, 1) ** 0.50

# 4. Figure & Network Styling (Enlarged Map & Compact Lower-Right Controls) ---
dpi = 100
fig = plt.figure(figsize=(SIZE_PX / dpi, SIZE_PX / dpi), dpi=dpi)
fig.patch.set_facecolor(BG)

ax = fig.add_axes([0.10, 0.28, 0.80, 0.58])
ax.set_facecolor(BG)
extent = (x0, x1, y0, y1)

# Using 'nearest' interpolation for a razor-sharp, clear network vector look
ax.imshow(base, cmap=BASE_CMAP, extent=extent, origin="lower", vmin=0, vmax=1,
          alpha=0.9, zorder=1, interpolation="nearest")
im = ax.imshow(pulse_img(fields[0]), cmap=PULSE_CMAP, extent=extent, origin="lower",
               vmin=0, vmax=1, zorder=2, interpolation="nearest")

ax.plot([x0 + 1000, x0 + 6000], [y0 + 1000] * 2, color="#8a93b0", lw=2, zorder=5)
ax.text(x0 + 3500, y0 + 1700, "5 km", color="#8a93b0", fontsize=9, ha="center", zorder=5)

ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")

fig.text(0.10, 0.940, "Beijing Pedestrian Movement Network", color="white",
         fontsize=24, fontweight="bold", ha="left", va="center")
fig.text(0.10, 0.900, "Hourly pulse of human trajectories from Microsoft GeoLife",
         color="#ffb347", fontsize=13, ha="left", va="center")

# Lowered live clock stacked right on top of "Beijing local time"
clock = fig.text(0.53, 0.185, "00:00", color="white", fontsize=16, fontweight="bold",
                 ha="center", va="center")

# "Beijing local time" level with "Network activity per hour"
fig.text(0.53, 0.15, "Beijing local time", color="#c7cde0", fontsize=8,
         ha="center", va="center")

# Super-compact activity chart positioned on the right
bax = fig.add_axes([0.65, 0.17, 0.20, 0.07])
bax.set_facecolor(BG)
max_cnt = hourly_counts.max() if hourly_counts.max() > 0 else 1.0
bars = bax.bar(np.arange(24), hourly_counts / max_cnt, width=0.75, color="#3a4466")
bax.set_xticks([0, 6, 12, 18, 23])
bax.set_xticklabels(["0", "6", "12", "18", "23"], color="#8a93b0", fontsize=7)
bax.set_yticks([])
bax.tick_params(length=0, pad=1)
for sp in bax.spines.values():
    sp.set_visible(False)

# Network activity title positioned directly underneath the chart
fig.text(0.75, 0.15, "Network activity per hour", color="#c7cde0", fontsize=8, ha="center")

# Centered stats and legend aligned under the chart block
fig.text(0.75, 0.115, f"{len(walk):,} active nodes  |  {walk['seg'].nunique():,} routes", color="white", fontsize=8, ha="center")

fig.text(0.68, 0.085, "—— active flow", color="#ff7a3d", fontsize=8, fontweight="bold", ha="center")
fig.text(0.82, 0.085, "—— network skeleton", color="#00e5ff", fontsize=8, fontweight="bold", ha="center")

fig.text(0.10, 0.08,
         f"#30DayMapChallenge | Day 24: Network | {AUTHOR}\n"
         "Network flow = timestamped walking vectors (0.6–2.0 m/s) over spatial grid.\n"
         "Data: Microsoft Research GeoLife Trajectories (Beijing)",
         color="#8a93b0", fontsize=8.5, ha="left", va="bottom", linespacing=1.5)

def draw_frame(day_pos):
    k0 = int(day_pos) % 24
    k1 = (k0 + 1) % 24
    w = day_pos - int(day_pos)
    im.set_data(pulse_img((1 - w) * fields[k0] + w * fields[k1]))
    clock.set_text(f"{k0:02d}:{int(w * 60):02d}")
    for i, b in enumerate(bars):
        b.set_color("#ffb347" if i == k0 else "#3a4466")

def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# 5. Render & Output --------------------------------------------------
n_frames = FPS * SECONDS_PER_DAY
frames = []
print(f"Rendering {n_frames} frames...")
for i in range(n_frames):
    draw_frame(24 * i / n_frames)
    frames.append(grab())
    if (i + 1) % 30 == 0:
        print(f"Rendered frame {i + 1}/{n_frames}")

peak = int(np.argmax(hourly_counts))
draw_frame(peak + 0.0)
fig.savefig(OUT_PNG, facecolor=BG, dpi=dpi)
print(f"Saved {OUT_PNG}")

print(f"Writing MP4 video to {OUT_MP4}...")
try:
    with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=None, pixelformat="yuv420p") as wr:
        for idx, fr in enumerate(frames):
            wr.append_data(fr)
    print(f"Successfully saved {OUT_MP4}!")
except Exception as e:
    print(f"Error writing MP4: {e}")

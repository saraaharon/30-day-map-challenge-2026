"""
30 Day Map Challenge - Day 7: 10 Minute Map
Volcanoes of Fire: Chronological Eruption Timeline animation (Widescreen Landscape).
- Dynamic .mp4 and .png exports.
- Follows Sara AHARON's template styling, colors, layout, and metadata.

Install once:  pip install numpy scipy matplotlib geopandas pyproj imageio imageio-ffmpeg shapely
Outputs:        day7_volcanoes.mp4  +  day7_volcanoes_final.png
"""
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from shapely.geometry import Point

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
FPS = 30
HOLD_SECONDS = 3
WIDTH_PX, HEIGHT_PX = 1920, 1080  # Widescreen landscape format
OUT_MP4, OUT_PNG = "day7_volcanoes.mp4", "day7_volcanoes_final.png"
# ----------------------------

BG, LAND = "#0b0f1a", "#151b2b"

# Iconic historical violent volcanic eruptions (Chronologically sorted)
VOLCANOES = [
    ("Thera (Santorini)", 36.4044, 25.3972, -1610, "VEI 7 · ~1610 BC"),
    ("Vesuvius", 40.8224, 14.4289, 79, "VEI 5 · 79 AD"),
    ("Changbaishan (Baoan)", 41.99, 128.05, 946, "VEI 7 · 946 AD"),
    ("Kuwae", -16.82, 168.47, 1452, "VEI 6 · 1452 AD"),
    ("Huaynaputina", -16.61, -70.85, 1600, "VEI 6 · 1600 AD"),
    ("Tambora", -8.25, 118.00, 1815, "VEI 7 · 1815 AD"),
    ("Krakatoa", -6.102, 105.423, 1883, "VEI 6 · 1883 AD"),
    ("Novarupta", 58.27, -155.15, 1912, "VEI 6 · 1912 AD"),
    ("Mount Katmai", 58.27, -155.53, 1912, "VEI 5 · 1912 AD"),
    ("Mount St. Helens", 46.1912, -122.1944, 1980, "VEI 5 · 1980 AD"),
    ("El Chichón", 17.36, -93.22, 1982, "VEI 5 · 1982 AD"),
    ("Nevado del Ruiz", 4.89, -75.32, 1985, "VEI 3 · 1985 AD"),
    ("Pinatubo", 15.13, 120.35, 1991, "VEI 6 · 1991 AD"),
    ("Tonga (Hunga Tonga)", -20.546, -175.388, 2022, "VEI 5-6 · 2022 AD")
]

# 1. Prepare Spatial Data ----------------------------------------------
df = pd.DataFrame(VOLCANOES, columns=["name", "lat", "lon", "year", "vei"])
df = df.sort_values("year").reset_index(drop=True)

df['x'] = df['lon']
df['y'] = df['lat']

# Load Natural Earth World Boundaries
url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

# 2. Figure Setup (Widescreen 16:9 Landscape) --------------------------
dpi = 100
fig = plt.figure(figsize=(WIDTH_PX / dpi, HEIGHT_PX / dpi), dpi=dpi)
fig.patch.set_facecolor(BG)

# Map axis occupying the upper-middle section
ax = fig.add_axes([0.03, 0.22, 0.94, 0.65])
ax.set_facecolor(BG)

world.plot(ax=ax, facecolor=LAND, edgecolor="#2a3350", linewidth=0.5, zorder=1)

ax.set_xlim(-180, 180)
ax.set_ylim(-85, 85)
ax.set_aspect("equal")
ax.set_axis_off()

# Animated elements setup on map
N_VOLC = len(df)
scat_glow = ax.scatter([], [], s=0, c="#ff2e97", alpha=0.3, zorder=10)
scat_core = ax.scatter([], [], s=0, c="#fff3c4", alpha=1.0, zorder=11)

# Custom offsets to prevent overlapping labels for close/crowded volcanoes
custom_offsets = {
    "Novarupta": (-70, 10),
    "Mount Katmai": (15, -15),
    "Krakatoa": (-50, -20),
    "Tambora": (15, 10),
    "Kuwae": (15, -5)
}

labels = {}
for i, row in df.iterrows():
    offset = custom_offsets.get(row['name'], (8, 8))
    # Map label now only shows the volcano name cleanly without year clutter
    labels[i] = ax.annotate(
        f"{row['name']}",
        (row['x'], row['y']),
        xytext=offset, textcoords="offset points",
        color="white", fontsize=9, alpha=0, fontweight="bold", zorder=15
    )

# Header & Metadata Texts following template layout
fig.text(0.05, 0.945, "Volcanoes of Fire: Violent Eruptions", color="white",
         fontsize=26, fontweight="bold", ha="left", va="center")
fig.text(0.05, 0.905, "Chronological timeline of Earth's most catastrophic historical explosions",
         color="#ffb347", fontsize=14, ha="left", va="center")

fig.text(0.05, 0.055,
         f"#30DayMapChallenge | Day 7: 10 Minute Map | {AUTHOR}\n"
         "Mapping recorded major historical eruptions chronologically by Volcanic Explosivity Index (VEI).\n"
         "Data: Global Volcanism Program & Historical Records",
         color="#8a93b0", fontsize=8.5, ha="left", va="bottom", linespacing=1.5)

# --- Bottom Right Timeline Axis (Scale) ---
ax_time = fig.add_axes([0.55, 0.04, 0.40, 0.08])
ax_time.set_facecolor(BG)
ax_time.spines['top'].set_visible(False)
ax_time.spines['right'].set_visible(False)
ax_time.spines['left'].set_visible(False)
ax_time.spines['bottom'].set_color('#4a5578')
ax_time.spines['bottom'].set_linewidth(1.5)
ax_time.tick_params(colors='#8a93b0', labelsize=8)
ax_time.set_xlim(-2000, 2050)
ax_time.set_ylim(0, 1)
ax_time.yaxis.set_visible(False)

# Draw baseline and all event tick markers on the timeline axis
ax_time.axhline(0.5, color='#2a3350', linewidth=2, zorder=2)
all_years = df['year'].values
ax_time.scatter(all_years, np.zeros_like(all_years) + 0.5, color='#4a5578', s=35, zorder=3)

# Active indicator on the timeline axis
active_timeline_dot = ax_time.scatter([], [], color='#ff2e97', s=120, zorder=5, marker='o')

# Text fields directly above the timeline axis, brought in close
timeline_info_txt = fig.text(0.75, 0.13, "", color="white", fontsize=16, fontweight="bold",
                             ha="center", va="bottom")

# Animation Timing Constants
STEP_FRAMES = 45  # frames per volcano entry
total_anim_frames = N_VOLC * STEP_FRAMES

def draw_frame(t):
    current_idx = min(t // STEP_FRAMES, N_VOLC - 1)
    sub_t = t % STEP_FRAMES
    
    active_x = df['x'].iloc[:current_idx + 1].values
    active_y = df['y'].iloc[:current_idx + 1].values
    
    # Pulse sizing effect for the latest eruption
    sizes_glow = np.ones(current_idx + 1) * 350
    sizes_core = np.ones(current_idx + 1) * 70
    if sub_t < 15:
        pulse = 1.0 + (15 - sub_t) * 0.15
        sizes_glow[-1] *= pulse
        
    scat_glow.set_offsets(np.c_[active_x, active_y])
    scat_glow.set_sizes(sizes_glow)
    
    scat_core.set_offsets(np.c_[active_x, active_y])
    scat_core.set_sizes(sizes_core)
    
    # Update label opacities
    for i, txt in labels.items():
        if i < current_idx:
            txt.set_alpha(0.8)
        elif i == current_idx:
            fade_in = min(1.0, sub_t / 10.0)
            txt.set_alpha(fade_in * 0.9)
        else:
            txt.set_alpha(0.0)

    cur_row = df.iloc[current_idx]
    
    # Update timeline axis marker position
    active_timeline_dot.set_offsets([[cur_row['year'], 0.5]])
    
    # Update text directly above timeline axis
    vei_clean = cur_row['vei'].split('·')[0].strip()
    year_label = f"{cur_row['year']} AD" if cur_row['year'] > 0 else f"{abs(cur_row['year'])} BC"
    timeline_info_txt.set_text(f"{cur_row['name']}  ({year_label})  ·  {vei_clean}")

def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

# 3. Render Frames -----------------------------------------------------
frames = []
for t in range(total_anim_frames):
    draw_frame(t)
    frames.append(grab())
    if t % 30 == 0:
        print(f"frame {t + 1}/{total_anim_frames}")

# Hold final frame
frames += [frames[-1]] * int(HOLD_SECONDS * FPS)

fig.savefig(OUT_PNG, facecolor=BG, dpi=dpi)
print(f"Saved {OUT_PNG}")

with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as wr:
    for fr in frames:
        wr.append_data(fr)
print(f"Saved {OUT_MP4}")

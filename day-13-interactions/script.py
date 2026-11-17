"""
#30DayMapChallenge | Day 13: Interactions
Gravity-model "desire lines" between Romania's county seats, animated.

    flow(i, j) = pop_i * pop_j / distance_ij^2

Inputs : ro_cities_2021.csv  (city, county, pop_2021, lat, lon)
         romania.json        (border geometry; downloaded automatically if missing)
Output : day13_interactions.mp4  (1080x1350, 30 fps)

Needs: numpy, pandas, matplotlib, imageio, imageio-ffmpeg
Usage: python day13_interactions.py            -> full video
       python day13_interactions.py --still    -> one PNG preview (still.png)
"""
import json
import os
import sys
import urllib.request

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import Polygon, Rectangle

# ------------------------------------------------------------------ settings
DAY, THEME, AUTHOR = 13, "Interactions", "Sara AHARON"
CSV = "ro_cities_2021.csv"
BORDER = "romania.json"
BORDER_URL = ("https://raw.githubusercontent.com/datasets/geo-countries/"
              "master/data/countries.geojson")
OUT = "day13_interactions.mp4"

W, H, DPI, FPS = 1080, 1350, 100, 30
DURATION = 13.0
N_PAIRS = 130            # strongest gravity pairs drawn as arcs
M = 90                   # points per arc
T_NODES = 1.2            # nodes fade in
T_ARCS0, T_ARCS1 = 1.3, 6.3   # arcs draw in (strongest first)
T_DRAW = 0.9             # time one arc takes to draw

BG = "#06030f"
GREY = "#8a93b0"
CMAP = LinearSegmentedColormap.from_list(
    "synth", ["#4a2cc9", "#9d4edd", "#ff2d95", "#ff8a5c", "#ffe066"])
CYAN, PINK = "#00f0ff", "#ff4fd8"

LABELS = {  # city: (dx, dy, ha) offsets in points
    "București": (10, -16, "left"), "Cluj-Napoca": (-9, 7, "right"),
    "Iași": (9, 5, "left"), "Constanța": (-10, -4, "right"),
    "Timișoara": (-9, 5, "right"), "Brașov": (9, -3, "left"),
    "Craiova": (-9, -10, "right"), "Galați": (9, 3, "left"),
    "Oradea": (-9, 5, "right"), "Ploiești": (-9, 6, "right"),
    "Sibiu": (-9, -10, "right"), "Focșani": (9, -3, "left"),
    "Pitești": (-9, -9, "right"),
}


# ------------------------------------------------------------------ data
def haversine(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_border():
    if not os.path.exists(BORDER):
        print("downloading border ...")
        with urllib.request.urlopen(BORDER_URL, timeout=60) as r:
            data = json.load(r)
        geom = None
        for f in data["features"]:
            if f["properties"].get("ISO3166-1-Alpha-3") == "ROU":
                geom = f["geometry"]
        json.dump(geom, open(BORDER, "w"))
    g = json.load(open(BORDER))
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [np.array(p[0]) for p in polys]  # exterior rings, (lon, lat)


COS = np.cos(np.radians(45.9))


def proj(lon, lat):
    return np.column_stack([np.asarray(lon) * COS, np.asarray(lat)])


df = pd.read_csv(CSV)
xy = proj(df.lon.values, df.lat.values)
pop = df.pop_2021.values.astype(float)
names = df.city.tolist()

pairs = []
for i in range(len(df)):
    for j in range(i + 1, len(df)):
        d = haversine(df.lat[i], df.lon[i], df.lat[j], df.lon[j])
        pairs.append((pop[i] * pop[j] / d ** 2, i, j))
pairs.sort(reverse=True)
TOTAL_PAIRS = len(pairs)
pairs = pairs[:N_PAIRS]
F = np.array([p[0] for p in pairs])
S = (F / F.max()) ** 0.30            # 0..1 strength used for colour/brightness
INDEX = 100 * F / F.max()            # interaction index (strongest = 100)
LW = 0.5 + 3.4 * S ** 1.6

# arcs: quadratic Bezier bulging north
tt = np.linspace(0, 1, M)[:, None]
ARCS = np.zeros((N_PAIRS, M, 2))
for k, (_, i, j) in enumerate(pairs):
    a, b = xy[i], xy[j]
    v = b - a
    perp = np.array([-v[1], v[0]])
    if perp[1] < 0:
        perp = -perp
    c = (a + b) / 2 + perp * 0.20
    ARCS[k] = (1 - tt) ** 2 * a + 2 * (1 - tt) * tt * c + tt ** 2 * b

# timing: strongest arc first
START = T_ARCS0 + (T_ARCS1 - T_ARCS0) * (np.arange(N_PAIRS) / N_PAIRS) ** 0.85
DONE = START + T_DRAW

# pulses (travel both ways: interaction is symmetric)
rng = np.random.default_rng(13)
SPEED = 0.16 + 0.22 * S
NPULSE = 1 + np.round(2 * S).astype(int)
pulses = []   # (arc k, direction, phase offset)
for k in range(N_PAIRS):
    for n in range(NPULSE[k]):
        off = (n + rng.random() * 0.3) / NPULSE[k]
        pulses.append((k, +1, off))
        pulses.append((k, -1, (off + 0.5) % 1))
P_ARC = np.array([p[0] for p in pulses])
P_DIR = np.array([p[1] for p in pulses])
P_OFF = np.array([p[2] for p in pulses])
TRAIL = 6
TRAIL_STEP = 0.014

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)

bg = fig.add_axes([0, 0, 1, 1])
bg.axis("off")
yy, xx = np.mgrid[0:1:400j, 0:1:320j]
r = np.sqrt((xx - 0.5) ** 2 + 0.8 * (yy - 0.58) ** 2)
glow = np.clip(1 - r * 1.55, 0, 1) ** 1.7
rgb = (np.array(to_rgb(BG))[None, None] * (1 - glow[..., None])
       + np.array(to_rgb("#27114f"))[None, None] * glow[..., None])
bg.imshow(rgb, extent=[0, 1, 0, 1], aspect="auto", origin="lower")
srng = np.random.default_rng(7)
bg.scatter(srng.random(160), srng.random(160), s=srng.random(160) * 2.2,
           c="white", alpha=0.25, linewidths=0)
bg.set_xlim(0, 1)
bg.set_ylim(0, 1)

ax = fig.add_axes([0.03, 0.30, 0.94, 0.55])
ax.set_facecolor("none")
ax.axis("off")
ax.set_xlim(20.0 * COS, 30.0 * COS)
ax.set_ylim(43.4, 48.6)
ax.set_aspect("equal", adjustable="box")

# graticule
for lon in range(20, 31):
    ax.plot([lon * COS] * 2, [43.4, 48.6], color="#1b1440", lw=0.5, zorder=1)
for lat in np.arange(43.5, 48.7, 1.0):
    ax.plot([20 * COS, 30 * COS], [lat, lat], color="#1b1440", lw=0.5, zorder=1)

# border
for ring in load_border():
    pxy = proj(ring[:, 0], ring[:, 1])
    ax.add_patch(Polygon(pxy, closed=True, fc="#0e0824", ec="none", zorder=2))
    for lw, al in [(9, 0.05), (5, 0.10), (2.5, 0.22)]:
        ax.plot(pxy[:, 0], pxy[:, 1], color="#7a5cff", lw=lw, alpha=al,
                zorder=2, solid_joinstyle="round")
    ax.plot(pxy[:, 0], pxy[:, 1], color="#b39bff", lw=1.0, alpha=0.9, zorder=3)

# arcs (3 glow layers)
arc_rgba = CMAP(S)
layers = []
for wmul, amul in [(5.0, 0.07), (2.4, 0.16), (1.0, 0.90)]:
    col = arc_rgba.copy()
    col[:, 3] = amul * (0.35 + 0.65 * S)
    lc = LineCollection([], colors=col, linewidths=LW * wmul,
                        capstyle="round", zorder=4)
    ax.add_collection(lc)
    layers.append(lc)

# pulses: glow + core
glow_sc = ax.scatter([], [], s=[], zorder=6, linewidths=0)
core_sc = ax.scatter([], [], s=[], zorder=7, linewidths=0)

# nodes
node_s = 8 + 300 * (pop / pop.max()) ** 0.6
node_halo = ax.scatter(xy[:, 0], xy[:, 1], s=node_s * 4.5, c=CYAN, alpha=0,
                       linewidths=0, zorder=8)
node_core = ax.scatter(xy[:, 0], xy[:, 1], s=node_s, c="#0b0620",
                       edgecolors=CYAN, linewidths=1.3, alpha=0, zorder=9)
labels = []
for nm, (dx, dy, ha) in LABELS.items():
    i = names.index(nm)
    t = ax.annotate(nm, xy[i], xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=8.5, color="#ece9ff",
                    alpha=0, zorder=10,
                    path_effects=[pe.withStroke(linewidth=2.4,
                                                foreground="#06030f")])
    labels.append(t)

# titles / footer (layout standard)
fig.text(0.06, 0.93, "Cities in Gravity", color="white", size=26,
         weight="bold")
fig.text(0.06, 0.895, "Modeled pull between 41 county seats  ·  "
         "every line pulses both ways", color=PINK, size=14)
fig.text(0.06, 0.04, f"#30DayMapChallenge | Day {DAY}: {THEME} | {AUTHOR}",
         color=GREY, size=8)
fig.text(0.06, 0.025,
         "Data: INS Census 2021 (populations), Natural Earth (border)  ·  "
         "Model: pop A × pop B / straight-line distance²",
         color=GREY, size=8)

# leaderboard panel
panel = fig.add_axes([0.06, 0.095, 0.88, 0.175])
panel.set_facecolor("none")
panel.axis("off")
panel.set_xlim(0, 1)
panel.set_ylim(0, 6.2)
panel.text(0, 5.85, "STRONGEST INTERACTIONS  (index, top pair = 100)",
           color=GREY, size=9, va="center")
counter = panel.text(1, 5.85, "", color=CYAN, size=9, va="center", ha="right")
bars, bar_lbls, bar_vals = [], [], []
for r_ in range(5):
    y = 4.55 - r_ * 1.0
    k = r_
    _, i, j = pairs[k]
    lbl = panel.text(0, y, f"{names[i]}  ↔  {names[j]}", color="#ece9ff",
                     size=10.5, va="center")
    rect = Rectangle((0.40, y - 0.25), 0.0, 0.5, fc=CMAP(S[k]), ec="none")
    panel.add_patch(rect)
    val = panel.text(0.41, y, "", color="white", size=9, va="center")
    bars.append(rect)
    bar_lbls.append(lbl)
    bar_vals.append(val)


# ------------------------------------------------------------------ animation
def smooth(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def point_on(k_idx, u):
    f = np.clip(u, 0, 1) * (M - 1)
    i0 = np.minimum(f.astype(int), M - 2)
    fr = (f - i0)[:, None]
    a = ARCS[k_idx, i0]
    b = ARCS[k_idx, i0 + 1]
    return a + (b - a) * fr


def frame(t):
    # nodes + labels
    na = smooth(t / T_NODES)
    node_halo.set_alpha(0.10 * na)
    node_core.set_alpha(na)
    for lb in labels:
        lb.set_alpha(smooth((t - 0.4) / T_NODES))

    # arcs draw-in
    prog = np.clip((t - START) / T_DRAW, 0, 1)
    prog = smooth(prog)
    segs = []
    for k in range(N_PAIRS):
        n = int(prog[k] * (M - 1)) + 1
        segs.append(ARCS[k, :max(n, 1)] if n >= 2 else np.empty((0, 2)))
    for lc in layers:
        lc.set_segments(segs)

    # pulses
    active = t > DONE[P_ARC]
    fade = np.clip((t - DONE[P_ARC]) / 0.5, 0, 1)
    phase0 = (t * SPEED[P_ARC] + P_OFF) % 1.0
    pts, sz_core, sz_glow, cols = [], [], [], []
    base = arc_rgba[P_ARC][:, :3]
    white = np.ones_like(base)
    for s_ in range(TRAIL):
        ph = phase0 - s_ * TRAIL_STEP
        ok = active & (ph >= 0)
        u = np.where(P_DIR > 0, ph, 1 - ph)
        p_ = point_on(P_ARC, np.where(ok, u, 0))
        a = np.where(ok, fade * (1 - s_ / TRAIL) ** 1.6, 0)
        mix = 0.65 if s_ == 0 else 0.15
        c = base * (1 - mix) + white * mix
        pts.append(p_)
        cols.append(np.column_stack([c, a]))
        sz = (6 + 26 * S[P_ARC]) * (1 - 0.14 * s_)
        sz_core.append(sz)
        sz_glow.append(sz * 4.2)
    pts = np.vstack(pts)
    cols = np.vstack(cols)
    core_sc.set_offsets(pts)
    core_sc.set_sizes(np.concatenate(sz_core))
    core_sc.set_facecolors(cols)
    gcols = cols.copy()
    gcols[:, 3] *= 0.16
    glow_sc.set_offsets(pts)
    glow_sc.set_sizes(np.concatenate(sz_glow))
    glow_sc.set_facecolors(gcols)

    # leaderboard
    drawn = int((prog >= 1).sum())
    counter.set_text(f"{drawn} of {TOTAL_PAIRS} possible links drawn")
    for r_ in range(5):
        p = prog[r_]
        bars[r_].set_width(0.52 * (INDEX[r_] / 100) * p)
        bar_vals[r_].set_x(0.40 + 0.52 * (INDEX[r_] / 100) * p + 0.012)
        bar_vals[r_].set_text(f"{INDEX[r_] * p:.0f}" if p > 0 else "")


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


if __name__ == "__main__":
    if "--still" in sys.argv:
        frame(9.0)
        fig.savefig("still.png", dpi=DPI, facecolor=BG)
        print("saved still.png")
        print("top pairs:")
        for k in range(8):
            _, i, j = pairs[k]
            print(f"  {names[i]} - {names[j]}  index {INDEX[k]:.0f}")
        print("weakest drawn index:", round(INDEX[-1], 2), "pulses:", len(pulses))
    else:
        n = int(DURATION * FPS)
        with imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8,
                                macro_block_size=1,
                                ffmpeg_params=["-pix_fmt", "yuv420p"]) as w:
            for f in range(n):
                frame(f / FPS)
                w.append_data(grab())
                if f % 30 == 0:
                    print(f"{f}/{n}", flush=True)
        print("saved", OUT)

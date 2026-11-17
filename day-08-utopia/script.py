"""
Pangaea 2.0 - The Common Sky: Starry Cyber-Cloud Edition
Author: Sara AHARON

Usage:
    python pangaea_stars.py --png    # quick test: renders 2 stills (surface + x-ray)
    python pangaea_stars.py          # full spinning mp4 + final png
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import imageio.v2 as imageio

# ---------- SETTINGS ----------
AUTHOR = "Sara AHARON"
SECONDS = 10
FPS = 30
SIZE_W, SIZE_H = 1080, 1350         # 4:5 portrait
OUT_MP4 = "pangaea_stars.mp4"
OUT_PNG = "pangaea_stars_final.png"
SHELL_SOLID, SHELL_XRAY = 0.55, 0.05  # rock opacity: surface view vs x-ray view

# ---------- PALETTE ----------
BG = "#070b1c"
GOLD = "#f0b94d"
AQUA = "#5ef0e0"
LEAF = "#6fd08c"
ROCK = "#9db4cc"
CREAM = "#f4f1e6"
SKY = "#8fc3e8"
DIM = "#4a5578"
COPPER = "#d97736"
CYBER_CLOUD = "#1e2a5e"
CYBER_GLOW = "#00ffff"

rng = np.random.default_rng(8)

# ---------- FIGURE ----------
fig = plt.figure(figsize=(SIZE_W / 100, SIZE_H / 100), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.0, 0.12, 1.0, 0.78], projection="3d")
ax.set_facecolor(BG)
ax.set_axis_off()
ax.set_xlim(-17.5, 17.5)
ax.set_ylim(-17.5, 17.5)
ax.set_zlim(-12, 12)
ax.set_box_aspect((1, 1, 0.72), zoom=1.4)

shells = []   # rock surfaces whose opacity pulses


# ---------- DRAWING HELPERS ----------
def box(cx, cy, z, w, d, h, face, fa, edge, ea=0.7, lw=0.5):
    ax.bar3d(cx - w / 2, cy - d / 2, z, w, d, h, color=to_rgba(face, fa),
             edgecolor=to_rgba(edge, ea), linewidth=lw, shade=False)


def tower(cx, cy, z, w, h, core=GOLD):
    box(cx, cy, z, w, w, h, AQUA, 0.08, AQUA, 0.6)
    box(cx, cy, z + 0.1, w * 0.4, w * 0.4, h * 0.85, core, 0.55, core, 0.8, 0.4)


def cone(cx, cy, z, r, h, color, fa, edge, sides=24):
    th = np.linspace(0, 2 * np.pi, sides + 1)
    s = np.linspace(0, 1, 3)[:, None]
    X = cx + r * (1 - s) * np.cos(th)[None, :]
    Y = cy + r * (1 - s) * np.sin(th)[None, :]
    Z = z + h * s * np.ones_like(th)[None, :]
    ax.plot_surface(X, Y, Z, color=to_rgba(color, fa), linewidth=0.5,
                    edgecolor=to_rgba(edge, 0.8), shade=False)


def dome(cx, cy, z, r, color, fa, edge):
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi / 2, 6)
    X = cx + r * np.outer(np.cos(u), np.sin(v))
    Y = cy + r * np.outer(np.sin(u), np.sin(v))
    Z = z + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, color=to_rgba(color, fa), linewidth=0.4,
                    edgecolor=to_rgba(edge, 0.8), shade=False)


def circle(cx, cy, z, r, color, a=0.8, lw=1.0, vertical=False):
    t = np.linspace(0, 2 * np.pi, 80)
    if vertical:
        ax.plot(cx + r * np.cos(t), np.full_like(t, cy), z + r * np.sin(t),
                color=color, alpha=a, lw=lw)
    else:
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), np.full_like(t, z),
                color=color, alpha=a, lw=lw)


def disc(cx, cy, z, r, color, a):
    th = np.linspace(0, 2 * np.pi, 40)
    s = np.linspace(0, 1, 4)[:, None]
    ax.plot_surface(cx + r * s * np.cos(th)[None, :], cy + r * s * np.sin(th)[None, :],
                    np.full((4, 40), z), color=to_rgba(color, a), shade=False, linewidth=0)


def trees(cx, cy, z, R, n):
    ang = rng.uniform(0, 2 * np.pi, n)
    rad = R * 0.85 * np.sqrt(rng.uniform(0.05, 1, n))
    ax.scatter(cx + rad * np.cos(ang), cy + rad * np.sin(ang), np.full(n, z + 0.2),
               s=9, c=LEAF, alpha=0.85, depthshade=False, linewidths=0)


def island(cx, cy, R, depth, z0, seed):
    r_ = np.random.default_rng(seed)
    nth, nt = 40, 12
    th = np.linspace(0, 2 * np.pi, nth)
    t = np.linspace(0, 1, nt)[:, None]
    ph = r_.uniform(0, 2 * np.pi, 3)
    wob = (0.5 * np.sin(3 * th + ph[0]) + 0.3 * np.sin(5 * th + ph[1]))[None, :]
    wob2 = 0.25 * np.sin(7 * th[None, :] + 6 * t + ph[2])
    r = R * (1 - t ** 1.4) * (1 + 0.12 * wob + 0.08 * wob2)
    X = cx + r * np.cos(th)[None, :]
    Y = cy + r * np.sin(th)[None, :]
    Z = z0 - depth * t ** 0.85 * (1 + 0.15 * wob)
    under = ax.plot_surface(X, Y, Z, color=ROCK, alpha=SHELL_SOLID, linewidth=0.2,
                            edgecolor=to_rgba(ROCK, 0.5))
    shells.append(under)
    ax.plot_wireframe(X, Y, Z, rstride=2, cstride=3, color=AQUA, alpha=0.3, linewidth=0.4)
    s = np.linspace(0, 1, 5)[:, None]
    Xt = cx + s * (X[0] - cx)[None, :]
    Yt = cy + s * (Y[0] - cy)[None, :]
    Zt = z0 + 0.25 * (1 - s ** 2) * np.ones((1, nth))
    ax.plot_surface(Xt, Yt, Zt, color=to_rgba(LEAF, 0.30), shade=False, linewidth=0)
    ax.scatter([cx], [cy], [z0 - depth * 0.38], s=220, c=GOLD, alpha=0.9,
               edgecolors="white", linewidths=0.8, depthshade=False)
    hz = z0 - depth * r_.uniform(0.2, 0.7, 30)
    ax.scatter(cx + r_.normal(0, 0.35 * R * 0.4, 30), cy + r_.normal(0, 0.35 * R * 0.4, 30), hz,
               s=10, c=GOLD, alpha=0.7, depthshade=False, linewidths=0)
    for a in np.linspace(0, 2 * np.pi, 7)[:-1] + r_.uniform(0, 1):
        ax.plot([cx + 0.55 * R * np.cos(a), cx + 0.2 * R * np.cos(a), cx],
                [cy + 0.55 * R * np.sin(a), cy + 0.2 * R * np.sin(a), cy],
                [z0, z0 - depth * 0.55, z0 - depth * 0.95], color=AQUA, alpha=0.35, lw=0.7)
    trees(cx, cy, z0, R, int(8 * R))


def cyber_cloud(cx, cy, cz, rx, ry, rz):
    u = np.linspace(0, 2 * np.pi, 14)
    v = np.linspace(0, np.pi, 8)
    X = cx + rx * np.outer(np.cos(u), np.sin(v))
    Y = cy + ry * np.outer(np.sin(u), np.sin(v))
    Z = cz + rz * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, color=to_rgba(CYBER_CLOUD, 0.18), linewidth=0.3,
                    edgecolor=to_rgba(CYBER_GLOW, 0.25), shade=False)

cyber_cloud(12.0, 12.0, 5.0, 4.5, 3.5, 1.8)
cyber_cloud(14.0, 8.0, -3.0, 5.0, 4.0, 2.0)
cyber_cloud(-13.0, 11.0, 2.0, 4.0, 4.0, 1.5)


# ---------- BACKGROUND RANDOM STARS ----------
star_rng = np.random.default_rng(99)
num_stars = 120
star_base_x = star_rng.uniform(-22, 22, num_stars)
star_base_y = star_rng.uniform(-22, 22, num_stars)
star_base_z = star_rng.uniform(-11, 11, num_stars)
star_phases = star_rng.uniform(0, 2 * np.pi, num_stars)
star_sc = ax.scatter(star_base_x, star_base_y, star_base_z, s=star_rng.uniform(3, 8, num_stars),
                     c=CREAM, alpha=0.6, depthshade=False, linewidths=0)


# ---------- THE 11 DISTRICTS ----------
SATS = {
    "02 The Glass Gardens":        (135, 12.0, 3.2, 5.0, 1.2),
    "03 Crystal Heights":          (35, 13.0, 3.0, 6.0, 2.0),
    "04 Hearth Quarter":           (210, 12.5, 3.3, 5.2, 0.6),
    "05 Echo Lake":                (325, 9.5,  2.6, 4.5, 0.2),
    "06 Arrival Port":             (270, 13.5, 2.5, 3.5, -0.8),
    "07 The Skyforge & Kilns":     (75,  12.0, 2.9, 5.0, 1.5),
    "08 The Cloud Harvesters":     (165, 14.5, 2.8, 4.8, 0.9),
    "09 The Forum of Voices":      (15,  9.0,  2.5, 4.2, 0.4),
    "10 The Remembrance Gardens":  (355, 13.5, 3.1, 5.5, 0.8),
    "11 Neighborhood Commons":     (300, 11.5, 2.7, 4.6, 0.5),
}

pos = {}
for name, (ang, dist, R, depth, z0) in SATS.items():
    a = np.radians(ang)
    pos[name] = (dist * np.cos(a), dist * np.sin(a), R, depth, z0, a)

CR, CD = 6.0, 8.5
island(0, 0, CR, CD, 0, 1)
for rr in (3.2, 4.0, 4.8):
    circle(0, 0, 0.1, rr, LEAF, 0.6, 0.8)
for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
    ax.plot([1.8 * np.cos(a), 5.0 * np.cos(a)], [1.8 * np.sin(a), 5.0 * np.sin(a)], [0.1, 0.1],
            color=GOLD, alpha=0.5, lw=0.6)
tower(0, 0, 0, 2.8, 1.5)
tower(0, 0, 1.5, 2.0, 3.0)
tower(0, 0, 4.5, 1.3, 2.8)
cone(0, 0, 7.3, 0.55, 3.0, GOLD, 0.35, GOLD)
for sx, sy in ((2.2, 2.2), (-2.2, 2.2), (2.2, -2.2), (-2.2, -2.2)):
    tower(sx, sy, 0, 0.65, 2.4)
    dome(sx, sy, 2.4, 0.4, GOLD, 0.45, GOLD)

x, y, R, dp, z0, a = pos["02 The Glass Gardens"]
island(x, y, R, dp, z0, 2)
for dx, dy, rr in ((-0.7, 0.5, 0.8), (0.8, 0.2, 0.7), (0.0, -0.8, 0.75)):
    dome(x + dx, y + dy, z0 + 0.1, rr, AQUA, 0.14, AQUA)

x, y, R, dp, z0, a = pos["03 Crystal Heights"]
island(x, y, R, dp, z0, 3)
cone(x, y, z0 + 0.1, 0.7, 6.5, AQUA, 0.28, "white", sides=6)
cone(x, y, z0 + 0.1, 0.28, 4.5, GOLD, 0.6, GOLD, sides=6)
circle(x + 1.2, y, z0 + 1.5, 0.8, GOLD, 0.8, 1.0, vertical=True)

x, y, R, dp, z0, a = pos["04 Hearth Quarter"]
island(x, y, R, dp, z0, 4)
for k in range(8):
    ang_h = rng.uniform(0, 2 * np.pi)
    rad_h = R * 0.6 * np.sqrt(rng.uniform(0.1, 1))
    tower(x + rad_h * np.cos(ang_h), y + rad_h * np.sin(ang_h), z0, 0.6, 1.0, core=GOLD)

x, y, R, dp, z0, a = pos["05 Echo Lake"]
island(x, y, R, dp, z0, 5)
disc(x, y, z0 + 0.06, 1.7, AQUA, 0.55)
circle(x, y, z0 + 0.1, 1.8, GOLD, 0.7, 0.8)

x, y, R, dp, z0, a = pos["06 Arrival Port"]
island(x, y, R, dp, z0, 6)
disc(x, y, z0 + 0.06, 1.2, AQUA, 0.55)
circle(x, y, z0 + 1.3, 1.1, GOLD, 0.95, 2.0, vertical=True)

x, y, R, dp, z0, a = pos["07 The Skyforge & Kilns"]
island(x, y, R, dp, z0, 7)
box(x - 0.5, y, z0, 0.8, 0.8, 1.8, COPPER, 0.6, GOLD, 0.8, 0.5)
box(x + 0.5, y + 0.4, z0, 0.6, 0.6, 2.2, COPPER, 0.6, GOLD, 0.8, 0.5)

x, y, R, dp, z0, a = pos["08 The Cloud Harvesters"]
island(x, y, R, dp, z0, 8)
dome(x, y, z0 + 0.1, 1.1, SKY, 0.3, AQUA)

x, y, R, dp, z0, a = pos["09 The Forum of Voices"]
island(x, y, R, dp, z0, 9)
circle(x, y, z0 + 0.1, 1.2, GOLD, 0.8, 1.5)

x, y, R, dp, z0, a = pos["10 The Remembrance Gardens"]
island(x, y, R, dp, z0, 10)
dome(x, y, z0 + 0.1, 0.9, LEAF, 0.4, GOLD)

x, y, R, dp, z0, a = pos["11 Neighborhood Commons"]
island(x, y, R, dp, z0, 11)
for dx, dy in ((-0.5, 0.0), (0.5, 0.0), (0.0, 0.5)):
    tower(x + dx, y + dy, z0, 0.5, 0.8, core=GOLD)

for name, (x, y, R, dp, z0, a) in pos.items():
    sx, sy = CR * np.cos(a), CR * np.sin(a)
    ex, ey = (x - R * np.cos(a)), (y - R * np.sin(a))
    s = np.linspace(0, 1, 25)
    bx, by = sx + (ex - sx) * s, sy + (ey - sy) * s
    bz = 0.1 + (z0 - 0.1) * s + 0.9 * np.sin(np.pi * s)
    ax.plot(bx, by, bz, color=GOLD, alpha=0.18, lw=6)
    ax.plot(bx, by, bz, color=GOLD, alpha=0.9, lw=1.8)

p1, p2 = pos["06 Arrival Port"], pos["05 Echo Lake"]
s = np.linspace(0, 1, 35)
ax.plot(p1[0] + (p2[0] - p1[0]) * s, p1[1] + (p2[1] - p1[1]) * s,
        1.0 + 2.0 * np.sin(np.pi * s), color=AQUA, alpha=0.9, lw=1.2, linestyle=(0, (3, 3)))

labels = {"01 The Sunward Citadel": (0, 0, 11.0)}
for name, (x, y, R, dp, z0, a) in pos.items():
    labels[name] = (x, y, z0 + 3.2)
for name, (lx, ly, lz) in labels.items():
    ax.text(lx, ly, lz, name, color=CREAM, fontsize=7.5, ha="center", va="bottom",
            fontweight="bold")

falls = []
for (cx, cy, R, depth, z0, name) in [(0, 0, CR, CD, 0, "c")] + \
        [(p[0], p[1], p[2], p[3], p[4], n) for n, p in pos.items()]:
    for ang in ([0.7, 2.3, 3.9, 5.5] if name == "c" else [rng.uniform(0, 6.28)]):
        falls.append((cx, cy, R, z0, depth, ang))
NP = 14
fall_phase = np.concatenate([np.linspace(0, 1, NP, endpoint=False) for _ in falls])
fall_sc = ax.scatter(np.zeros(NP * len(falls)), np.zeros(NP * len(falls)), np.zeros(NP * len(falls)),
                     s=6, c=AQUA, alpha=0.85, depthshade=False, linewidths=0)

ships = [ax.scatter([0], [0], [0], s=90, marker="D", c=GOLD, edgecolors="white",
                    linewidths=0.8, depthshade=False) for _ in range(2)]
scan, = ax.plot([], [], [], color=AQUA, alpha=0.55, lw=1.4)

# ---------- HUD ----------
fig.text(0.07, 0.945, "PANGAEA 2.0 // THE COMMON SKY", color="white", fontsize=20,
         fontweight="bold", ha="left", va="center")
fig.text(0.07, 0.915, "EXTENDED CITY ATLAS  |  X-RAY SCAN", color=AQUA, fontsize=10,
         ha="left", va="center")
status = fig.text(0.93, 0.915, "", color=GOLD, fontsize=10, fontweight="bold",
                  ha="right", va="center")
fig.text(0.07, 0.035,
         f"#30DayMapChallenge | Day 8: Utopia | {AUTHOR}\n"
         "Data: Procedural floating-city model | Dynamic motion + drifting starfield",
         color=DIM, fontsize=8, ha="left", va="bottom", linespacing=1.5)

# ---------- ANIMATION LOOP ----------
total_frames = SECONDS * FPS


def draw_frame(k):
    p = k / total_frames
    
    # Independent speeds & timing
    azimuth = 30 + 360 * p + 10 * np.sin(2 * np.pi * p)
    elevation = 20 + 8 * np.cos(2 * np.pi * p * 1.5)
    ax.view_init(elev=elevation, azim=azimuth)

    # Gentle shifting/drifting for the background stars to make them feel alive
    drift_x = star_base_x + 0.4 * np.sin(2 * np.pi * p + star_phases)
    drift_y = star_base_y + 0.4 * np.cos(2 * np.pi * p * star_phases)
    star_sc._offsets3d = (drift_x, drift_y, star_base_z)

    w = 0.5 + 0.5 * np.cos(2 * np.pi * 3 * p)
    for sh in shells:
        sh.set_alpha(SHELL_XRAY + (SHELL_SOLID - SHELL_XRAY) * w)
    status.set_text("VIEW: SURFACE" if w > 0.5 else "VIEW: X-RAY ACTIVE")

    u = (fall_phase + 5 * p) % 1.0
    xs, ys, zs = [], [], []
    for j, (cx, cy, R, z0, depth, ang) in enumerate(falls):
        uu = u[j * NP:(j + 1) * NP]
        k_in = 1 - 0.85 * uu
        xs.append(cx + R * 0.95 * np.cos(ang) * k_in)
        ys.append(cy + R * 0.95 * np.sin(ang) * k_in)
        zs.append(z0 - depth * 0.9 * uu ** 0.9)
    fall_sc._offsets3d = (np.concatenate(xs), np.concatenate(ys), np.concatenate(zs))

    for n, sh in enumerate(ships):
        ang = 2 * np.pi * (p * (1.5 if n == 0 else -2.2) + 0.3 * n)
        rad = 16.0 if n == 0 else 11.0
        sh._offsets3d = ([rad * np.cos(ang)], [rad * np.sin(ang)], [4.0 + 2.5 * np.sin(2 * np.pi * p * (n + 1))])

    f = (1.5 * p) % 1.0
    t = np.linspace(0, 2 * np.pi, 120)
    scan.set_data(16.0 * np.cos(t), 16.0 * np.sin(t))
    scan.set_3d_properties(np.full_like(t, -10 + 20 * f))


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


if "--png" in sys.argv:
    for name, k in (("surface", 0), ("xray", total_frames // 4)):
        draw_frame(k)
        fig.savefig(f"pangaea_stars_test_{name}.png", facecolor=BG, dpi=100)
    print("saved starry test stills successfully.")
    sys.exit()

print(f"Rendering {total_frames} frames with dynamic stars -> {OUT_MP4}...")
with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as wr:
    for k in range(total_frames):
        draw_frame(k)
        wr.append_data(grab())
        if k % 30 == 0:
            print(f"Frame {k + 1}/{total_frames}")

draw_frame(total_frames // 4)
fig.savefig(OUT_PNG, facecolor=BG, dpi=100)
print(f"Done: {OUT_MP4} and {OUT_PNG}")

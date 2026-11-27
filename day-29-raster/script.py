#!/usr/bin/env python3
"""
#30DayMapChallenge 2026 | Day 29: Raster
"Aurora Borealis over Focșani"

A pure-pixel view of the northern auroral oval, seen from above the North Pole.
Every pixel is one cell of a 216 x 216 raster (~0.55 deg, ~60 km). The oval is
Sun-fixed, so the Earth spins underneath it: watch Romania roll through the
night side while the May 2024 G5 geomagnetic storm pushes the oval south.

Model: schematic, empirical-style auroral oval driven by Kp
  - geomagnetic frame: centred dipole, pole 80.65N 72.68W
  - oval centre colatitude moves equatorward with Kp, lower on the night side
  - arcs / rays = smoothed random fields stretched along magnetic longitude
It is NOT a forecast. For the real thing see NOAA SWPC OVATION.

Run:    python day29_raster_aurora.py            # MP4 + peak-frame PNG
        python day29_raster_aurora.py --still    # PNG only (fast)
Needs:  numpy, scipy, matplotlib, pillow, geopandas, ffmpeg
Input:  land.geojson (Natural Earth 50m land; downloaded automatically if missing)
"""
import argparse
import os
import urllib.request

import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
N_PIX = 216                 # raster is N x N pixels
LAT_MIN = 30.0              # edge of the view (deg N)
R_MAX = 90.0 - LAT_MIN      # radius of the view in degrees
N_FRAMES = 360              # 1 frame = 4 minutes -> exactly one Earth rotation
FPS = 30
UT_START = 12.0             # hours UT, 10 May 2024 (Historic G5 storm date)
DECL = 18.0                 # solar declination mid-May (deg)
POLE_LAT, POLE_LON = 80.65, -72.68   # geomagnetic north pole (dipole approx.)
FOCSANI = (45.70, 27.18)    # lat, lon
TZ_OFFSET = 3.0             # EEST, UTC+3 in May
LAND_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
            "master/geojson/ne_50m_land.geojson")

BG = "#05070d"
GREY = "#8b93a7"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(29)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la) + 0 * lo], -1)


Z_M = unit(POLE_LAT, POLE_LON)
Y_M = np.cross([0, 0, 1], Z_M)
Y_M /= np.linalg.norm(Y_M)
X_M = np.cross(Y_M, Z_M)
M_MAT = np.stack([X_M, Y_M, Z_M], 1)       # geographic -> geomagnetic


def to_mag(lat, lon):
    m = unit(lat, lon) @ M_MAT
    return np.degrees(np.arcsin(np.clip(m[..., 2], -1, 1))), np.degrees(np.arctan2(m[..., 1], m[..., 0])) % 360


def subsolar_lon(ut):
    return -15.0 * (ut - 12.0)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# Kp scenario (fast rise, slow recovery) -- peak when Focsani is at magnetic midnight
# ----------------------------------------------------------------------------
def focsani_mlt(ut):
    _, mlon_f = to_mag(*FOCSANI)
    _, mlon_s = to_mag(DECL, subsolar_lon(ut))
    return (12 + (mlon_f - mlon_s) / 15.0) % 24


_uts = np.linspace(UT_START, UT_START + 24, 1441)
_mlt = np.array([focsani_mlt(u) for u in _uts])
UT_PEAK = _uts[np.argmin(np.abs(((_mlt + 12) % 24) - 12))]   # MLT closest to 0 / 24


def kp_of(ut):
    d = ut - UT_PEAK
    sig = np.where(d < 0, 1.7, 3.6)
    return 1.3 + 7.7 * np.exp(-0.5 * (d / sig) ** 2)


# ----------------------------------------------------------------------------
# Land raster (0.3 deg land-fraction grid, anti-aliased by box averaging)
# ----------------------------------------------------------------------------
def build_land():
    path = os.path.join(OUT_DIR, "land.geojson")
    if not os.path.exists(path):
        print("Downloading Natural Earth land ...")
        urllib.request.urlretrieve(LAND_URL, path)
    g = gpd.read_file(path)
    res = 0.1
    W, H = int(360 / res), int(90 / res)           # lat 0..90
    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)

    def draw(poly):
        pts = [((x + 180) / res, (90 - y) / res) for x, y in poly.exterior.coords]
        dr.polygon(pts, fill=255)

    for geom in g.geometry:
        for poly in getattr(geom, "geoms", [geom]):
            draw(poly)
    img = img.resize((W // 3, H // 3), Image.BOX)
    return np.asarray(img, dtype=np.float32) / 255.0   # 0.3 deg cells


LAND = build_land()
LH, LW = LAND.shape


def land_fraction(lat, lon):
    r = np.clip(((90 - lat) / 0.3).astype(int), 0, LH - 1)
    c = (((lon + 180) % 360) / 0.3).astype(int) % LW
    return LAND[r, c]


# ----------------------------------------------------------------------------
# Pixel grid (view from above the pole; Sun at the top)
# ----------------------------------------------------------------------------
xs = (np.arange(N_PIX) + 0.5) / N_PIX * 2 - 1
X, Y = np.meshgrid(xs, -xs)                         # row 0 = top
RHO = np.hypot(X, Y)
INSIDE = RHO <= 1.0
LAT = 90.0 - np.clip(RHO, 0, 1) * R_MAX
HANG = np.degrees(np.arctan2(-X, Y))                # hour angle: counter-clockwise from the Sun


# ----------------------------------------------------------------------------
# Noise (arcs + rays) in magnetic lon / lat, evolving in time
# ----------------------------------------------------------------------------
NLON, NLAT = 360, 90
print("Building aurora noise ...")
arcs = gaussian_filter(rng.standard_normal((N_FRAMES, NLON, NLAT)).astype(np.float32),
                       sigma=(14, 11, 1.1), mode=("wrap", "wrap", "nearest"))
rays = gaussian_filter(rng.standard_normal((N_FRAMES, NLON, NLAT)).astype(np.float32),
                       sigma=(5, 2.2, 0.6), mode=("wrap", "wrap", "nearest"))


def norm01(a):
    lo, hi = np.percentile(a, [3, 97])
    return np.clip((a - lo) / (hi - lo), 0, 1)


NOISE = (0.65 * norm01(arcs) + 0.35 * norm01(rays)).astype(np.float32)
del arcs, rays


# ----------------------------------------------------------------------------
# Aurora intensity at any (lat, lon) -- works on arrays or single points
# ----------------------------------------------------------------------------
def aurora(lat, lon, ut, kp, fi):
    mlat, mlon = to_mag(lat, lon)
    _, mlon_s = to_mag(DECL, subsolar_lon(ut))
    mlt = (12 + (mlon - mlon_s) / 15.0) % 24
    c = np.cos(np.radians(mlt * 15.0))              # +1 at magnetic midnight
    center = 70.0 - 2.6 * kp - 3.0 * c
    w = 2.0 + 0.3 * kp
    s = (mlat - center) / w
    nf = 0.30 + 0.70 * ((1 + c) / 2) ** 1.5
    dm = (mlt - 23.3 + 12) % 24 - 12
    pulse = 0.5 + 0.5 * np.sin(2 * np.pi * fi / 45.0)
    bulge = 0.6 * (kp / 9.0) ** 2 * pulse * np.exp(-0.5 * (dm / 1.8) ** 2)
    amp = 0.30 + 0.70 * (kp / 9.0) ** 0.8
    base = amp * (nf + bulge) * np.exp(-0.5 * s ** 2)
    il = (mlon % 360).astype(int) % NLON
    ia = np.clip(((mlat - 30.0) / 60.0 * NLAT).astype(int), 0, NLAT - 1)
    n = NOISE[fi % N_FRAMES][il, ia]
    I = np.clip(base * (0.25 + 1.15 * n ** 1.3) * 1.5, 0, 1.2)
    return I, s, mlt


def sun_elev_sin(lat, lon, ut):
    h = np.radians(lon - subsolar_lon(ut))
    la, d = np.radians(lat), np.radians(DECL)
    return np.sin(la) * np.sin(d) + np.cos(la) * np.cos(d) * np.cos(h)


# ----------------------------------------------------------------------------
# Compose one frame -> RGB image
# ----------------------------------------------------------------------------
C_GREEN = np.array([0.12, 1.00, 0.42])
C_RED = np.array([1.00, 0.10, 0.32])
C_TEAL = np.array([0.10, 0.65, 1.00])

OCEAN_N, LAND_N = np.array([0.012, 0.022, 0.050]), np.array([0.040, 0.062, 0.095])
OCEAN_D, LAND_D = np.array([0.040, 0.105, 0.210]), np.array([0.200, 0.270, 0.330])


def render_rgb(fi):
    ut = UT_START + fi * 24.0 / N_FRAMES
    kp = float(kp_of(ut))
    lon = subsolar_lon(ut) + HANG
    lf = land_fraction(LAT, lon)[..., None]
    se = sun_elev_sin(LAT, lon, ut)
    day = smoothstep(-0.10, 0.22, se)[..., None]
    twi = np.exp(-(se / 0.09) ** 2)[..., None] * 0.5

    night_col = OCEAN_N * (1 - lf) + LAND_N * lf
    day_col = OCEAN_D * (1 - lf) + LAND_D * lf
    ground = night_col * (1 - day) + day_col * day + twi * np.array([0.20, 0.06, 0.08])

    I, s, _ = aurora(LAT, lon, ut, kp, fi)
    rw = np.clip((kp / 9.0) ** 1.2 * 0.9 - 0.30 * s, 0, 1)
    tw = np.clip(0.45 * s, 0, 0.7)
    gw = np.clip(1 - rw - tw, 0, 1)
    col = gw[..., None] * C_GREEN + rw[..., None] * C_RED + tw[..., None] * C_TEAL
    vis = (1 - 0.85 * day)                           # aurora washed out in daylight
    glow = I[..., None] * col * vis
    glow += np.clip(I - 0.85, 0, 1)[..., None] * 0.7 * vis   # white-hot cores

    rgb = np.clip(ground + glow, 0, 1)
    rgb[~INSIDE] = matplotlib.colors.to_rgb(BG)
    return rgb, ut, kp


def focsani_state(fi, ut, kp):
    """aurora strength at Focsani / on its northern horizon, and darkness."""
    lat, lon = np.array(FOCSANI[0]), np.array(FOCSANI[1])
    vals = []
    for dlat in (0.0, 2.5, 5.0):
        I, _, _ = aurora(lat + dlat, lon, ut, kp, fi)
        vals.append(float(I) * (0.85 if dlat else 1.0))
    dark = float(sun_elev_sin(lat, lon, ut)) < np.sin(np.radians(-6))
    return max(vals), dark


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
KP_COLS = ["#27d3a2", "#27d3a2", "#27d3a2", "#27d3a2", "#c6e03a",
           "#f2c230", "#f28a30", "#f2552e", "#ff2d6f"]
G_SCALE = {5: "G1 minor", 6: "G2 moderate", 7: "G3 strong", 8: "G4 severe", 9: "G5 extreme"}

fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)

# Shorter, downsized title & streamlined subtitle
fig.text(0.5, 0.972, "Aurora Borealis over Focșani", color="white", fontsize=23,
         fontweight="bold", ha="center", va="center")
fig.text(0.5, 0.947, "10–11 May 2024 G5 geomagnetic storm",
         color=GREY, fontsize=14, ha="center", va="center")

ax = fig.add_axes([0.08, 0.262, 0.84, 0.84 * 10.8 / 13.5])
ax.set_facecolor(BG)
ax.set_xlim(-1.06, 1.06)
ax.set_ylim(-1.06, 1.06)
ax.axis("off")
im = ax.imshow(np.zeros((N_PIX, N_PIX, 3)), extent=[-1, 1, -1, 1], interpolation="nearest", zorder=1)

for lat_ring in (60, 45):
    r = (90 - lat_ring) / R_MAX
    ax.add_patch(Circle((0, 0), r, fill=False, ec="white", lw=0.8, ls=(0, (2, 4)), alpha=0.28, zorder=3))
    ax.text(0.012, r + 0.012, f"{lat_ring}°N", color="white", alpha=0.45, fontsize=9, zorder=3)
ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#2b3350", lw=1.6, zorder=3))
ax.text(0, 1.035, "toward the Sun", color="#f2c230", fontsize=11, ha="center", va="bottom", alpha=0.9)
ax.text(0, -1.035, "night side (magnetic midnight)", color="#6ea8ff", fontsize=11, ha="center",
        va="top", alpha=0.9)
ax.text(-1.04, 0, "dusk", color=GREY, fontsize=12, ha="right", va="center", rotation=90)
ax.text(1.04, 0, "dawn", color=GREY, fontsize=12, ha="left", va="center", rotation=-90)

mark = ax.scatter([], [], s=150, marker="s", facecolors="none", edgecolors="white", linewidths=1.8, zorder=5)
mark_lbl = ax.text(0, 0, "Focșani", color="white", fontsize=12, fontweight="bold", zorder=6,
                   path_effects=None)

# HUD
hud_time = fig.text(0.08, 0.222, "", color="white", fontsize=16, family="DejaVu Sans Mono",
                    ha="left", va="center")
hud_foc = fig.text(0.92, 0.222, "", color="white", fontsize=16, family="DejaVu Sans Mono",
                   ha="right", va="center")
hud_status = fig.text(0.92, 0.195, "", fontsize=14, fontweight="bold", ha="right", va="center")
hud_kp = fig.text(0.08, 0.195, "", color="white", fontsize=14, fontweight="bold", ha="left", va="center")

ax_kp = fig.add_axes([0.08, 0.158, 0.84, 0.022])
ax_kp.set_xlim(0, 9)
ax_kp.set_ylim(0, 1)
ax_kp.axis("off")
kp_blocks = []
for i in range(9):
    p = Rectangle((i + 0.06, 0.05), 0.88, 0.9, fc="#161b2b", ec="none")
    ax_kp.add_patch(p)
    kp_blocks.append(p)

ax_ts = fig.add_axes([0.08, 0.075, 0.84, 0.065], facecolor="none")
uts_all = UT_START + np.arange(N_FRAMES) * 24.0 / N_FRAMES
kp_all = kp_of(uts_all)
dark_all = np.array([float(sun_elev_sin(np.array(FOCSANI[0]), np.array(FOCSANI[1]), u))
                     < np.sin(np.radians(-6)) for u in uts_all])
ax_ts.fill_between(uts_all, 0, 9.5, where=dark_all, color="#1a2a55", alpha=0.75, lw=0, step="mid")
ax_ts.plot(uts_all, kp_all, color="#27d3a2", lw=2)
play = ax_ts.axvline(UT_START, color="white", lw=1.4)
dot, = ax_ts.plot([UT_START], [kp_all[0]], "s", color="white", ms=6)
ax_ts.set_xlim(UT_START, UT_START + 24)
ax_ts.set_ylim(0, 9.5)
for sp in ax_ts.spines.values():
    sp.set_visible(False)
ax_ts.set_yticks([])
ax_ts.set_xticks(np.arange(UT_START, UT_START + 25, 4))
ax_ts.set_xticklabels([f"{int(t % 24):02d}" for t in np.arange(UT_START, UT_START + 25, 4)])
ax_ts.tick_params(colors=GREY, labelsize=9, length=0)

# Rotated vertical "Hours UT" label placed nicely next to the time series graph
fig.text(0.04, 0.107, "Hours UT", color=GREY, fontsize=9.5, ha="center", va="center", rotation=90)

fig.text(0.08, 0.145, "Kp index  ·  shaded = night in Focșani (sun > 6° below horizon)", color=GREY,
         fontsize=9.5, ha="left", va="center")

# Footer metadata
fig.text(0.5, 0.038, "#30DayMapChallenge  |  Day 29: Raster  |  Sara AHARON", color="white",
         fontsize=11.5, ha="center", va="center")
fig.text(0.5, 0.012, "Data: schematic Kp-driven auroral-oval model on a 216×216 pixel grid "
         "(dipole pole 80.65°N 72.68°W)  ·  Natural Earth", color=GREY, fontsize=11.5,
         ha="center", va="center")


def to_xy(lat, lon, ut):
    h = np.radians(lon - subsolar_lon(ut))
    r = (90 - lat) / R_MAX
    return -r * np.sin(h), r * np.cos(h)


def update(fi):
    rgb, ut, kp = render_rgb(fi)
    im.set_data(rgb)

    fx, fy = to_xy(*FOCSANI, ut)
    mark.set_offsets([[fx, fy]])
    mark_lbl.set_position((fx + 0.045, fy - 0.01))

    hh = int(ut % 24)
    mm = int(round((ut % 1) * 60)) % 60
    loc = (ut + TZ_OFFSET) % 24
    hud_time.set_text(f"{hh:02d}:{mm:02d} UT")
    hud_foc.set_text(f"Focșani {int(loc):02d}:{int(round((loc % 1) * 60)) % 60:02d} EEST")

    g = int(np.floor(kp + 1e-6))
    gtxt = G_SCALE.get(g, "quiet" if kp < 4.5 else "")
    hud_kp.set_text(f"Kp {kp:3.1f}  {gtxt}")
    hud_kp.set_color(KP_COLS[min(8, max(0, g - 1))] if g >= 1 else "#27d3a2")
    for i, p in enumerate(kp_blocks):
        f = float(np.clip(kp - i, 0, 1))
        base = np.array(matplotlib.colors.to_rgb(KP_COLS[i]))
        p.set_facecolor(tuple(np.array(matplotlib.colors.to_rgb("#161b2b")) * (1 - f) + base * f))

    strength, dark = focsani_state(fi, ut, kp)
    if not dark:
        txt, col = "daylight / twilight", "#f2c230"
    elif strength > 0.30:
        txt, col = "AURORA OVERHEAD", "#ff4f8b"
    elif strength > 0.12:
        txt, col = "glow on the horizon", "#27d3a2"
    else:
        txt, col = "dark sky, no aurora", GREY
    hud_status.set_text(txt)
    hud_status.set_color(col)

    play.set_xdata([ut, ut])
    dot.set_data([ut], [kp])
    return im,


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", action="store_true", help="only write the peak-frame PNG")
    args = ap.parse_args()

    peak_fi = int(round((UT_PEAK - UT_START) / 24.0 * N_FRAMES)) % N_FRAMES
    print(f"Kp peak at {UT_PEAK % 24:5.2f} UT  (frame {peak_fi})")

    update(peak_fi)
    png = os.path.join(OUT_DIR, "day29_raster_aurora.png")
    fig.savefig(png, dpi=100, facecolor=BG)
    print("Saved", png)
    if args.still:
        return

    mp4 = os.path.join(OUT_DIR, "day29_raster_aurora.mp4")
    try:
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    writer = animation.FFMpegWriter(fps=FPS, bitrate=9000,
                                    extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, mp4, dpi=100):
        for fi in range(N_FRAMES):
            update(fi)
            writer.grab_frame()
            if fi % 30 == 0:
                print(f"  frame {fi}/{N_FRAMES}")
    print("Saved", mp4)


if __name__ == "__main__":
    main()

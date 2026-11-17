"""
#30DayMapChallenge | Day 15: Inside out | Sara AHARON
Synthwave / cyberpunk edition: looking out from Earth's core.
Azimuthal Equidistant projection: the ENTIRE globe (0-180 deg from the center)
fills the circle edge-to-edge (r = 1.0 is the antipode).

Run: python amis.py
Requires: pip install geopandas matplotlib numpy imageio imageio-ffmpeg shapely>=2
"""
import math
import urllib.request
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from shapely import make_valid
from shapely.geometry import (LineString, MultiPolygon, Point, Polygon)

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
CENTER_LAT, CENTER_LON = 45.9432, 24.9668  # Romania center
FPS = 30
SECONDS = 16
HOLD_START, HOLD_END = 1.0, 2.0
SIZE_W, SIZE_H = 1080, 1350                 # 4:5 portrait
PREVIEW_ONLY = False                        # True = save 1 frame (day15_preview.png)
OUT_MP4 = "day15_magma_hollow_earth.mp4"
OUT_PNG = "day15_magma_hollow_earth_final.png"

# Synthwave / Cyberpunk palette
BG = "#0b0220"          # deep indigo-black
DISK_BG = "#14052e"     # slightly lighter interior
LAND_FILL = "#2a0a52"   # dark violet
LAND_EDGE = "#00f0ff"   # neon cyan
RIM_PINK = "#ff2e97"    # hot pink
RIM_PURPLE = "#8a2be2"  # electric purple
GRID_COLOR = "#b14dff"  # neon purple
ACCENT = "#ffb347"      # sunset orange
TEXT_DIM = "#9d8cc9"
NE_FILE = Path("ne_10m_admin_0_countries.geojson")
SEGMENT_DEG = 1.0       # densification so long edges follow great circles
# ----------------------------

if not NE_FILE.exists():
    print("Downloading Natural Earth vector boundaries...")
    url = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
           "master/geojson/ne_10m_admin_0_countries.geojson")
    urllib.request.urlretrieve(url, NE_FILE)

world = gpd.read_file(NE_FILE)

DISK = Point(0, 0).buffer(1.0, 256)


def inside_out_project_full(lons, lats, center_lon, center_lat):
    """
    Azimuthal Equidistant, inside-out (east/west mirrored).
    Angular distance c in [0, pi] maps to r = c / pi in [0, 1],
    so the antipode sits exactly on the outer circle.
    """
    lam0 = math.radians(center_lon)
    phi0 = math.radians(center_lat)

    lam = np.radians(lons)
    phi = np.radians(lats)
    dlam = lam - lam0

    cos_c = np.sin(phi0) * np.sin(phi) + np.cos(phi0) * np.cos(phi) * np.cos(dlam)
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))  # 0..pi

    x = np.cos(phi) * np.sin(dlam)
    y = np.cos(phi0) * np.sin(phi) - np.sin(phi0) * np.cos(phi) * np.cos(dlam)

    x = -x  # mirror reversal (inside-out view)

    r_xy = np.hypot(x, y)
    r_xy = np.where(r_xy < 1e-12, 1e-12, r_xy)

    r_scaled = c / math.pi  # FIX: 180 deg -> r = 1.0

    return (x / r_xy) * r_scaled, (y / r_xy) * r_scaled


def _polys_only(g):
    """Keep only polygonal parts of a (possibly mixed) geometry."""
    if g is None or g.is_empty:
        return None
    if g.geom_type == "Polygon":
        return g
    if g.geom_type == "MultiPolygon":
        return g
    if hasattr(g, "geoms"):
        parts = []
        for p in g.geoms:
            q = _polys_only(p)
            if q is None:
                continue
            parts.extend(q.geoms if q.geom_type == "MultiPolygon" else [q])
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else MultiPolygon(parts)
    return None


def _ring(coords, clon, clat):
    arr = np.asarray(coords)
    px, py = inside_out_project_full(arr[:, 0], arr[:, 1], clon, clat)
    return np.column_stack([px, py])


def project_polygon(poly, clon, clat):
    poly = poly.segmentize(SEGMENT_DEG)
    ext = _ring(poly.exterior.coords, clon, clat)
    if len(ext) < 4:
        return None
    holes = [_ring(r.coords, clon, clat) for r in poly.interiors if len(r.coords) >= 4]

    # Antipode inside this polygon: the polygon projects to the area BETWEEN
    # its projected ring and the outer circle, not to the area inside the ring.
    anti_lon = ((clon + 180.0 + 180.0) % 360.0) - 180.0
    anti = Point(anti_lon, -clat)
    minx, miny, maxx, maxy = poly.bounds
    if minx <= anti.x <= maxx and miny <= anti.y <= maxy and poly.contains(anti):
        shape = make_valid(Polygon(ext))
        return _polys_only(DISK.difference(shape))

    shape = make_valid(Polygon(ext, holes))
    return _polys_only(shape.intersection(DISK))


def project_geometry(geom, clon, clat):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return project_polygon(geom, clon, clat)
    if geom.geom_type == "MultiPolygon":
        parts = []
        for p in geom.geoms:
            q = project_polygon(p, clon, clat)
            if q is None:
                continue
            parts.extend(q.geoms if q.geom_type == "MultiPolygon" else [q])
        if not parts:
            return None
        return MultiPolygon(parts)
    return None


def build_inside_graticule():
    lines = []
    for lon in range(-180, 180, 20):
        lats = np.linspace(-90, 90, 181)
        lines.append(np.column_stack([np.full_like(lats, lon), lats]))
    for lat in range(-80, 85, 20):
        lons = np.linspace(-180, 180, 361)
        lines.append(np.column_stack([lons, np.full_like(lons, lat)]))
    return lines


graticules = build_inside_graticule()

# Figure Setup
fig = plt.figure(figsize=(SIZE_W / 100, SIZE_H / 100), dpi=100)
fig.patch.set_facecolor(BG)

ax = fig.add_axes([0.05, 0.15, 0.90, 0.72])
ax.set_facecolor(BG)
ax.set_aspect("equal")
ax.axis("off")

glow_text = [pe.withStroke(linewidth=6, foreground=RIM_PINK, alpha=0.35)]
fig.text(0.08, 0.93, "INSIDE OUT: CORE PERSPECTIVE", color="white",
         fontsize=26, fontweight="bold", ha="left", va="center",
         path_effects=glow_text)
fig.text(0.08, 0.895, "Looking outward from the molten core — Full Global Extent",
         color=LAND_EDGE, fontsize=12, fontweight="bold", ha="left", va="center")

txt_rotation = fig.text(0.08, 0.12, "", color=ACCENT, fontsize=13, fontweight="bold")

fig.text(0.08, 0.03,
         f"#30DayMapChallenge | Day 15: Inside out | {AUTHOR}\n"
         "Concept: Observer at Earth's core ($r=0$) looking out. "
         "Full globe stretched to the neon rim.",
         color=TEXT_DIM, fontsize=8, ha="left", va="bottom", linespacing=1.4)


def render_frame(frame_idx, total_frames):
    ax.cla()
    ax.set_facecolor(BG)
    ax.axis("off")

    spin_deg = (frame_idx / total_frames) * 360.0
    clon = CENTER_LON + spin_deg

    # Neon rim: stacked translucent rings = glow
    for lw, a in [(26, 0.05), (18, 0.08), (11, 0.14), (6, 0.35)]:
        ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=RIM_PINK,
                            linewidth=lw, alpha=a, zorder=1))
    ax.add_patch(Circle((0, 0), 1.0, facecolor=DISK_BG, edgecolor=RIM_PINK,
                        linewidth=2.5, zorder=2))
    ax.add_patch(Circle((0, 0), 1.035, facecolor="none", edgecolor=RIM_PURPLE,
                        linewidth=1.2, alpha=0.9, zorder=2))

    # Distance rings from the core (every 45 deg of arc)
    for frac in (0.25, 0.5, 0.75):
        ax.add_patch(Circle((0, 0), frac, facecolor="none", edgecolor=RIM_PURPLE,
                            linewidth=0.6, alpha=0.35, linestyle=(0, (2, 4)), zorder=2))

    # Graticule (broken where a line jumps across the antipode)
    for line in graticules:
        px, py = inside_out_project_full(line[:, 0], line[:, 1], clon, CENTER_LAT)
        jump = np.hypot(np.diff(px), np.diff(py)) > 0.3
        px = px.copy()
        py = py.copy()
        px[1:][jump] = np.nan
        py[1:][jump] = np.nan
        ax.plot(px, py, color=GRID_COLOR, lw=0.5, alpha=0.30, zorder=3)

    # Countries
    for _, row in world.iterrows():
        geom_proj = project_geometry(row.geometry, clon, CENTER_LAT)
        if geom_proj is not None and not geom_proj.is_empty:
            gpd.GeoSeries([geom_proj]).plot(
                ax=ax, facecolor=LAND_FILL, edgecolor=LAND_EDGE,
                linewidth=0.7, alpha=0.95, zorder=4)

    # Core marker (observer at r = 0)
    ax.plot([0], [0], marker="o", color=ACCENT, markersize=5, zorder=6)
    ax.add_patch(Circle((0, 0), 0.025, facecolor="none", edgecolor=ACCENT,
                        linewidth=1, alpha=0.7, zorder=6))

    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)

    txt_rotation.set_text(f"Core Azimuth: {clon % 360:.1f}° | Full Globe Extent Active")


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


if PREVIEW_ONLY:
    render_frame(0, 100)
    fig.savefig("day15_preview.png", facecolor=BG, dpi=100)
    print("Saved day15_preview.png")
    raise SystemExit

total_frames = SECONDS * FPS
schedule = ([0] * int(HOLD_START * FPS) + list(range(total_frames))
            + [total_frames - 1] * int(HOLD_END * FPS))

print(f"Rendering synthwave inside-out animation ({len(schedule)} frames) to {OUT_MP4}...")
with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as w:
    for i, f_idx in enumerate(schedule):
        render_frame(f_idx, total_frames)
        w.append_data(grab())
        if i % 30 == 0:
            print(f"Frame {i + 1}/{len(schedule)}")

fig.savefig(OUT_PNG, facecolor=BG, dpi=100)
print(f"Saved {OUT_MP4} and {OUT_PNG}")

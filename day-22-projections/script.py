"""
#30DayMapChallenge 2026 - Day 22: Projections
"Me at the center": azimuthal equidistant projection centred on Focsani.
The view zooms out from ~250 km to the whole planet. Distance rings are true geodesic
distances; Tissot circles (600 km radius on the ground) show shape distortion growing
with distance. The antipode is not a point here: it is the whole outer rim.

Needs: geopandas, pyproj, shapely>=2, matplotlib, numpy, ffmpeg
Data : Natural Earth 50m admin-0 countries (geojson)
Usage: python day22_me_at_the_center.py [countries.geojson] [--frame N]
Edit PLACES below to add the places that matter to you.
"""
import sys
import numpy as np
import geopandas as gpd
import shapely
from pyproj import Transformer, Geod
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import PolyCollection, LineCollection, PatchCollection
from matplotlib.patches import PathPatch, Circle
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize

SRC = next((a for a in sys.argv[1:] if a.endswith(".geojson")), "ne50.geojson")
ONLY = int(sys.argv[sys.argv.index("--frame") + 1]) if "--frame" in sys.argv else None
OUT = "day22_me_at_the_center.mp4"

# ---------- style ----------
BG, CYAN, MAG, YEL, WHITE, GREY = "#0a0a12", "#00f0ff", "#ff2bd6", "#ffe600", "#f2f2f7", "#6b6b80"
plt.rcParams["font.family"] = "DejaVu Sans Mono"
FPS = 30
FW, FH = 10.8, 13.5
CMAP = LinearSegmentedColormap.from_list("neon", [CYAN, "#7a5cff", MAG, YEL])
NORM = Normalize(0, 2.9)  # log2(stretch)

# ---------- places ----------
HOME = dict(name="FOCȘANI", lon=27.1864, lat=45.6967)
PLACES = [
    dict(name="Cape Town", lon=18.42, lat=-33.92),
    dict(name="Drake Passage", lon=-65.0, lat=-58.5),
    dict(name="Easter Island", lon=-109.37, lat=-27.11),
    dict(name="Chatham Islands", lon=-176.12, lat=-44.27, note="farthest land from home"),
]
RING_KM = [100, 200, 500, 1000, 2000, 5000, 7500, 10000, 15000]
TISSOT_R = 600_000.0
R_KM = 6371.0

geod = Geod(ellps="WGS84")
AEQD = f"+proj=aeqd +lat_0={HOME['lat']} +lon_0={HOME['lon']} +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
tr = Transformer.from_crs("EPSG:4326", AEQD, always_xy=True)

def fwd(c):
    x, y = tr.transform(c[:, 0], c[:, 1])
    return np.c_[x, y] / 1000.0  # km

# ---------- land ----------
world = gpd.read_file(SRC)
col = "ADMIN" if "ADMIN" in world.columns else ("NAME" if "NAME" in world.columns else world.columns[0])
world = world[world[col] != "Antarctica"]
land = shapely.segmentize(world.geometry.union_all(), 0.5)

def polys_of(g):
    return list(g.geoms) if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]

def runs(coords):
    """Polylines of a lon/lat ring, dropping the artificial cuts along lon=+-180."""
    out, cur = [], [coords[0]]
    for a, b in zip(coords[:-1], coords[1:]):
        if abs(a[0]) > 179.9999 and abs(b[0]) > 179.9999:
            if len(cur) > 1:
                out.append(np.array(cur))
            cur = [b]
        else:
            cur.append(b)
    if len(cur) > 1:
        out.append(np.array(cur))
    return out

def make_path(rings):
    verts, codes = [], []
    for ring in rings:
        n = len(ring)
        verts += list(ring) + [ring[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
    return Path(np.array(verts), codes)

patches, coast = [], []
for p in polys_of(land):
    if p.geom_type != "Polygon" or p.is_empty:
        continue
    rings = [np.array(p.exterior.coords)] + [np.array(i.coords) for i in p.interiors]
    patches.append(PathPatch(make_path([fwd(r) for r in rings])))
    for r in rings:
        coast += [fwd(rn) for rn in runs(r)]

borders = []
for g in world.geometry:
    for p in polys_of(shapely.segmentize(g, 0.5)):
        if p.geom_type != "Polygon" or p.is_empty:
            continue
        for rn in runs(np.array(p.exterior.coords)):
            borders.append(fwd(rn))
print("land patches:", len(patches), "coast runs:", len(coast), "border runs:", len(borders))

# ---------- rim, places, Tissot ----------
x_, y_, _ = geod.fwd(HOME["lon"], HOME["lat"], 0, 20_003_000.0)
RIM = float(np.hypot(*(fwd(np.array([[x_, y_]]))[0])))

for P in PLACES:
    az, _, d = geod.inv(HOME["lon"], HOME["lat"], P["lon"], P["lat"])
    P["dist"] = d / 1000.0
    P["xy"] = fwd(np.array([[P["lon"], P["lat"]]]))[0]

az = np.linspace(0, 360, 73)
cv, kk, dd = [], [], []
def add_circle(clon, clat, dkm):
    x, y, _ = geod.fwd(np.full_like(az, clon), np.full_like(az, clat), az, np.full_like(az, TISSOT_R))
    cv.append(fwd(np.c_[x, y]))
    th = dkm / R_KM
    kk.append(th / np.sin(th) if th > 1e-6 else 1.0)
    dd.append(dkm)
add_circle(HOME["lon"], HOME["lat"], 0)
for ring_i, dkm in enumerate(range(2500, 17501, 2500)):
    n = max(6, int(round(2 * np.pi * R_KM * np.sin(dkm / R_KM) / 2400)))
    off = (ring_i % 2) * 0.5
    for j in range(n):
        bearing = (j + off) * 360 / n
        clon, clat, _ = geod.fwd(HOME["lon"], HOME["lat"], bearing, dkm * 1000.0)
        add_circle(clon, clat, dkm)
cv, kk, dd = np.array(cv), np.array(kk), np.array(dd)
cen = cv[:, :-1].mean(1)
tcol = CMAP(NORM(np.log2(np.clip(kk, 1, None))))
print("tissot circles:", len(cv), "max stretch:", round(kk.max(), 1), "rim km:", round(RIM))

# ---------- timeline ----------
W0, W1 = 300.0, 20600.0
T0, T1 = 0.9, 11.6
HOLD_END = 16.5
NF = int(HOLD_END * FPS)
smooth = lambda t: (lambda u: u * u * (3 - 2 * u))(np.clip(t, 0, 1))
def W_of(now):
    return W0 * (W1 / W0) ** smooth((now - T0) / (T1 - T0))

# first moment each place becomes visible (view reaches 1.25x its distance, capped near the rim)
for P in PLACES:
    thr = min(1.25 * P["dist"], 0.985 * W1)
    P["t_on"] = next((f / FPS for f in range(NF) if W_of(f / FPS) >= thr), T1)

# ---------- figure ----------
fig = plt.figure(figsize=(FW, FH), dpi=100, facecolor=BG)
ax = fig.add_axes([0.0625, 0.165, 0.875, 0.70]); ax.set_facecolor(BG); ax.axis("off"); ax.set_aspect("equal")

pc = PatchCollection(patches, facecolor=CYAN, edgecolor="none", alpha=0.09); ax.add_collection(pc)
bc = LineCollection(borders, colors="#5fa8b8", linewidths=0.5, alpha=0.0); ax.add_collection(bc)
lc = LineCollection(coast, colors=CYAN, linewidths=0.9, alpha=0.95); ax.add_collection(lc)
lc2 = LineCollection(coast, colors=CYAN, linewidths=3.2, alpha=0.12); ax.add_collection(lc2)
TIS_FACE, TIS_EDGE = 0.14, 0.65  # transparency of the Tissot circles
tis = PolyCollection(cv, facecolors=np.c_[tcol[:, :3], np.full(len(cv), TIS_FACE)], edgecolors=tcol, linewidths=1.0)
ax.add_collection(tis)
ax.add_patch(Circle((0, 0), RIM, fill=False, ec=GREY, lw=1.0, ls=(0, (3, 4)), alpha=0.7))
sweep = Circle((0, 0), 1, fill=False, ec=MAG, lw=2.0, alpha=0.0); ax.add_patch(sweep)

rings = []
for r in RING_KM:
    c = Circle((0, 0), r, fill=False, ec=WHITE, lw=0.9, alpha=0.0); ax.add_patch(c)
    lab = ax.text(r * np.cos(0.22 * np.pi), r * np.sin(0.22 * np.pi), f"{r:,} km", color=WHITE, fontsize=10,
                  ha="left", va="bottom", alpha=0.0)
    rings.append((r, c, lab))
rim_lab = ax.text(0, RIM, "the antipode is this whole circle · 20,004 km", color=GREY, fontsize=10,
                  ha="center", va="bottom", alpha=0.0)

clip = Circle((0.5, 0.5), 0.5, transform=ax.transAxes)
for art in (pc, bc, lc, lc2, tis, sweep):
    art.set_clip_path(clip)
for _, c_, _l in rings:
    c_.set_clip_path(clip)
view_ring = Circle((0.5, 0.5), 0.5, transform=ax.transAxes, fill=False, ec=GREY, lw=0.8, alpha=0.5)
ax.add_patch(view_ring)

home_dot, = ax.plot(0, 0, "o", color=YEL, ms=8, zorder=6)
home_lab = ax.annotate(HOME["name"], xy=(0, 0), xytext=(11, -17), textcoords="offset points",
                       color=YEL, fontsize=14, fontweight="bold", zorder=6)
for P in PLACES:
    P["line"], = ax.plot([0, 0], [0, 0], color=MAG, lw=1.4, alpha=0.0, zorder=4)
    P["dot"], = ax.plot(*P["xy"], "o", color=MAG, ms=7, alpha=0.0, zorder=6)
    side = -1 if P["xy"][0] > 0.35 * W1 else 1  # labels near the rim lean towards the centre
    txt = f'{P["name"]}\n{P["dist"]:,.0f} km' + (f'\n{P["note"]}' if P.get("note") else "")
    P["lab"] = ax.annotate(txt, xy=P["xy"], xytext=(10 * side, 6 if P["xy"][1] >= 0 else -6), textcoords="offset points",
                           color=WHITE, fontsize=12, alpha=0.0, zorder=6, linespacing=1.3,
                           ha="left" if side == 1 else "right", va="bottom" if P["xy"][1] >= 0 else "top",
                           multialignment="left" if side == 1 else "right")

# ---------- text ----------
T = [fig.text(0.04, 0.975, "Me at the center", color=WHITE, fontsize=27, fontweight="bold", va="top"),
     fig.text(0.04, 0.927, "Everything is measured from where you are.", color=GREY, fontsize=12.5, va="top"),
     fig.text(0.04, 0.906, "Distances stay true. Shapes don't. Each circle is 1,200 km across.", color=GREY, fontsize=12.5, va="top")]
lab_l = fig.text(0.04, 0.152, "VISIBLE RADIUS", color=GREY, fontsize=10, va="top")
val_l = fig.text(0.04, 0.132, "", color=WHITE, fontsize=24, fontweight="bold", va="top")
lab_r = fig.text(0.96, 0.152, "SHAPE STRETCH AT THE EDGE", color=GREY, fontsize=10, va="top", ha="right")
val_r = fig.text(0.96, 0.132, "", color=MAG, fontsize=24, fontweight="bold", va="top", ha="right")
bar_ax = fig.add_axes([0.04, 0.078, 0.34, 0.009]); bar_ax.axis("off")
bar_ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto", extent=[0, 1, 0, 1])
fig.text(0.04, 0.066, "×1 true", color=CYAN, fontsize=9.5, va="top")
fig.text(0.38, 0.066, "×7+ stretched", color=YEL, fontsize=9.5, va="top", ha="right")
fig.text(0.04, 0.035, "Azimuthal equidistant · centred on Focșani", color=GREY, fontsize=9.5, va="top")
fig.text(0.96, 0.055, "Data: Natural Earth | Made with pyproj & matplotlib", color=GREY, fontsize=9.5, va="top", ha="right")
fig.text(0.96, 0.035, "#30DayMapChallenge | Day 22: Projections | Sara AHARON", color=WHITE, fontsize=10, va="top", ha="right")

def draw(f):
    now = f / FPS
    W = W_of(now)
    ax.set_xlim(-W, W); ax.set_ylim(-W, W)

    view_ring.set_alpha(0.5 * smooth(now / 0.6) * (1 - smooth((W - 12000) / 6000)))
    for t in T:
        t.set_alpha(smooth(now / 0.6))
    # borders fade as we zoom out
    bc.set_alpha(0.55 * smooth(now / 0.6) * (1 - smooth((W - 1500) / 3500)))
    pc.set_alpha(0.09 * smooth(now / 0.6)); lc.set_alpha(0.95 * smooth(now / 0.6)); lc2.set_alpha(0.12 * smooth(now / 0.6))

    for r, c, lab in rings:
        fr = r / W
        a = smooth((fr - 0.07) / 0.08) * smooth((1.0 - fr) / 0.10) * smooth((now - 0.3) / 0.5)
        c.set_alpha(0.55 * a); lab.set_alpha(a)
    lw_scale = np.clip(W / 20000, 0, 1)
    rim_a = smooth((W - 16000) / 4000)
    rim_lab.set_alpha(rim_a)
    rim_lab.set_position((0, RIM + 0.012 * W))

    # Tissot appear once the view is big enough, wave after the zoom ends
    ta = smooth((W - 2500) / 5500)
    scale = np.full(len(cv), 1.0)
    if now > T1:
        scale *= 1 + 0.12 * (0.5 + 0.5 * np.sin(2 * np.pi * ((now - T1) / 2.6 - dd / 20000.0)))
    cc_ = cen[:, None, :]
    tis.set_verts(list(cc_ + (ta * scale)[:, None, None] * (cv - cc_)))
    fc = tis.get_facecolor().copy()
    tis.set_alpha(None)
    tis.set_facecolor(np.c_[tcol[:, :3], np.full(len(cv), TIS_FACE * ta)])
    tis.set_edgecolor(np.c_[tcol[:, :3], np.full(len(cv), TIS_EDGE * ta)])

    if now > T1:
        ph = ((now - T1) % 3.6) / 3.6
        sweep.set_radius(ph * RIM); sweep.set_alpha(0.55 * (1 - ph) * smooth((now - T1) / 0.4))
    else:
        sweep.set_alpha(0.0)

    home_dot.set_alpha(smooth(now / 0.5)); home_lab.set_alpha(smooth(now / 0.5))
    ms = 6 + 3 * (0.5 + 0.5 * np.sin(now * 4))
    home_dot.set_markersize(ms)

    for P in PLACES:
        u = (now - P["t_on"]) / 0.9
        a = smooth(u)
        P["dot"].set_alpha(a); P["lab"].set_alpha(smooth(u - 0.4))
        P["line"].set_alpha(0.75 * a)
        P["line"].set_data([0, P["xy"][0] * smooth(u)], [0, P["xy"][1] * smooth(u)])
        P["dot"].set_markersize(7 if W > 3000 else 8)

    # readouts
    wtxt = int(round(W / 10) * 10) if W < 1000 else int(round(W / 100) * 100)
    val_l.set_text(f"{wtxt:,} km")
    r_edge = min(0.92 * W, 20000.0)
    th = r_edge / R_KM
    k = th / np.sin(th)
    val_r.set_text(f"×{k:.2f}" if k < 10 else f"×{k:.1f}")
    val_r.set_color(CMAP(NORM(np.log2(max(k, 1)))))
    for t in (lab_l, val_l, lab_r, val_r):
        t.set_alpha(smooth((now - 0.4) / 0.5))

if ONLY is not None:
    draw(ONLY); fig.savefig(f"frame_{ONLY}.png", facecolor=BG); print("frames total:", NF); sys.exit()

print("frames:", NF)
wr = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
with wr.saving(fig, OUT, dpi=100):
    for f in range(NF):
        draw(f); wr.grab_frame(facecolor=BG)
print("saved", OUT)

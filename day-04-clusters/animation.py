import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.plot
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, to_rgb
from pyproj import Transformer
from scipy import ndimage as ndi
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import Delaunay

# ---------- CONFIG ----------
TIF = "ro_ntl.tif"
AUTHOR = "Sara AHARON"
THRESHOLD = 1.5        # radiance cutoff (nW/cm2/sr) for "lit" pixels. Higher = fewer/smaller clusters
CLOSING_ITER = 2       # merges suburbs into their city cluster
N_STARS = 60           # main stars joined by constellation lines
N_DIM = 400            # extra faint background stars
LABEL_CITIES = True    # name the biggest cities when their star lights up
FPS = 30
GLOW_FRAMES = 30       # fade-in of the night-light glow
GAP = 2                # frames between one star lighting up and the next
FADE = 12              # frames each star/line takes to fade in
HOLD_SECONDS = 3       # freeze on the last frame
SIZE_PX = 1080         # square video (LinkedIn-friendly)
OUT_MP4, OUT_PNG = "day4_clusters.mp4", "day4_clusters_final.png"
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
CRS_RO = "EPSG:3844"
STAR_COLOR, LINE_COLOR = "#fff3c4", "#8fb8ff"

CITIES = {
    "București": (44.4268, 26.1025), "Cluj-Napoca": (46.7712, 23.6236),
    "Timișoara": (45.7489, 21.2087), "Iași": (47.1585, 27.6014),
    "Constanța": (44.1598, 28.6348), "Brașov": (45.6427, 25.5887),
    "Craiova": (44.3302, 23.7949), "Galați": (45.4353, 28.0080),
    "Oradea": (47.0465, 21.9189), "Ploiești": (44.9462, 26.0225),
}

# 1. Night lights -> clusters ------------------------------------------
with rasterio.open(TIF) as src:
    a = src.read(1).astype("float64")
    T = src.transform
    raster_crs = src.crs
    extent = rasterio.plot.plotting_extent(src)

a = np.clip(np.nan_to_num(a), 0, None)
total_light = a.sum()

lit = ndi.binary_closing(a > THRESHOLD, iterations=CLOSING_ITER)
lab, n = ndi.label(lit)
idx = np.arange(1, n + 1)
mass = np.asarray(ndi.sum(a, lab, idx))
cy, cx = np.array(ndi.center_of_mass(a, lab, idx)).T
xs, ys = rasterio.transform.xy(T, cy, cx)
pts = np.c_[xs, ys]

order = np.argsort(mass)[::-1]
N_STARS = min(N_STARS, n)
top, dim = order[:N_STARS], order[N_STARS:N_STARS + N_DIM]
P, M = pts[top], mass[top]                  # P is sorted brightest -> faintest
cum_share = np.cumsum(M) / total_light * 100    # % of national light

# constellation lines = minimum spanning tree over a Delaunay graph
tri = Delaunay(P)
edges = {tuple(sorted((s[i], s[j]))) for s in tri.simplices
         for i, j in [(0, 1), (1, 2), (0, 2)]}
ei, ej = np.array(list(edges)).T
w = np.linalg.norm(P[ei] - P[ej], axis=1)
mst = minimum_spanning_tree(coo_matrix((w, (ei, ej)), shape=(N_STARS,) * 2)).tocoo()
eu, ev = mst.row, mst.col

# city names for the biggest stars
labels = {}
if LABEL_CITIES:
    tr = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    for name, (la, lo) in CITIES.items():
        x, y = tr.transform(lo, la)
        d = np.hypot(P[:, 0] - x, P[:, 1] - y)
        k = int(d.argmin())
        if d[k] < 25000 and (k not in labels or d[k] < labels[k][1]):
            labels[k] = (name, d[k])

# 2. Boundaries --------------------------------------------------------
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f).to_crs(CRS_RO)
outline = counties.dissolve().boundary
x0, y0, x1, y1 = counties.total_bounds

# 3. Figure ------------------------------------------------------------
dpi = 100
fig = plt.figure(figsize=(SIZE_PX / dpi, SIZE_PX / dpi), dpi=dpi)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.03, 0.17, 0.94, 0.62])
ax.set_facecolor(BG)
counties.plot(ax=ax, facecolor=LAND, edgecolor="none", zorder=1)

glow_data = gaussian_filter(np.log1p(a), 1.2)
cols = plt.get_cmap("magma")(np.linspace(0, 1, 256))
cols[:, 3] = np.linspace(0, 1, 256) ** 0.8        # dark = transparent, so the glow fades out cleanly
glow = ax.imshow(glow_data, cmap=ListedColormap(cols), extent=extent, alpha=0, zorder=2)

# accentuated country outline (soft glow + crisp line)
for lw, al in [(8, 0.05), (4.5, 0.10)]:
    outline.plot(ax=ax, color="#8a93b0", linewidth=lw, alpha=al, zorder=5)
outline.plot(ax=ax, color="#b8c2e6", linewidth=1.6, zorder=6)

ax.set_xlim(x0 - 15000, x1 + 15000)
ax.set_ylim(y0 - 15000, y1 + 15000)
ax.set_aspect("equal")
ax.set_axis_off()

# animated artists
base = np.array(to_rgb(STAR_COLOR))
rng = np.random.default_rng(4)
phase = rng.uniform(0, 2 * np.pi, N_STARS)
dphase = rng.uniform(0, 2 * np.pi, len(dim))
s0 = (M / M.max()) ** 0.6 * 400
GLOW_LAYERS = [(6, 0.05), (3, 0.12), (1.4, 0.35), (0.5, 1.0)]
layers = [ax.scatter(P[:, 0], P[:, 1], s=0, linewidths=0, zorder=12 + i)
          for i in range(len(GLOW_LAYERS))]
dim_sc = ax.scatter(pts[dim, 0], pts[dim, 1], s=4, linewidths=0, zorder=9) if len(dim) else None
segs = [[P[u], P[v]] for u, v in zip(eu, ev)]
lc = LineCollection(segs, linewidths=0.7, zorder=8)
ax.add_collection(lc)

ignite = GLOW_FRAMES + np.arange(N_STARS) * GAP          # frame each star starts to appear
line_start = np.maximum(ignite[eu], ignite[ev])          # line appears with its second star
label_txt = {}
for k, (name, _) in labels.items():
    label_txt[k] = ax.annotate(name, P[k], xytext=(7, 7), textcoords="offset points",
                               color="white", fontsize=9, alpha=0, zorder=15)

# text (same layout as the other days)
fig.text(0.05, 0.945, "Romania, seen at night", color="white",
         fontsize=26, fontweight="bold", ha="left", va="center")
fig.text(0.05, 0.905, f"The {N_STARS} brightest light clusters, joined like constellations",
         color=STAR_COLOR, fontsize=14, ha="left", va="center")
counter = fig.text(0.95, 0.135, "", color="white", fontsize=20, fontweight="bold",
                   ha="right", va="center")
share = fig.text(0.95, 0.098, "", color="#c7cde0", fontsize=14, ha="right", va="center")
fig.text(0.05, 0.075,
         f"#30DayMapChallenge | Day 4: Clusters | {AUTHOR}\n"
         f"Clusters: VIIRS pixels above {THRESHOLD} nW/cm\u00b2/sr, merged by distance. "
         "Lines: minimum spanning tree.\n"
         "Data: VIIRS Nighttime Lights 2024 (NOAA / Earth Observation Group, via Google Earth Engine), GADM 4.1",
         color="#8a93b0", fontsize=8.5, ha="left", va="bottom", linespacing=1.5)


def draw_frame(t):
    glow.set_alpha(0.7 * min(1.0, t / GLOW_FRAMES))
    g = min(1.0, t / GLOW_FRAMES)
    if dim_sc is not None:
        tw = 0.65 + 0.35 * np.sin(0.12 * t + dphase)
        rgba = np.c_[np.tile([1, 1, 1], (len(dim), 1)), 0.35 * g * tw]
        dim_sc.set_facecolor(rgba)

    f = np.clip((t - ignite) / FADE, 0, 1)
    twinkle = 1 + 0.12 * np.sin(0.15 * t + phase) * (f > 0)
    for sc, (k, al) in zip(layers, GLOW_LAYERS):
        rgba = np.c_[np.tile(base, (N_STARS, 1)), np.clip(al * f, 0, 1)]
        sc.set_sizes(s0 * k * f * twinkle)
        sc.set_facecolor(rgba)

    lf = np.clip((t - line_start) / FADE, 0, 1)
    lc.set_color(np.c_[np.tile(to_rgb(LINE_COLOR), (len(segs), 1)), 0.5 * lf])

    for k, txt in label_txt.items():
        txt.set_alpha(float(f[k]) * 0.9)

    lit_n = int((f > 0).sum())
    counter.set_text(f"{lit_n} clusters")
    share.set_text(f"{cum_share[lit_n - 1]:.0f}% of Romania's night light" if lit_n else "")


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# 4. Render ------------------------------------------------------------
n_frames = int(ignite[-1] + FADE + 1)
frames = []
for t in range(n_frames):
    draw_frame(t)
    frames.append(grab())
    if t % 20 == 0:
        print(f"frame {t + 1}/{n_frames}")
frames += [frames[-1]] * int(HOLD_SECONDS * FPS)

fig.savefig(OUT_PNG, facecolor=BG, dpi=dpi)
print(f"Saved {OUT_PNG}")

with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as wr:
    for fr in frames:
        wr.append_data(fr)
print(f"Saved {OUT_MP4}")

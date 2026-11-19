# pip install imageio imageio-ffmpeg h3 geopandas shapely matplotlib pandas numpy
import h3, numpy as np, pandas as pd, geopandas as gpd, imageio
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from shapely.geometry import Polygon
from shapely.ops import unary_union

AUTHOR = "Sara AHARON"
BG = "#0b0d1e"
GREY = "#8a93b0"
FPS = 30
INTRO, REVEAL, GLOW, HOLD = 20, 150, 12, 70
TOTAL = INTRO + REVEAL + GLOW + HOLD

# ---------- data (from the previous run) ----------
df = pd.read_csv("day20_ro2_hex_scores.csv")
geoms = [Polygon([(lng, lat) for lat, lng in h3.cell_to_boundary(c)]) for c in df.h3]
g = gpd.GeoDataFrame(df, geometry=geoms, crs=4326).to_crs(3844)
g["area"] = [h3.cell_area(c, unit="km^2") for c in df.h3]
N = len(g)

cmap = LinearSegmentedColormap.from_list(
    "synth", ["#12163a", "#3a1c71", "#b5179e", "#ff2e93", "#ffd166", "#00f5ff"])
norm = Normalize(g.score.min(), g.score.max())
base = cmap(norm(g.score.values))                       # final colors (N,4)
dim = np.array(to_rgba("#171b3d"))
glow_col = np.array([0.85, 1.0, 1.0, 1.0])

# reveal order: highest potential first
rank = np.empty(N)
rank[np.argsort(-g.score.values)] = np.arange(N)
reveal_f = INTRO + (rank / N) ** 0.9 * REVEAL

# quintile classes (4 = highest)
g["cls"] = pd.qcut(g.score, 5, labels=False)
cls_names = ["Very low", "Low", "Medium", "High", "Very high"]
cls_color = [cmap(norm(g.score[g.cls == k].median())) for k in range(5)]
cls_area = [g.area[g.cls == k].sum() for k in range(5)]
total_area = g.area.sum()

# ---------- figure ----------
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
ax = fig.add_axes([0.04, 0.27, 0.92, 0.60], facecolor=BG)
polys = [np.asarray(p.exterior.coords) for p in g.geometry]
pc = PolyCollection(polys, facecolors=[dim] * N, edgecolors=BG, linewidths=0.4)
ax.add_collection(pc)
minx, miny, maxx, maxy = g.total_bounds
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_aspect("equal"); ax.set_axis_off()
gpd.GeoSeries([unary_union(g.geometry)], crs=3844).boundary.plot(
    ax=ax, color="#00f5ff", linewidth=0.8, alpha=0.45)

ov = fig.add_axes([0, 0, 1, 1], facecolor="none")       # overlay in figure coords
ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")

fig.text(0.06, 0.93, "Territorial Potential", color="white", size=26, weight="bold")
fig.text(0.06, 0.895, "Where people, access, services and land align across Macroregion 2",
         color="#ff2e93", size=14)
counter = fig.text(0.94, 0.93, "", color="#00f5ff", size=12, ha="right")

# --- legend left: potential classes ---
ov.text(0.06, 0.245, "DEVELOPMENT POTENTIAL INDEX (relative, within RO2)",
        color=GREY, size=9, weight="bold")
swatches = []
for i, k in enumerate([4, 3, 2, 1, 0]):
    y = 0.215 - i * 0.027
    sw = Rectangle((0.06, y - 0.004), 0.022, 0.014, color=cls_color[k], alpha=0.2)
    ov.add_patch(sw); swatches.append((k, sw))
    ov.text(0.095, y, cls_names[k], color="white", size=11, va="center")
    ov.text(0.46, y, f"{cls_area[k] / total_area:.0%} · {cls_area[k]:,.0f} km²",
            color=GREY, size=10, va="center", ha="right")

# --- legend right: what the score measures ---
ov.text(0.54, 0.245, "WHAT THE SCORE COMBINES", color=GREY, size=9, weight="bold")
comps = [("Population density", 0.30, "people per km², WorldPop 2020"),
         ("Accessibility", 0.30, "travel time to nearest 50k+ city"),
         ("Services", 0.25, "night lights as proxy, VIIRS 2023"),
         ("Land resources", 0.15, "land-cover suitability, ESA WorldCover")]
for i, (name, w, desc) in enumerate(comps):
    y = 0.215 - i * 0.027
    ov.text(0.54, y + 0.004, f"{name} · {w:.0%}", color="white", size=10.5, va="center")
    ov.text(0.54, y - 0.009, desc, color=GREY, size=8, va="center")
    ov.add_patch(Rectangle((0.80, y), w / 0.30 * 0.14, 0.008, color="#ff2e93", alpha=0.85))

fig.text(0.06, 0.04, f"#30DayMapChallenge | Day 20: Hexagons | {AUTHOR}", color=GREY, size=8)
fig.text(0.06, 0.025, "Data: WorldPop 2020, Oxford MAP accessibility, ESA WorldCover 2021, "
         "VIIRS 2023, FAO GAUL | H3 res 6 hexagons | Index = weighted sum of percentile ranks",
         color=GREY, size=8)

# ---------- render ----------
def update(f):
    age = f - reveal_f
    rev = age >= 0
    glow = np.clip(1 - age / GLOW, 0, 1)[:, None] * rev[:, None] * 0.85
    cols = np.where(rev[:, None], base * (1 - glow) + glow_col * glow, dim)
    pc.set_facecolor(cols)
    counter.set_text(f"{rev.sum():,} / {N:,} hexagons")
    for k, sw in swatches:
        frac = rev[(g.cls == k).values].mean()
        sw.set_alpha(0.2 + 0.8 * frac)

writer = imageio.get_writer("day20_hexagons_ro2.mp4", fps=FPS, codec="libx264",
                            quality=8, macro_block_size=1)
for f in range(TOTAL):
    update(f)
    fig.canvas.draw()
    writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    if f % 20 == 0:
        print(f"frame {f}/{TOTAL}")
writer.close()

update(TOTAL - 1)
fig.savefig("day20_hexagons_ro2_final.png", facecolor=BG)
print("Done: day20_hexagons_ro2.mp4 + day20_hexagons_ro2_final.png")

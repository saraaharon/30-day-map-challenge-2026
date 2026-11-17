"""
30 Day Map Challenge - Day 1: Points
Emergency hospitals of Romania
"""
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

# ---------- CONFIG ----------
CSV_PATH = "hospitals_emergency_v2.csv"
OUT_PNG = "day1_points_hospitals.png"
AUTHOR = "Sara AHARON"
SOURCE = "Data: OpenStreetMap contributors | Boundaries: GADM 4.1"
INSET_X, INSET_Y, INSET_WIDTH = -0.08, 0.02, 0.26  # Bucharest inset position/size (fractions of the map area)
INCLUDE_INFERRED = False  # True = include ALL name-only points (not recommended)
INCLUDE_NAME_URGEN = True  # include hospitals with "Urgență" in the name (e.g. Focșani)
NAME_EXCLUDE = ["secți", "oncolog", "pneumolog", "psihiatr", "chirurgie plastic", "tbc"]
EXCLUDE_IDS = {"w1481391227"}  # duplicates / wrong points, add ids here to drop them
DEDUPE_METERS = 150   # name-only points closer than this to another point are dropped
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
RED, AMBER = "#ff3b4e", "#ffb347"
CRS_RO = "EPSG:3844"  # Stereo 70

# 1. Hospitals
df = pd.read_csv(CSV_PATH).dropna(subset=["lat", "lon"])
if not INCLUDE_INFERRED:
    tagged = df["emergency"].eq("yes")
    by_name = (
        df["basis"].eq("name_urgen")
        & ~df["name"].str.lower().str.contains("|".join(NAME_EXCLUDE), regex=True)
    ) if INCLUDE_NAME_URGEN else False
    df = df[tagged | by_name]
df = df[~df["id"].isin(EXCLUDE_IDS)]
# confirmed = tagged emergency=yes in OSM; inferred = flagged by name only
df["status"] = df["emergency"].eq("yes").map({True: "confirmed", False: "inferred"})

pts = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
).to_crs(CRS_RO)

# Drop name-only points that sit on top of another point (same site mapped
# twice, or a hospital section). Tagged-emergency points are always kept.
pts = pts.sort_values("status", key=lambda s: s.ne("confirmed")).reset_index(drop=True)
keep = []
for i, row in pts.iterrows():
    if row["status"] == "confirmed" or all(
        row.geometry.distance(pts.geometry[j]) > DEDUPE_METERS for j in keep
    ):
        keep.append(i)
removed = len(pts) - len(keep)
pts = pts.loc[keep]
print(f"{len(pts)} points kept ({removed} name-only near-duplicates removed)")

# 2. County boundaries (GADM level 1), downloaded once
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip"
    urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f)
counties = counties.to_crs(CRS_RO)

# 3. Plot
fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

counties.plot(ax=ax, facecolor=LAND, edgecolor=LINE, linewidth=0.6)
counties.dissolve().boundary.plot(ax=ax, color="#4a5578", linewidth=1.0)

for status, color in [("inferred", AMBER), ("confirmed", RED)]:
    sub = pts[pts["status"] == status]
    if sub.empty:
        continue
    for size, alpha in [(260, 0.04), (120, 0.08), (50, 0.15)]:   # glow
        sub.plot(ax=ax, color=color, markersize=size, alpha=alpha, linewidth=0)
    sub.plot(ax=ax, color=color, markersize=14, alpha=0.95,
             edgecolor="white", linewidth=0.3)

ax.set_axis_off()

# 3b. Bucharest inset (zoom on the dense cluster)
buch = pts[pts["lon"].between(25.95, 26.25) & pts["lat"].between(44.35, 44.52)]
if not buch.empty:
    x0, y0, x1, y1 = buch.total_bounds
    pad = 2500  # meters
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    w, h = x1 - x0, y1 - y0
    size = INSET_WIDTH                # inset width as a fraction of the main axes
    axins = ax.inset_axes([INSET_X, INSET_Y, size, size * h / w * 0.75])
    axins.set_facecolor(BG)
    counties.plot(ax=axins, facecolor=LAND, edgecolor=LINE, linewidth=0.6)
    for status, color in [("inferred", AMBER), ("confirmed", RED)]:
        sub = buch[buch["status"] == status]
        if sub.empty:
            continue
        for gsize, alpha in [(520, 0.05), (240, 0.09), (100, 0.16)]:
            sub.plot(ax=axins, color=color, markersize=gsize, alpha=alpha, linewidth=0)
        sub.plot(ax=axins, color=color, markersize=36, alpha=0.95,
                 edgecolor="white", linewidth=0.4)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_aspect("equal", adjustable="datalim")
    axins.set_xlabel("")
    axins.set_ylabel("")
    axins.set_xticks([])
    axins.set_yticks([])
    for sp in axins.spines.values():
        sp.set_edgecolor("#8a93b0")
        sp.set_linewidth(0.8)
    axins.set_title(f"București ({len(buch)})", color="white", fontsize=11,
                    loc="left", pad=4)
    ax.indicate_inset_zoom(axins, edgecolor="#8a93b0", linewidth=0.8, alpha=0.9)

# Legend
n_c = (pts["status"] == "confirmed").sum()
n_i = (pts["status"] == "inferred").sum()
handles = [
    Line2D([], [], marker="o", ls="", color=RED, markeredgecolor="white",
           markeredgewidth=0.3, markersize=8, label=f"Emergency hospital ({n_c})"),
]
if n_i:
    handles.append(
        Line2D([], [], marker="o", ls="", color=AMBER, markeredgecolor="white",
               markeredgewidth=0.3, markersize=8, label=f"Emergency hospital, identified by name ({n_i})"))
LEGEND_Y = -0.06  # lower = more negative
leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, LEGEND_Y),
                frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color("white")

# 4. Text
fig.text(0.06, 0.93, "Emergency Hospitals of Romania",
         color="white", fontsize=26, fontweight="bold", ha="left")
fig.text(0.06, 0.895, f"{len(pts)} emergency hospitals from OpenStreetMap",
         color=RED, fontsize=14, ha="left")
fig.text(0.06, 0.04, f"#30DayMapChallenge | Day 1: Points | {AUTHOR}\n{SOURCE}",
         color="#8a93b0", fontsize=8, ha="left", va="bottom")

plt.savefig(OUT_PNG, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {OUT_PNG}")

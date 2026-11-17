"""
30 Day Map Challenge - Day 4: Clusters
Towns and villages of Romania grouped into clusters (DBSCAN on OpenStreetMap places).

Install once:  pip install scikit-learn
"""
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from sklearn.cluster import DBSCAN

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
OUT_PNG = "day4_clusters.png"
EPS_KM = 5            # settlements closer than this are linked together
MIN_SAMPLES = 8       # min settlements to count as a cluster
LABEL_TOP = 0         # label the N biggest clusters (0 = no labels)
CACHE = Path("settlements_osm.json")
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
CRS_RO = "EPSG:3844"
CMAP = LinearSegmentedColormap.from_list("cl", ["#00e5ff", "#8a4dff", "#ff2e97", "#ffb347"])

QUERY = """
[out:json][timeout:240];
area["ISO3166-1"="RO"][admin_level=2]->.ro;
nwr["place"~"^(city|town|village)$"](area.ro);
out center tags;
"""
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]


def fetch():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    body = urllib.parse.urlencode({"data": QUERY}).encode()
    last = None
    for attempt in range(3):
        for url in ENDPOINTS:
            try:
                print(f"Downloading places from {url} (can take a few minutes)...")
                req = urllib.request.Request(
                    url, data=body, headers={"User-Agent": "30DayMapChallenge-sara/1.0"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = json.loads(r.read().decode("utf-8"))
                CACHE.write_text(json.dumps(data), encoding="utf-8")
                return data
            except Exception as e:
                last = e
                print(f"  failed: {e}")
        print("  all servers busy, waiting 20 s...")
        import time
        time.sleep(20)
    raise SystemExit(f"Overpass is not answering right now ({last}). Try again in a few minutes.")


# 1. Places
data = fetch()
rows = []
for el in data["elements"]:
    c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
    t = el.get("tags", {})
    if c["lat"] is None or "place" not in t:
        continue
    rows.append({"name": t.get("name", ""), "place": t["place"],
                 "geometry": gpd.points_from_xy([c["lon"]], [c["lat"]])[0]})
places = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS_RO)
places = places.drop_duplicates(subset=["name", "place"]).reset_index(drop=True)
print(f"{len(places)} settlements")

# 2. DBSCAN clustering (distances in meters, Stereo 70)
xy = np.column_stack([places.geometry.x, places.geometry.y])
labels = DBSCAN(eps=EPS_KM * 1000, min_samples=MIN_SAMPLES).fit_predict(xy)
places["cluster"] = labels
sizes = places[places.cluster >= 0].groupby("cluster").size().sort_values(ascending=False)
in_cl = (labels >= 0).sum()
print(f"{len(sizes)} clusters; {in_cl} of {len(places)} settlements "
      f"({100 * in_cl / len(places):.0f}%) are in a cluster; biggest has {sizes.iloc[0]}")

# 3. Boundaries
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f).to_crs(CRS_RO)

# 4. Plot
fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
counties.plot(ax=ax, facecolor=LAND, edgecolor=LINE, linewidth=0.5)
counties.dissolve().boundary.plot(ax=ax, color="#4a5578", linewidth=1.0)

iso = places[places.cluster < 0]
cl = places[places.cluster >= 0].copy()
cl["csize"] = cl.cluster.map(sizes)
norm = LogNorm(vmin=MIN_SAMPLES, vmax=sizes.iloc[0])

iso.plot(ax=ax, color="#5b6585", markersize=1.2, alpha=0.7, linewidth=0)
colors = [CMAP(norm(s)) for s in cl.csize]
for size, alpha in [(40, 0.05), (16, 0.10)]:         # glow
    cl.plot(ax=ax, color=colors, markersize=size, alpha=alpha, linewidth=0)
cl.plot(ax=ax, color=colors, markersize=2.5, alpha=0.95, linewidth=0)

# labels for the biggest clusters: name of the largest place inside
rank = {"city": 3, "town": 2, "village": 1}
labeled = []
for cid in sizes.index[:LABEL_TOP * 3]:
    if len(labeled) >= LABEL_TOP:
        break
    g = cl[cl.cluster == cid].copy()
    g["r"] = g.place.map(rank)
    best = g.sort_values("r", ascending=False).iloc[0]
    if not best["name"]:
        continue
    cx, cy = g.geometry.x.mean(), g.geometry.y.mean()
    if any((cx - lx) ** 2 + (cy - ly) ** 2 < 40000 ** 2 for lx, ly in labeled):
        continue            # too close to a bigger cluster's label
    labeled.append((cx, cy))
    ax.annotate(best["name"], (cx, cy), xytext=(0, 14), textcoords="offset points",
                color="white", fontsize=9, fontweight="bold", ha="center",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])
ax.set_axis_off()

# colorbar + legend
cax = fig.add_axes([0.70, 0.125, 0.22, 0.012])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
cb.set_label("Settlements in the cluster", color="#c7cde0", fontsize=9)
cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_tick_params(color="#c7cde0", labelcolor="#c7cde0", labelsize=8)
cb.outline.set_visible(False)
fig.text(0.70, 0.085, f"Grey dots: isolated settlements ({len(iso)})",
         color="#8a93b0", fontsize=9, ha="left")

# text
fig.text(0.06, 0.93, "Where Romania Clusters", color="white",
         fontsize=26, fontweight="bold", ha="left")
fig.text(0.06, 0.895,
         f"{len(places):,} towns and villages, {len(sizes)} clusters "
         f"(settlements within {EPS_KM} km of each other)",
         color="#00e5ff", fontsize=13, ha="left")
fig.text(0.06, 0.04,
         f"#30DayMapChallenge | Day 4: Clusters | {AUTHOR}\n"
         "Data: OpenStreetMap contributors | Boundaries: GADM 4.1 | Method: DBSCAN",
         color="#8a93b0", fontsize=8, ha="left", va="bottom")

plt.savefig(OUT_PNG, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {OUT_PNG}")


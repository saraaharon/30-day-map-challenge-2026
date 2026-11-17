"""
30 Day Map Challenge - Day 2: Lines
Romania's motorways: built, under construction, planned (from OpenStreetMap)
"""
import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely.geometry import LineString

# ---------- CONFIG ----------
OUT_PNG = "day2_motorways.png"
AUTHOR = "Sara AHARON"
SOURCE = ("Data: OpenStreetMap contributors | Boundaries: GADM 4.1\n"
          "Planned sections are as mapped in OSM and may differ from official CNAIR plans.")
CACHE = Path("motorways_osm.json")
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
BUILT, BUILDING, PLANNED = "#00e5ff", "#ffb347", "#ff2e97"
CRS_RO = "EPSG:3844"  # Stereo 70

QUERY = """
[out:json][timeout:180];
area["ISO3166-1"="RO"][admin_level=2]->.ro;
(
  way["highway"="motorway"](area.ro);
  way["highway"="construction"]["construction"="motorway"](area.ro);
  way["highway"="proposed"]["proposed"="motorway"](area.ro);
);
out geom;
"""
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


def fetch_osm():
    """Download motorways from Overpass once, then reuse the cached file."""
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    body = urllib.parse.urlencode({"data": QUERY}).encode()
    last_err = None
    for url in ENDPOINTS:
        try:
            print(f"Downloading from {url} (can take a minute)...")
            req = urllib.request.Request(
                url, data=body, headers={"User-Agent": "30DayMapChallenge-sara/1.0"})
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.loads(r.read().decode("utf-8"))
            CACHE.write_text(json.dumps(data), encoding="utf-8")
            return data
        except Exception as e:  # try next endpoint
            last_err = e
            print(f"  failed: {e}")
    raise SystemExit(f"Could not download OSM data: {last_err}")


# 1. Motorways
data = fetch_osm()
rows = []
for el in data["elements"]:
    geom = el.get("geometry")
    if el.get("type") != "way" or not geom or len(geom) < 2:
        continue
    hw = el.get("tags", {}).get("highway")
    status = {"motorway": "built", "construction": "building", "proposed": "planned"}.get(hw)
    if status:
        rows.append({"status": status,
                     "geometry": LineString([(p["lon"], p["lat"]) for p in geom])})
roads = gpd.GeoDataFrame(rows, crs="EPSG:4326").to_crs(CRS_RO)
print(roads["status"].value_counts().to_string())

# 2. County boundaries (GADM level 1), downloaded once
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f)
counties = counties.to_crs(CRS_RO)

# 3. Plot
fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
counties.plot(ax=ax, facecolor=LAND, edgecolor=LINE, linewidth=0.5)
counties.dissolve().boundary.plot(ax=ax, color="#4a5578", linewidth=1.0)

styles = {   # status: (color, linewidth, linestyle)
    "planned": (PLANNED, 1.0, (0, (1, 2))),
    "building": (BUILDING, 1.5, (0, (4, 2))),
    "built": (BUILT, 1.6, "-"),
}
for status, (color, lw, ls) in styles.items():
    sub = roads[roads["status"] == status]
    if sub.empty:
        continue
    if status == "built":  # glow
        for glow_lw, alpha in [(7, 0.05), (4, 0.10)]:
            sub.plot(ax=ax, color=color, linewidth=glow_lw, alpha=alpha)
    sub.plot(ax=ax, color=color, linewidth=lw, linestyle=ls, alpha=0.95)

ax.set_axis_off()

# Legend
handles = [
    Line2D([], [], color=BUILT, lw=2, label="Motorway, open"),
    Line2D([], [], color=BUILDING, lw=2, ls=(0, (4, 2)), label="Under construction"),
    Line2D([], [], color=PLANNED, lw=2, ls=(0, (1, 2)), label="Planned / proposed"),
]
LEGEND_Y = -0.06
leg = ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, LEGEND_Y),
                frameon=False, fontsize=10)
for t in leg.get_texts():
    t.set_color("white")

# 4. Text
fig.text(0.06, 0.93, "Romania's Motorways", color="white",
         fontsize=26, fontweight="bold", ha="left")
fig.text(0.06, 0.895, "Open, under construction, and planned",
         color=BUILT, fontsize=14, ha="left")
fig.text(0.06, 0.04, f"#30DayMapChallenge | Day 2: Lines | {AUTHOR}\n{SOURCE}",
         color="#8a93b0", fontsize=8, ha="left", va="bottom")

plt.savefig(OUT_PNG, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {OUT_PNG}")


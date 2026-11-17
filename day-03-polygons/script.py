
"""
30 Day Map Challenge - Day 3: Polygons
STEP 1 of 2: compute driving time from a grid of points to every Romanian airport.

How it works
  1. Airports are downloaded from OpenStreetMap (aeroway=aerodrome with an IATA code)
     and saved to airports.csv. EDIT THAT FILE to remove/add airports, then re-run.
  2. A regular grid of points is laid over Romania.
  3. OpenRouteService (Matrix API) gives the driving time grid point -> each airport.
  4. Results are saved to grid_times.csv (resumable: progress is cached in matrix_cache.json).

Setup: create a free key at openrouteservice.org, then in the terminal:
    Windows (cmd):   set ORS_KEY=your_key_here
    then:            python day3_compute.py
"""
import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# ---------- CONFIG ----------
ORS_KEY = os.environ.get("ORS_KEY", "")
GRID_KM = 6              # grid spacing; smaller = finer but more requests (6 km is ~33 requests)
AIRPORTS_CSV = Path("airports.csv")
CACHE = Path("matrix_cache.json")
OUT_CSV = Path("grid_times.csv")
# ----------------------------

CRS_RO = "EPSG:3844"
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
ORS_MATRIX = "https://api.openrouteservice.org/v2/matrix/driving-car"

AIRPORT_QUERY = """
[out:json][timeout:120];
area["ISO3166-1"="RO"][admin_level=2]->.ro;
nwr["aeroway"="aerodrome"]["iata"](area.ro);
out center tags;
"""


def get_airports():
    if AIRPORTS_CSV.exists():
        with open(AIRPORTS_CSV, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    print("Downloading airports from OpenStreetMap...")
    body = urllib.parse.urlencode({"data": AIRPORT_QUERY}).encode()
    data, last_err = None, None
    for attempt in range(3):                      # 3 rounds over all servers
        for url in OVERPASS_ENDPOINTS:
            try:
                print(f"  trying {url} ...")
                req = urllib.request.Request(
                    url, data=body, headers={"User-Agent": "30DayMapChallenge-sara/1.0"})
                with urllib.request.urlopen(req, timeout=180) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except Exception as e:
                last_err = e
                print(f"    failed: {e}")
        if data is not None:
            break
        print("  all servers busy, waiting 20 s...")
        time.sleep(20)
    if data is None:
        raise SystemExit(f"Overpass is not answering right now ({last_err}). Try again in a few minutes.")
    rows, seen = [], set()
    for el in data["elements"]:
        t = el.get("tags", {})
        c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        iata = t.get("iata", "").strip()
        if not iata or iata in seen or c["lat"] is None:
            continue
        seen.add(iata)
        rows.append({"iata": iata, "name": t.get("name:en") or t.get("name", iata),
                     "lon": c["lon"], "lat": c["lat"]})
    rows.sort(key=lambda r: r["iata"])
    with open(AIRPORTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["iata", "name", "lon", "lat"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved {len(rows)} airports to {AIRPORTS_CSV} - open it, delete any you don't want, "
          "then run this script again.")
    return None  # stop so you can review the list


def make_grid():
    zip_path = Path("gadm41_ROU_1.json.zip")
    if not zip_path.exists():
        urllib.request.urlretrieve(
            "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".json")][0]
        with z.open(name) as f:
            country = gpd.read_file(f).to_crs(CRS_RO).dissolve().geometry.iloc[0]
    x0, y0, x1, y1 = country.bounds
    step = GRID_KM * 1000
    pts = [Point(x, y) for x in np.arange(x0, x1, step) for y in np.arange(y0, y1, step)]
    pts = [p for p in pts if country.contains(p)]
    g = gpd.GeoSeries(pts, crs=CRS_RO).to_crs("EPSG:4326")
    return [(p.x, p.y) for p in g]


def ors_matrix(airports, dests):
    """Seconds from each airport to each destination (None = unreachable)."""
    n = len(airports)
    locs = [[float(a["lon"]), float(a["lat"])] for a in airports] + [list(d) for d in dests]
    payload = {"locations": locs, "sources": list(range(n)),
               "destinations": list(range(n, n + len(dests))), "metrics": ["duration"]}
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                ORS_MATRIX, data=json.dumps(payload).encode(),
                headers={"Authorization": ORS_KEY, "Content-Type": "application/json",
                         "User-Agent": "30DayMapChallenge-sara/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())["durations"]
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print("  rate limit, waiting 60 s...")
                time.sleep(60)
            else:
                raise SystemExit(f"ORS error {e.code}: {e.read().decode()[:300]}")
    raise SystemExit("Too many retries.")


def main():
    airports = get_airports()
    if airports is None:
        return
    if not ORS_KEY:
        raise SystemExit("No ORS_KEY found. In the terminal run:  set ORS_KEY=your_key_here")
    print(f"{len(airports)} airports:", ", ".join(a["iata"] for a in airports))

    grid = make_grid()
    per_req = max(1, 3500 // len(airports) - 1)   # ORS limit: 3500 origin x destination pairs
    batches = [grid[i:i + per_req] for i in range(0, len(grid), per_req)]
    print(f"{len(grid)} grid points, {len(batches)} requests")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for i, batch in enumerate(batches):
        if str(i) in cache:
            continue
        cache[str(i)] = ors_matrix(airports, batch)
        CACHE.write_text(json.dumps(cache))
        print(f"  request {i + 1}/{len(batches)} done")
        time.sleep(1.6)   # stay under 40 requests/minute

    iatas = [a["iata"] for a in airports]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon", "lat"] + [f"t_{c}" for c in iatas])
        for i, batch in enumerate(batches):
            dur = cache[str(i)]   # [airport][dest]
            for j, (lon, lat) in enumerate(batch):
                row = [round(lon, 5), round(lat, 5)]
                for k in range(len(iatas)):
                    v = dur[k][j]
                    row.append("" if v is None else round(v / 60, 1))
                w.writerow(row)
    print(f"Saved {OUT_CSV}. Next: send it to Claude for the animation step.")


if __name__ == "__main__":
    main()

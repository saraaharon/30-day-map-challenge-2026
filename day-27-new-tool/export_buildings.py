"""
Downloads building footprints around Focsani from Overture Maps and saves buildings.json
(local km coordinates, Focsani = 0,0) for render_blender.py.

  pip install overturemaps geopandas shapely
  python export_buildings.py
"""
import json, math, random
import numpy as np
from overturemaps import geodataframe

LON0, LAT0 = 27.1786, 45.6967
RADIUS_KM = 12.0                         # city radius to include
KM_LON = 111.32 * math.cos(math.radians(LAT0))
KM_LAT = 111.13

dlat = (RADIUS_KM + 0.3) / KM_LAT
dlon = (RADIUS_KM + 0.3) / KM_LON
bbox = (LON0 - dlon, LAT0 - dlat, LON0 + dlon, LAT0 + dlat)

print("Downloading buildings from Overture (can take a minute)...")
gdf = geodataframe("building", bbox=bbox)
if gdf.crs is not None:
    gdf = gdf.to_crs(4326)
print(len(gdf), "features downloaded")

rng = random.Random(27)


def ring_km(coords):
    pts = []
    for lon, lat in coords:
        p = (round((lon - LON0) * KM_LON, 4), round((lat - LAT0) * KM_LAT, 4))
        if not pts or p != pts[-1]:
            pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    return pts


def area_m2(pts):
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2 * 1e6


def estimate_height(area, dist_km):
    c = max(0.0, 1 - dist_km / 2.5)  # blocks are taller near the centre
    r = rng.random()
    if area < 60:
        return 3 + r
    if area < 220:
        return 4.5 + 2.5 * r
    if area < 2500:
        return 8 + (6 + 22 * c) * r
    return 7 + 5 * r  # big low sheds / industrial


def clean(v):
    try:
        return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
    except (TypeError, ValueError):
        return None


out, real = [], 0
for _, row in gdf.iterrows():
    geom = row.geometry
    if geom is None:
        continue
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.geom_type != "Polygon":
            continue
        pts = ring_km(poly.exterior.coords)
        if len(pts) < 3:
            continue
        a = area_m2(pts)
        if a < 15:
            continue
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        d = math.hypot(cx, cy)
        if d > RADIUS_KM:
            continue
        hm = clean(row.get("height"))
        nf = clean(row.get("num_floors"))
        if hm and hm > 0:
            real += 1
        elif nf and nf > 0:
            hm = nf * 3.2
            real += 1
        else:
            hm = estimate_height(a, d)
        flat = [v for p in pts for v in p]
        out.append([round(hm, 1), flat])

json.dump({"b": out}, open("buildings.json", "w"), separators=(",", ":"))
print(f"Saved buildings.json: {len(out)} buildings ({real} with real height/floors, rest estimated)")

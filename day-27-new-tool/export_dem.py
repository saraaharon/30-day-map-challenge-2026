import ee, requests, zipfile, io, json
import numpy as np
import rasterio
from rasterio.warp import transform

ee.Initialize(project="advance-vector-464019-f8")

# Box: Focșani (east, plain) -> Vrancea Carpathians (west)
W, S, E, N = 26.0, 45.35, 27.5, 46.1
region = ee.Geometry.Rectangle([W, S, E, N])

dem = ee.Image("USGS/SRTMGL1_003").clip(region)

url = dem.getDownloadURL({
    "region": region,
    "scale": 90,
    "crs": "EPSG:32635",   # UTM 35N, so units are meters
    "format": "GEO_TIFF",
})

r = requests.get(url, timeout=300)
r.raise_for_status()
content = r.content
if content[:2] == b"PK":  # sometimes it comes zipped
    z = zipfile.ZipFile(io.BytesIO(content))
    content = z.read(z.namelist()[0])
open("dem.tif", "wb").write(content)

with rasterio.open("dem.tif") as ds:
    h = ds.read(1).astype("float32")
    h = np.nan_to_num(h, nan=0.0)
    xs, ys = transform("EPSG:4326", ds.crs, [27.1786], [45.6967])  # Focșani
    row, col = ds.index(xs[0], ys[0])
    meta = {
        "rows": int(h.shape[0]), "cols": int(h.shape[1]),
        "pixel_m": 90,
        "focsani_row": int(row), "focsani_col": int(col),
        "min_m": float(h.min()), "max_m": float(h.max()),
    }
np.save("dem.npy", h)
json.dump(meta, open("dem_meta.json", "w"), indent=2)
print(meta)

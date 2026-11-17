"""
30 Day Map Challenge - Day 5: Sight (VIDEO, 360-degree sweep)
Top: the viewshed map is revealed direction by direction, like a lighthouse beam.
Bottom: the panorama you would actually see from the summit, looking that way.

Run it in the SAME folder as day5_viewshed.py (re-uses dem_cache/ and the GADM download).
Install once:  pip install rasterio scipy geopandas matplotlib shapely imageio imageio-ffmpeg
Tip: set PREVIEW_ONLY = True first -> renders ONE frame (day5_preview.png) in seconds.
"""
import math
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb
from matplotlib.patches import Circle, Wedge
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform, transform_bounds
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt, map_coordinates

# ---------- CONFIG ----------
AUTHOR = "Sara AHARON"
PEAK = "Moldoveanu"          # Moldoveanu, Omu, Peleaga, Ceahlău (or add your own to PEAKS)
MAX_KM = 150                 # how far to look (radius)
RES_M = 120                  # analysis grid (m)
OBSERVER_H = 2.0             # eye height above the summit (m)
REFRACTION_K = 0.13
SECONDS = 18                 # length of the 360-degree sweep
FPS = 30
HOLD_START, HOLD_END = 0.5, 3.0   # seconds frozen at the start / end
FOV_DEG = 90                 # width of the panorama window (degrees)
PANO_V_DEG = 9               # vertical span of the panorama (degrees, exaggerated on purpose)
SIZE_W, SIZE_H = 1080, 1350  # 4:5 portrait = best for LinkedIn / Instagram
PREVIEW_ONLY = False         # True = one frame only -> day5_preview.png
OUT_MP4 = f"day5_sight_{PEAK.lower()}.mp4"
OUT_PNG = f"day5_sight_{PEAK.lower()}_final.png"
CACHE_DIR = Path("dem_cache")
PEAKS = {
    "Moldoveanu": (45.5997, 24.7358, 2544),
    "Omu": (45.4458, 25.4569, 2505),
    "Peleaga": (45.3631, 22.9006, 2509),
    "Ceahlău": (46.9792, 25.9503, 1904),
}
# ----------------------------

BG, LAND, LINE = "#0b0f1a", "#151b2b", "#2a3350"
CRS_RO = "EPSG:3844"
CMAP = LinearSegmentedColormap.from_list(
    "reach", ["#00e5ff", "#3d8bff", "#8a4dff", "#ff2e97", "#ffb347"])
EARTH_R = 6371000.0
NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


# ---------- helpers (same as day5_viewshed.py) ----------
def tile_name(lat0, lon0):
    ns = "N" if lat0 >= 0 else "S"
    ew = "E" if lon0 >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat0):02d}_00_{ew}{abs(lon0):03d}_00_DEM"


def get_tile(lat0, lon0):
    name = tile_name(lat0, lon0)
    path = CACHE_DIR / f"{name}.tif"
    if path.exists():
        return path
    CACHE_DIR.mkdir(exist_ok=True)
    url = f"https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    print(f"  downloading {name} ...")
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise
    tmp.rename(path)
    return path


def find_summit(lat, lon, radius_m=300):
    path = get_tile(math.floor(lat), math.floor(lon))
    if path is None:
        raise SystemExit("No DEM tile for this peak.")
    dlat = radius_m / 111000
    dlon = radius_m / (111000 * math.cos(math.radians(lat)))
    with rasterio.open(path) as src:
        r0, c0 = src.index(lon - dlon, lat + dlat)
        r1, c1 = src.index(lon + dlon, lat - dlat)
        z = src.read(1, window=Window.from_slices((r0, r1 + 1), (c0, c1 + 1)))
        i, j = np.unravel_index(np.nanargmax(z), z.shape)
        px, py = src.xy(r0 + i, c0 + j)
        elev = float(z[i, j])
    xs, ys = transform("EPSG:4326", CRS_RO, [px], [py])
    return xs[0], ys[0], elev


def build_dem(tf, n):
    left, top = tf.c, tf.f
    w, s, e, nn = transform_bounds(CRS_RO, "EPSG:4326", left, top - n * RES_M,
                                   left + n * RES_M, top, densify_pts=21)
    dem = np.full((n, n), np.nan, dtype="float32")
    lats = range(math.floor(s), math.floor(nn) + 1)
    lons = range(math.floor(w), math.floor(e) + 1)
    print(f"DEM tiles needed: {len(lats) * len(lons)}")
    for la in lats:
        for lo in lons:
            path = get_tile(la, lo)
            if path is None:
                continue
            tmp = np.full((n, n), np.nan, dtype="float32")
            with rasterio.open(path) as src:
                reproject(source=rasterio.band(src, 1), destination=tmp,
                          dst_transform=tf, dst_crs=CRS_RO, dst_nodata=np.nan,
                          resampling=Resampling.average)
            dem = np.where(np.isnan(dem), tmp, dem)
    bad = np.isnan(dem)
    if bad.all():
        raise SystemExit("DEM is empty - check the downloads.")
    if bad.any():
        dist, (ii, jj) = distance_transform_edt(bad, return_indices=True)
        fill = bad & (dist <= 3)
        dem[fill] = dem[ii[fill], jj[fill]]
        dem[np.isnan(dem)] = 0
    return dem.astype("float64")


def hillshade(z, res, az=315, alt=40, exag=1.6):
    dz_row, dz_col = np.gradient(z * exag, res)
    nx, ny, nz = -dz_col, dz_row, np.ones_like(z)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    a, h = math.radians(az), math.radians(alt)
    lx, ly, lz = math.sin(a) * math.cos(h), math.cos(a) * math.cos(h), math.sin(h)
    return np.clip((nx * lx + ny * ly + nz * lz) / norm, 0, 1)


def cast_rays(dem, hs, res, orow, ocol, z_obs, max_r, k=REFRACTION_K, chunk=400):
    """
    Ray casting. Besides visibility, keeps for every ray the running-maximum elevation angle
    (= the skyline profile at each distance) and the hillshade under every sample (for the panorama).
    """
    r_eff = EARTH_R / (1 - k)
    n_steps = int(max_r / res)
    r = res * np.arange(1, n_steps + 1)
    n_rays = int(math.ceil(2 * math.pi * max_r / res))
    az = np.arange(n_rays) * 2 * math.pi / n_rays          # clockwise from north
    drop = r ** 2 / (2 * r_eff)
    vis = np.empty((n_rays, n_steps), dtype=bool)
    cm_deg = np.empty((n_rays, n_steps), dtype="float32")
    hs_s = np.empty((n_rays, n_steps), dtype="float32")
    for a in range(0, n_rays, chunk):
        t = az[a:a + chunk, None]
        rows = orow - (r[None, :] / res) * np.cos(t)
        cols = ocol + (r[None, :] / res) * np.sin(t)
        z = map_coordinates(dem, [rows, cols], order=1, mode="nearest")
        h = map_coordinates(hs, [rows, cols], order=1, mode="nearest")
        ang = (z - drop - z_obs) / r
        cm = np.maximum.accumulate(ang, axis=1)
        prev = np.empty_like(cm)
        prev[:, 0] = -np.inf
        prev[:, 1:] = cm[:, :-1]
        vis[a:a + chunk] = ang >= prev
        cm_deg[a:a + chunk] = np.degrees(np.arctan(cm))
        hs_s[a:a + chunk] = h
    return vis, cm_deg, hs_s, r, n_rays, n_steps


def polar_to_grid(vis, n_rays, n_steps, shape, res, orow, ocol):
    I, J = np.indices(shape)
    dx = (J - ocol) * res
    dy = (orow - I) * res
    dist = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dx, dy), 2 * math.pi)
    ray = np.rint(theta / (2 * math.pi) * n_rays).astype(int) % n_rays
    step = np.rint(dist / res).astype(int) - 1
    ok = (step >= 0) & (step < n_steps)
    out = np.zeros(shape, dtype=bool)
    out[ok] = vis[ray[ok], step[ok]]
    out[dist < 0.5 * res] = True
    return out, dist, ray


# 1. Boundaries --------------------------------------------------------
zip_path = Path("gadm41_ROU_1.json.zip")
if not zip_path.exists():
    urllib.request.urlretrieve(
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_1.json.zip", zip_path)
with zipfile.ZipFile(zip_path) as z:
    name = [n for n in z.namelist() if n.endswith(".json")][0]
    with z.open(name) as f:
        counties = gpd.read_file(f).to_crs(CRS_RO)
country = counties.dissolve().geometry.iloc[0]
ro_area_km2 = country.area / 1e6
counties = counties.simplify(150)            # lighter geometry = much faster video frames
country_line = gpd.GeoSeries([country], crs=CRS_RO).simplify(150).boundary

# 2. Observer + DEM ------------------------------------------------------
lat, lon, official_elev = PEAKS[PEAK]
print(f"Peak: {PEAK}")
ox, oy, elev = find_summit(lat, lon)
print(f"  highest DEM cell nearby: {elev:.0f} m (official {official_elev} m)")
z_obs = elev + OBSERVER_H
half = MAX_KM * 1000
n = int(round(2 * half / RES_M))
tf = from_origin(ox - half, oy + half, RES_M, RES_M)
dem = build_dem(tf, n)
ocol, orow = ~tf * (ox, oy)
ocol, orow = ocol - 0.5, orow - 0.5
hs = hillshade(dem, RES_M)

# 3. Rays -> visibility, skyline, statistics ----------------------------------
print("Casting rays ...")
vis_polar, cm_deg, hs_s, r, n_rays, n_steps = cast_rays(dem, hs, RES_M, orow, ocol, z_obs, half)
vis, dist, ray = polar_to_grid(vis_polar, n_rays, n_steps, dem.shape, RES_M, orow, ocol)
in_ro = rasterize([(country, 1)], out_shape=dem.shape, transform=tf, fill=0,
                  dtype="uint8").astype(bool)
mask = (dist <= half) & in_ro
vis_ro = vis & mask
vis_km2 = vis_ro.sum() * RES_M ** 2 / 1e6
pct_country = 100 * vis_km2 / ro_area_km2
# share of Romania seen "so far" as the beam turns clockwise from north
cum_pct = 100 * np.cumsum(np.bincount(ray[vis_ro], minlength=n_rays)) * RES_M ** 2 / 1e6 / ro_area_km2
print(f"Visible: {vis_km2:,.0f} km2 = {pct_country:.1f}% of Romania")

# 4. Panorama strip: what you see looking in every direction ------------------------
top = max(float(cm_deg[:, -1].max()) + 0.7, 0.7)         # degrees above eye level
bottom = top - PANO_V_DEG
PANO_H = int(SIZE_H * 0.17)
alpha_asc = np.linspace(bottom, top, PANO_H)
cols_ray = np.arange(0, n_rays, 2)
ncols = len(cols_ray)
deg_per_col = 360.0 / ncols
IDX = np.empty((PANO_H, ncols), dtype=np.int32)
for c, k in enumerate(cols_ray):
    # first distance step whose running-max angle reaches this elevation = the surface you see there
    IDX[:, c] = np.searchsorted(cm_deg[k], alpha_asc)
sky = IDX >= n_steps
IDXc = np.minimum(IDX, n_steps - 1)
dcol = r[IDXc] / half
shade_p = np.clip(0.45 + 0.8 * hs_s[cols_ray[None, :], IDXc], 0.3, 1.0)
pano = CMAP(np.clip(dcol, 0, 1))[..., :3] * shade_p[..., None]
sk = cm_deg[cols_ray, -1]                                  # skyline angle per column
glow_t = np.exp(-np.clip(alpha_asc[:, None] - sk[None, :], 0, None) / 1.0)
bg_c, glow_c = np.array(to_rgb(BG)), np.array(to_rgb("#8a4dff"))
# fade the near foreground (below the first sample) towards dark so it doesn't read as a flat block
cm0 = cm_deg[cols_ray, 0]
frac = np.clip((alpha_asc[:, None] - bottom) / (cm0[None, :] - bottom + 1e-6), 0, 1)
pano = np.where((IDX == 0)[..., None], pano * (0.3 + 0.7 * frac)[..., None], pano)
pano = np.where(sky[..., None], bg_c + 0.45 * glow_t[..., None] * (glow_c - bg_c), pano)
thick = 1.5 * PANO_V_DEG / PANO_H
edge = (alpha_asc[:, None] <= sk[None, :]) & (alpha_asc[:, None] >= sk[None, :] - thick)
pano = np.where(edge[..., None], 0.95, pano)
pano = (np.clip(pano[::-1], 0, 1) * 255).astype(np.uint8)  # row 0 = highest angle
pad = int(math.ceil(FOV_DEG / 2 / deg_per_col)) + 2        # wrap around so 359 -> 0 is seamless
pano_w = np.concatenate([pano[:, -pad:], pano, pano[:, :pad]], axis=1)
pano_ext = (-pad * deg_per_col, 360 + pad * deg_per_col, bottom, top)

# 5. Map layers (down-sampled for speed) -----------------------------------------
DS = max(1, round(n / 1000))
sl = (slice(None, None, DS), slice(None, None, DS))
hs_d, vis_d, dist_d, ray_d, mask_d = hs[sl], vis[sl], dist[sl], ray[sl].astype(np.int16), mask[sl]
vis_rgb = CMAP(np.clip(dist_d / half, 0, 1))[..., :3] * (0.45 + 0.55 * hs_d)[..., None]
hid_rgb = np.array(to_rgb(LAND))[None, None, :] * (0.3 + 1.6 * hs_d[..., None])
base = np.dstack([np.clip(hid_rgb, 0, 1), mask_d.astype(float)])
rgba_vis = np.dstack([np.clip(vis_rgb, 0, 1) * 255, np.zeros(hs_d.shape)]).astype(np.uint8)
show = vis_d & mask_d
x0, y1 = tf.c, tf.f
map_ext = (x0, x0 + n * RES_M, y1 - n * RES_M, y1)

# 6. Figure -------------------------------------------------------------------
fig = plt.figure(figsize=(SIZE_W / 100, SIZE_H / 100), dpi=100)
fig.patch.set_facecolor(BG)

ax = fig.add_axes([0.125, 0.325, 0.75, 0.60])          # 810 x 810 px square
ax.set_facecolor(BG)
counties.plot(ax=ax, facecolor=LAND, edgecolor=LINE, linewidth=0.5)
ax.imshow(base, extent=map_ext, origin="upper", interpolation="nearest", zorder=2)
im_vis = ax.imshow(rgba_vis, extent=map_ext, origin="upper", interpolation="nearest", zorder=3)
country_line.plot(ax=ax, color="#4a5578", linewidth=1.0, zorder=4)
for rk in (50, 100, 150):
    if rk <= MAX_KM:
        ax.add_patch(Circle((ox, oy), rk * 1000, fill=False, ec="#c7cde0", lw=0.5,
                            ls=(0, (2, 4)), alpha=0.45, zorder=5))
        a135 = math.radians(135)
        ax.text(ox + rk * 1000 * math.sin(a135), oy + rk * 1000 * math.cos(a135), f"{rk} km",
                color="#c7cde0", fontsize=7, ha="center", va="center", zorder=5,
                path_effects=[pe.withStroke(linewidth=2, foreground=BG)])
wedge = Wedge((ox, oy), half, 0, 1, facecolor="white", alpha=0.10, edgecolor="none", zorder=5)
ax.add_patch(wedge)
beam, = ax.plot([ox, ox], [oy, oy + half], color="white", lw=1.4, alpha=0.95, zorder=6)
for s, a in [(600, 0.05), (250, 0.10), (90, 0.18)]:
    ax.scatter([ox], [oy], s=s, color="white", alpha=a, linewidths=0, zorder=6)
ax.scatter([ox], [oy], s=34, marker="^", color="white", edgecolor=BG, linewidth=0.6, zorder=7)
ax.annotate(f"{PEAK}  {elev:,.0f} m", (ox, oy), xytext=(-10, 12), textcoords="offset points", ha="right",
            color="white", fontsize=10, fontweight="bold", zorder=8,
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
ax.set_xlim(ox - half * 1.03, ox + half * 1.03)
ax.set_ylim(oy - half * 1.03, oy + half * 1.03)
ax.set_aspect("equal")
ax.set_axis_off()

axp = fig.add_axes([0.06, 0.115, 0.88, 0.17])
axp.imshow(pano_w, extent=pano_ext, origin="upper", aspect="auto", interpolation="nearest")
axp.set_ylim(bottom, top)
axp.set_yticks([])
axp.axhline(0, color="#c7cde0", lw=0.6, ls=(0, (2, 3)), alpha=0.55)
axp.annotate("eye level", xy=(0.008, 0), xycoords=("axes fraction", "data"), xytext=(0, 3),
             textcoords="offset points", color="#c7cde0", fontsize=8, va="bottom")
for sp in axp.spines.values():
    sp.set_edgecolor("#4a5578")
    sp.set_linewidth(0.8)
axp.tick_params(axis="x", colors="#c7cde0", labelsize=12, length=3, pad=4)

cax = fig.add_axes([0.58, 0.034, 0.36, 0.008])
cb = fig.colorbar(ScalarMappable(norm=Normalize(0, MAX_KM), cmap=CMAP), cax=cax,
                  orientation="horizontal")
cb.set_label("Distance from the summit (km)", color="#c7cde0", fontsize=9, labelpad=3)
cb.ax.xaxis.set_label_position("top")
cb.ax.xaxis.set_tick_params(color="#c7cde0", labelcolor="#c7cde0", labelsize=8)
cb.outline.set_visible(False)

fig.text(0.06, 0.968, f"How much of Romania can you see from {PEAK}?", color="white",
         fontsize=22, fontweight="bold", ha="left", va="center")
fig.text(0.06, 0.943, "A 360° sweep from the summit, and what the view looks like",
         color="#00e5ff", fontsize=13, ha="left", va="center")
txt_dir = fig.text(0.06, 0.303, "", color="white", fontsize=17, fontweight="bold",
                   ha="left", va="center")
txt_pct = fig.text(0.94, 0.303, "", color="#00e5ff", fontsize=17, ha="right", va="center")
fig.text(0.06, 0.012,
         f"#30DayMapChallenge | Day 5: Sight | {AUTHOR}\n"
         f"Data: Copernicus DEM GLO-30 (ESA), GADM 4.1 | Ray-cast viewshed, {RES_M} m grid,\n"
         f"eye height {OBSERVER_H:g} m, earth curvature + refraction. Vertical scale of panorama exaggerated.",
         color="#8a93b0", fontsize=8, ha="left", va="bottom", linespacing=1.4)


def draw_frame(k):
    """k = how many rays have been swept so far (0 .. n_rays-1)."""
    az = k * 360.0 / n_rays
    a = math.radians(az)
    beam.set_data([ox, ox + half * math.sin(a)], [oy, oy + half * math.cos(a)])
    wedge.set_theta1(90 - (az + FOV_DEG / 2))
    wedge.set_theta2(90 - (az - FOV_DEG / 2))
    rgba_vis[..., 3] = (show & (ray_d <= k)) * 255
    im_vis.set_data(rgba_vis)
    txt_dir.set_text(f"Looking {NAMES[int(((az + 22.5) % 360) // 45)]}  ·  {az:03.0f}°")
    txt_pct.set_text(f"{cum_pct[k]:.1f}% of Romania in sight so far")
    lo, hi = az - FOV_DEG / 2, az + FOV_DEG / 2
    axp.set_xlim(lo, hi)
    ticks = [t for t in range(-90, 451, 45) if lo <= t <= hi]
    axp.set_xticks(ticks)
    axp.set_xticklabels([NAMES[(t // 45) % 8] for t in ticks])


def grab():
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


# 7. Render ---------------------------------------------------------------
if PREVIEW_ONLY:
    draw_frame(int(n_rays * 0.35))
    fig.savefig("day5_preview.png", facecolor=BG, dpi=100)
    print("Saved day5_preview.png  (set PREVIEW_ONLY = False for the full video)")
    raise SystemExit

ks = np.linspace(0, n_rays - 1, int(SECONDS * FPS)).astype(int)
schedule = [0] * int(HOLD_START * FPS) + list(ks) + [n_rays - 1] * int(HOLD_END * FPS)
with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=None, pixelformat="yuv420p") as w:
    for i, k in enumerate(schedule):
        draw_frame(k)
        w.append_data(grab())
        if i % 30 == 0:
            print(f"frame {i + 1}/{len(schedule)}")
fig.savefig(OUT_PNG, facecolor=BG, dpi=100)
print(f"Saved {OUT_MP4} and {OUT_PNG}")

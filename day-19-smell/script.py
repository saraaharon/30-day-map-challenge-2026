"""
#30DayMapChallenge | Day 19: Senses: Smell
"The Smell of Vrancea": land cover "emits" a scent whose intensity follows the seasons.

Pipeline
  1. GEE: Corine 2018 + SRTM + Vrancea mask -> one GeoTIFF (EPSG:3844, 250 m)
  2. Split broad-leaved forest by altitude (acacia / linden / beech)  [interpretive]
  3. Blur each scent class (smell has no borders), weight by monthly intensity
  4. Additive neon blending on dark background -> 1080x1350 mp4 (+ optional gif)

pip install earthengine-api rasterio numpy scipy matplotlib imageio imageio-ffmpeg pyproj requests
"""
import io
import zipfile

import ee
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.io import MemoryFile
from scipy.ndimage import gaussian_filter, gaussian_filter1d

# ----------------------------------------------------------------- settings
GEE_PROJECT = "advance-vector-464019-f8"
AUTHOR = "Sara AHARON"
DAY_LINE = f"#30DayMapChallenge | Day 19: Senses: Smell | {AUTHOR}"
SOURCE_LINE = ("Data: Copernicus CORINE Land Cover 2018, SRTM, FAO GAUL | "
               "Scent classes are interpretive: CORINE has no tree species, "
               "so broadleaf forest is split by altitude")

SCALE = 250                 # metres per pixel
FRAMES_PER_MONTH = 10       # 120 frames total
FPS = 20
OUT_MP4 = "day19_smell_vrancea.mp4"
OUT_GIF = "day19_smell_vrancea.gif"
MAKE_GIF = True

BG = "#0b0d1f"
GREY = "#8a93b0"
W, H, DPI = 1080, 1350, 100

# ----------------------------------------------------------------- scents
# key: (label, colour, spatial blur px, gain, 12 monthly intensities Jan..Dec)
SCENTS = {
    "acacia":   ("Acacia blossom",         "#fff3d6", 3, 1.6,
                 [0, 0, 0, 0.05, 1.0, 0.35, 0, 0, 0, 0, 0, 0]),
    "linden":   ("Linden",                 "#c6ff5e", 3, 1.6,
                 [0, 0, 0, 0, 0.05, 0.7, 1.0, 0.25, 0, 0, 0, 0]),
    "beech":    ("Beech & damp leaves",    "#ff9a3c", 3, 1.6,
                 [0.1, 0.1, 0.2, 0.3, 0.2, 0.2, 0.2, 0.3, 0.5, 0.9, 1.0, 0.5]),
    "resin":    ("Pine & spruce resin",    "#19e6a0", 3, 1.6,
                 [0.35, 0.3, 0.3, 0.4, 0.5, 0.7, 1.0, 1.0, 0.7, 0.5, 0.4, 0.4]),
    "hay":      ("Cut hay & meadow",       "#ffd23f", 3, 1.8,
                 [0, 0, 0.05, 0.2, 0.5, 0.9, 1.0, 0.4, 0.1, 0, 0, 0]),
    "vineyard": ("Grape must",             "#b44cff", 3, 2.2,
                 [0.05, 0.05, 0.05, 0.1, 0.15, 0.3, 0.2, 0.3, 0.9, 1.0, 0.5, 0.1]),
    "orchard":  ("Plum orchards & tuica",  "#ff5fa2", 3, 2.5,
                 [0, 0, 0.1, 0.8, 0.25, 0.05, 0.2, 0.6, 0.9, 0.7, 0.5, 0.1]),
    "soil":     ("Wet earth",              "#c97b4a", 3, 1.4,
                 [0.1, 0.2, 0.85, 1.0, 0.5, 0.2, 0.3, 0.45, 0.3, 0.4, 0.2, 0.1]),
    "smoke":    ("Woodsmoke",              "#9aa3c7", 4, 2.6,
                 [1.0, 0.9, 0.5, 0.15, 0, 0, 0, 0, 0.1, 0.4, 0.85, 1.0]),
}

# caption per month: (text, scent key that colours the subtitle)
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
CAPTIONS = [
    ("woodsmoke and cold air",   "smoke"),
    ("cold, spruce and smoke",   "resin"),
    ("wet thawing earth",        "soil"),
    ("plum blossom and soil",    "orchard"),
    ("acacia",                   "acacia"),
    ("linden and cut hay",       "linden"),
    ("hot resin and hay",        "resin"),
    ("ripe plums, dry grass",    "orchard"),
    ("grape must",               "vineyard"),
    ("new wine and tuica",       "vineyard"),
    ("damp leaves and woodsmoke", "beech"),
    ("woodsmoke",                "smoke"),
]

TOWNS = {  # name: (lat, lon)
    "Focsani": (45.6967, 27.1865),
    "Panciu": (45.9, 27.0833),
    "Odobesti": (45.7667, 27.0667),
    "Marasesti": (45.8833, 27.2333),
}


# ----------------------------------------------------------------- data
def get_vrancea():
    for lvl, col in (("level1", "ADM1_NAME"), ("level2", "ADM2_NAME")):
        fc = (ee.FeatureCollection(f"FAO/GAUL/2015/{lvl}")
              .filter(ee.Filter.eq("ADM0_NAME", "Romania"))
              .filter(ee.Filter.eq(col, "Vrancea")))
        if fc.size().getInfo() > 0:
            return fc.geometry()
    print("GAUL name not found, falling back to a bounding box")
    return ee.Geometry.Rectangle([26.3, 45.35, 27.55, 46.05])


def fetch_data():
    ee.Initialize(project=GEE_PROJECT)
    vr = get_vrancea()
    region = vr.bounds().buffer(6000).bounds()  # 6 km margin so scents can bleed

    lc = ee.Image("COPERNICUS/CORINE/V20/100m/2018").select("landcover").rename("lc")
    dem = ee.Image("USGS/SRTMGL1_003").rename("el")
    mask = ee.Image.constant(1).clip(vr).unmask(0).rename("mask")
    img = ee.Image.cat([lc, dem, mask]).toFloat()

    url = img.getDownloadURL({
        "region": region, "scale": SCALE, "crs": "EPSG:3844", "format": "GEO_TIFF",
    })
    raw = requests.get(url, timeout=300).content
    if raw[:2] == b"PK":  # sometimes returned zipped
        z = zipfile.ZipFile(io.BytesIO(raw))
        raw = z.read([n for n in z.namelist() if n.endswith(".tif")][0])
    with MemoryFile(raw) as mf, mf.open() as src:
        arr = src.read()
        b = src.bounds
    lc_a, el_a, mk_a = arr[0], arr[1], arr[2]
    lc_a = np.nan_to_num(lc_a, nan=0)
    el_a = np.nan_to_num(el_a, nan=0)
    return lc_a, el_a, mk_a, (b.left, b.right, b.bottom, b.top)


# ----------------------------------------------------------------- scent fields
def class_masks(lc, el):
    broad = np.isin(lc, [311, 313, 324])
    return {
        "acacia":   broad & (el < 350),
        "linden":   broad & (el >= 350) & (el < 750),
        "beech":    broad & (el >= 750),
        "resin":    lc == 312,
        "hay":      np.isin(lc, [231, 321, 322, 243]),
        "vineyard": lc == 221,
        "orchard":  lc == 222,
        "soil":     np.isin(lc, [211, 212, 242]),
        "smoke":    np.isin(lc, [111, 112]),
    }


def spatial_fields(lc, el):
    masks = class_masks(lc, el)
    fields = {}
    for k, m in masks.items():
        _, _, sigma, gain, _ = SCENTS[k]
        fields[k] = np.clip(gaussian_filter(m.astype("float32"), sigma) * gain, 0, 1)
    return fields


def hillshade(el, cell=SCALE, az=315, alt=45):
    gy, gx = np.gradient(el.astype("float32"), cell)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azr, altr = np.radians(az), np.radians(alt)
    hs = (np.sin(altr) * np.cos(slope)
          + np.cos(altr) * np.sin(slope) * np.cos(azr - aspect))
    return np.clip(hs, 0, 1)


def time_curves():
    """Smooth, looping intensity per scent at frame resolution."""
    n = 12 * FRAMES_PER_MONTH
    t = np.arange(n) / FRAMES_PER_MONTH
    out = {}
    for k, (*_, vals) in SCENTS.items():
        v = np.array(vals, dtype=float)
        xp = np.concatenate([np.arange(-12, 0), np.arange(0, 12), np.arange(12, 24)])
        fp = np.tile(v, 3)
        c = np.interp(t, xp, fp)
        c = gaussian_filter1d(c, sigma=FRAMES_PER_MONTH * 0.35, mode="wrap")
        if c.max() > 0:
            c = c / c.max() * v.max()
        out[k] = c
    return t, out


# ----------------------------------------------------------------- main
def main():
    lc, el, mk, (left, right, bottom, top) = fetch_data()
    fields = spatial_fields(lc, el)
    hs = hillshade(el)
    # smell ignores borders: dim (not cut) outside Vrancea
    soft = gaussian_filter(mk, 2)
    border_w = 0.30 + 0.70 * np.clip(soft, 0, 1)
    t, curves = time_curves()
    nframes = len(t)

    rgbs = {k: np.array(matplotlib.colors.to_rgb(SCENTS[k][1])) for k in SCENTS}
    bg_rgb = np.array(matplotlib.colors.to_rgb(BG))
    terrain = bg_rgb + hs[..., None] * np.array([0.05, 0.06, 0.11])

    # ---- figure
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.04, 0.30, 0.92, 0.58])
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    im = ax.imshow(terrain, extent=(left, right, bottom, top), origin="upper",
                   interpolation="bicubic")
    ax.contour(np.linspace(left, right, mk.shape[1]), np.linspace(top, bottom, mk.shape[0]),
               mk, levels=[0.5], colors=["#ffffff"], linewidths=0.6, linestyles=[(0, (3, 4))],
               alpha=0.45)
    ax.set_xlim(left, right); ax.set_ylim(bottom, top)

    tr = Transformer.from_crs("EPSG:4326", "EPSG:3844", always_xy=True)
    for name, (lat, lon) in TOWNS.items():
        x, y = tr.transform(lon, lat)
        ax.plot(x, y, "o", ms=3, color="white", alpha=0.9)
        ax.text(x + 1500, y + 1200, name, color="white", fontsize=9, alpha=0.85, clip_on=True)

    fig.text(0.06, 0.93, "The Smell of Vrancea", color="white", fontsize=26,
             fontweight="bold", ha="left", va="center")
    sub = fig.text(0.06, 0.895, "", fontsize=14, ha="left", va="center")
    fig.text(0.06, 0.04, DAY_LINE, color=GREY, fontsize=8, ha="left", va="center")
    fig.text(0.06, 0.022, SOURCE_LINE, color=GREY, fontsize=6.5, ha="left", va="center")

    # ---- legend (9 scents, glow with their current intensity)
    lax = fig.add_axes([0.06, 0.095, 0.88, 0.10])
    lax.set_xlim(0, 3); lax.set_ylim(0, 3); lax.axis("off")
    leg = {}
    for i, k in enumerate(SCENTS):
        col, row = i // 3, 2 - i % 3
        dot = lax.scatter([col + 0.06], [row + 0.5], s=70, color=SCENTS[k][1], alpha=0.3)
        txt = lax.text(col + 0.14, row + 0.5, SCENTS[k][0], color="white", fontsize=9,
                       va="center", alpha=0.3)
        leg[k] = (dot, txt)

    # ---- month bar
    bax = fig.add_axes([0.06, 0.215, 0.88, 0.04])
    bax.set_xlim(0, 12); bax.set_ylim(0, 1); bax.axis("off")
    for m, name in enumerate(MONTHS):
        bax.text(m + 0.5, 0.5, name, color=GREY, fontsize=9, ha="center", va="center")
    cursor = Rectangle((0, 0.05), 1, 0.9, facecolor="white", alpha=0.18,
                       edgecolor="white", linewidth=0.8)
    bax.add_patch(cursor)

    frames = []
    for f in range(nframes):
        acc = np.zeros(lc.shape + (3,), dtype="float32")
        for k in SCENTS:
            acc += (fields[k] * curves[k][f])[..., None] * rgbs[k]
        acc *= border_w[..., None]
        glow = 1 - np.exp(-1.7 * acc)
        im.set_data(np.clip(terrain + glow, 0, 1))

        mi = int(round(t[f])) % 12
        text, key = CAPTIONS[mi]
        sub.set_text(f"{MONTHS[mi].capitalize()}: {text}")
        sub.set_color(SCENTS[key][1])
        cursor.set_x(t[f] % 12)
        for k, (dot, txt) in leg.items():
            a = 0.25 + 0.75 * float(curves[k][f])
            dot.set_alpha(a); txt.set_alpha(a)

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if f % 20 == 0:
            print(f"frame {f}/{nframes}")

    with imageio.get_writer(OUT_MP4, fps=FPS, codec="libx264", quality=8,
                            macro_block_size=1) as w:
        for fr in frames:
            w.append_data(fr)
    print("saved", OUT_MP4)

    if MAKE_GIF:
        small = [fr[::2, ::2] for fr in frames[::2]]
        imageio.mimsave(OUT_GIF, small, fps=FPS // 2, loop=0)
        print("saved", OUT_GIF)


if __name__ == "__main__":
    main()

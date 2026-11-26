"""
#30DayMapChallenge | Day 26: Water  ->  "Where the Rain Goes"

Rain falls on all of Romania and flows downhill (HydroSHEDS D8 flow directions).
Rivers draw themselves from the paths of the drops. One drop (the hero) starts in
Focsani and we follow it all the way to the Black Sea, with its elevation profile.

Data: HydroSHEDS 15s (15DIR flow direction, 15CON conditioned DEM) via GEE,
      USDOS LSIB boundaries (Romania mask).
Run:  python day26_water.py          (first run downloads and caches data)
      python day26_water.py --demo   (synthetic terrain, just to test the pipeline)
Out:  day26_water.mp4 (1080x1350) + day26_water_final.png
Needs: earthengine-api numpy matplotlib imageio imageio-ffmpeg
"""
import io, os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ------------------------------------------------------------------ config
GEE_PROJECT = "advance-vector-464019-f8"
AUTHOR = "Sara AHARON"
LON0, LON1, LAT0, LAT1 = 20.0, 30.0, 43.5, 48.5      # download / display box
RES = 240                                            # 15 arc-sec = 240 cells/degree
W, H = int((LON1 - LON0) * RES), int((LAT1 - LAT0) * RES)
START = (27.1867, 45.6967)                           # Focsani (lon, lat)
GALATI = (28.04, 45.43)                              # Siret meets the Danube (label only)

FPS = 30
FRAMES = int(os.environ.get("FRAMES", 480))          # 16 s
N_PART = 50000                                       # rain particles
SUBSTEPS = 2                                         # cells moved per frame
TRAIL_DECAY = 0.90                                   # lower = shorter trails
RIVER_GAIN = 0.05                                    # how fast rivers "draw themselves"
MAX_AGE = 2500

BG = "#0a0818"
CYAN, PINK, GREY = "#27e1ff", "#ff2bd6", "#8a93b0"
OUT_DIR = os.environ.get("OUT_DIR", ".")
CACHE = os.path.join(OUT_DIR, "day26_cache.npz")

# ------------------------------------------------------------------ data
def fetch_from_gee():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    grid = {"dimensions": {"width": W, "height": H},
            "affineTransform": {"scaleX": 1 / RES, "shearX": 0, "translateX": LON0,
                                "shearY": 0, "scaleY": -1 / RES, "translateY": LAT1},
            "crsCode": "EPSG:4326"}
    ro = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(
        ee.Filter.eq("country_na", "Romania"))
    imgs = {
        "dir": ee.Image("WWF/HydroSHEDS/15DIR").select(0).rename("dir").unmask(0).toFloat(),
        "elv": ee.Image("WWF/HydroSHEDS/15CONDEM").select(0).rename("elv").unmask(0).toFloat(),
        "ro": ee.Image.constant(1).clip(ro.geometry()).unmask(0).rename("ro").toFloat(),
    }
    out = {}
    for k, img in imgs.items():                      # one band per request (size limit)
        print("downloading", k, "...")
        raw = ee.data.computePixels({"expression": img, "fileFormat": "NPY", "grid": grid})
        a = np.load(io.BytesIO(raw))
        out[k] = (a[k] if a.dtype.names else a).astype(np.float32).reshape(H, W)
    return out


def synthetic():
    """Fake terrain, only to test the animation code."""
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    z = (2200 * np.exp(-(((x - 700) / 420) ** 2 + ((y - 520) / 330) ** 2))
         + 900 * np.exp(-(((x - 1300) / 300) ** 2 + ((y - 250) / 250) ** 2))
         + 300 - 0.12 * x + 40 * np.sin(x / 37) * np.cos(y / 29))
    z = np.clip(z, 1, None)
    zp = np.pad(z, 1, constant_values=-1)
    best = np.zeros((H, W), np.float32)
    dr = np.zeros((H, W), np.uint8)
    for code, (r, c) in {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1),
                         16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}.items():
        drop = z - zp[1 + r:1 + r + H, 1 + c:1 + c + W]
        m = drop > best
        best[m] = drop[m]
        dr[m] = code
    ro = (((x - 1200) / 900) ** 2 + ((y - 600) / 450) ** 2 < 1).astype(np.float32)
    return {"dir": dr.astype(np.float32), "elv": z, "ro": ro}


def load_data(demo):
    if demo:
        return synthetic()
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return {k: d[k] for k in d.files}
    d = fetch_from_gee()
    np.savez_compressed(CACHE, **d)
    return d


# ------------------------------------------------------------------ flow routing
DIRS = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1),
        64: (-1, 0), 128: (-1, 1)}                   # rows grow southwards
LUT_R, LUT_C = np.zeros(256, np.int32), np.zeros(256, np.int32)
VALID = np.zeros(256, bool)
for k, (r, c) in DIRS.items():
    LUT_R[k], LUT_C[k], VALID[k] = r, c, True


def make_stepper(D):
    def step(r, c):
        code = D[r, c]
        nr, nc = r + LUT_R[code], c + LUT_C[code]
        ok = VALID[code] & (nr >= 0) & (nr < H) & (nc >= 0) & (nc < W)
        return np.where(ok, nr, r), np.where(ok, nc, c), ok
    return step


def trace_drop(D, ELV, r, c):
    path, seen = [], set()
    while 0 <= r < H and 0 <= c < W and (r, c) not in seen and len(path) < 8000:
        path.append((r, c)); seen.add((r, c))
        code = int(D[r, c])
        if not VALID[code]:
            break
        r, c = r + int(LUT_R[code]), c + int(LUT_C[code])
    p = np.array(path)
    lat = LAT1 - (p[:, 0] + 0.5) / RES
    lon = LON0 + (p[:, 1] + 0.5) / RES
    seg_y = np.diff(p[:, 0]) * 463.0
    seg_x = np.diff(p[:, 1]) * 463.0 * np.cos(np.radians(lat[1:]))
    km = np.r_[0, np.cumsum(np.hypot(seg_x, seg_y))] / 1000
    return lon, lat, km, ELV[p[:, 0], p[:, 1]]


# ------------------------------------------------------------------ base layer
def hillshade(z):
    dy, dx = 463.0, 463.0 * math.cos(math.radians(46))
    gy, gx = np.gradient(z * 2.5, dy, dx)
    nx, ny, nz = -gx, gy, np.ones_like(z)            # y points north
    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    az, alt = math.radians(315), math.radians(38)
    shade = (nx * math.sin(az) * math.cos(alt) + ny * math.cos(az) * math.cos(alt)
             + nz * math.sin(alt)) / n
    return np.clip((shade - math.sin(alt)) * 3 + 0.25, 0, 1)


def bmean(a):
    return a.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def bmax(a):
    return a.reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))


def build_base(ELV, RO, D):
    lum = bmean(hillshade(ELV)) ** 1.3
    ro = bmax(RO) > 0
    base = np.array([0.04, 0.03, 0.11]) + lum[..., None] * np.array([0.15, 0.09, 0.27])
    base[~ro] *= 0.55
    sea = bmax((ELV <= 0).astype(np.float32)) > 0
    base[sea & ~ro] = [0.015, 0.015, 0.05]
    er = ro & np.roll(ro, 1, 0) & np.roll(ro, -1, 0) & np.roll(ro, 1, 1) & np.roll(ro, -1, 1)
    base[ro & ~er] = [0.9, 0.17, 0.75]               # neon border
    return base.astype(np.float32)


# ------------------------------------------------------------------ main
def main():
    demo = "--demo" in sys.argv
    d = load_data(demo)
    D = np.clip(d["dir"], 0, 255).astype(np.uint8)
    ELV, RO = d["elv"], d["ro"] > 0
    step = make_stepper(D)
    rng = np.random.default_rng(26)

    # hero drop
    sr, sc = int((LAT1 - START[1]) * RES), int((START[0] - LON0) * RES)
    hlon, hlat, hkm, helv = trace_drop(D, ELV, sr, sc)
    P = len(hlon)
    print(f"hero path: {P} cells, {hkm[-1]:.0f} km, {helv[0]:.0f} m -> {helv[-1]:.0f} m")
    gi = int(np.argmin((hlon - GALATI[0]) ** 2 + (hlat - GALATI[1]) ** 2))
    show_galati = math.hypot(hlon[gi] - GALATI[0], hlat[gi] - GALATI[1]) < 0.15

    # rain
    spawn_cells = np.flatnonzero(RO.ravel() & VALID[D].ravel())

    def spawn(n):
        k = rng.choice(spawn_cells, n)
        return (k // W).astype(np.int32), (k % W).astype(np.int32)

    pr, pc = spawn(N_PART)
    age = rng.integers(0, 400, N_PART)
    E = np.zeros((H, W), np.float32)                 # fading trails
    M = np.zeros((H, W), np.float32)                 # accumulated rivers

    def simulate(grow=True):
        nonlocal pr, pc, age
        E[...] *= TRAIL_DECAY
        for _ in range(SUBSTEPS):
            pr, pc, ok = step(pr, pc)
            E[pr, pc] = 1.0
            if grow:
                np.add.at(M, (pr, pc), RIVER_GAIN)
            age += 1
            dead = ~ok | (age > MAX_AGE)
            n = int(dead.sum())
            if n:
                pr[dead], pc[dead] = spawn(n)
                age[dead] = 0
        if grow:
            np.minimum(M, 1.0, out=M)

    for _ in range(40):                              # pre-roll so the rain is already falling
        simulate(grow=False)

    base = build_base(ELV, RO, D)
    H2, W2 = H // 2, W // 2
    river_col, trail_col = np.array([0.05, 0.55, 0.95]), np.array([0.25, 0.85, 1.0])

    def compose():
        Em, Mm = bmax(E), bmax(M)
        rgb = (base + (Mm ** 0.6)[..., None] * river_col * 0.6
               + (Em ** 1.2)[..., None] * trail_col
               + (Em ** 4)[..., None] * 0.45)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    # figure
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
    fig.text(0.06, 0.93, "Where the Rain Goes", color="white", size=26, weight="bold")
    fig.text(0.06, 0.895, "Every drop finds the sea. This one starts in Focșani.",
             color=CYAN, size=14)
    fig.text(0.06, 0.04, f"#30DayMapChallenge | Day 26: Water | {AUTHOR}",
             color=GREY, size=8)
    fig.text(0.06, 0.022, "Data: HydroSHEDS 15s flow direction & conditioned DEM (Google Earth "
             "Engine), USDOS LSIB. D8 flow routing, particle simulation.", color=GREY, size=8)

    ax = fig.add_axes([0.03, 0.27, 0.94, 0.60])
    ax.axis("off")
    im = ax.imshow(compose(), extent=(LON0, LON1, LAT0, LAT1),
                   aspect=1 / math.cos(math.radians(46)), interpolation="bilinear")
    ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)

    (full,) = ax.plot([], [], color=PINK, lw=1.6, alpha=0.55, solid_capstyle="round")
    (glow,) = ax.plot([], [], color=PINK, lw=7, alpha=0.25, solid_capstyle="round")
    (comet,) = ax.plot([], [], color="#ffd6f6", lw=2.6, solid_capstyle="round")
    head = ax.scatter([], [], s=90, c="white", edgecolors=PINK, linewidths=2, zorder=5)
    ax.scatter([hlon[0]], [hlat[0]], s=18, c="white", zorder=4)
    ax.text(hlon[0] - 0.12, hlat[0] + 0.1, "Focșani", color="white", size=12,
            weight="bold", ha="right")
    t_gal = ax.text(hlon[gi] + 0.12, hlat[gi] + 0.1, "Siret meets the Danube",
                    color="white", size=10, ha="left", visible=False)
    t_sea = ax.text(hlon[-1] - 0.12, hlat[-1] - 0.14, "Black Sea", color=CYAN, size=13,
                    weight="bold", ha="right", visible=False)

    pax = fig.add_axes([0.09, 0.09, 0.82, 0.11], facecolor=BG)
    for s in ("top", "right", "left"):
        pax.spines[s].set_visible(False)
    pax.spines["bottom"].set_color(GREY)
    pax.tick_params(colors=GREY, labelsize=8)
    pax.set_xlim(0, hkm[-1]); pax.set_ylim(0, max(helv.max() * 1.1, 10))
    pax.set_xlabel("km travelled by the drop", color=GREY, size=8)
    pax.set_ylabel("elevation (m)", color=GREY, size=8)
    pax.plot(hkm, helv, color=GREY, lw=1, alpha=0.5)
    (pprog,) = pax.plot([], [], color=CYAN, lw=2)
    pdot = pax.scatter([], [], s=40, c="white", zorder=5)
    hud = fig.text(0.91, 0.215, "", ha="right", color="white", size=13, weight="bold")

    F0, F1 = 25, FRAMES - 45
    frames = []
    for f in range(FRAMES):
        simulate()
        t = min(max((f - F0) / (F1 - F0), 0), 1)
        t = 0.65 * t + 0.35 * t * t * (3 - 2 * t)    # gentle ease
        i = int(t * (P - 1))
        im.set_data(compose())
        full.set_data(hlon[:i + 1], hlat[:i + 1])
        glow.set_data(hlon[max(0, i - 70):i + 1], hlat[max(0, i - 70):i + 1])
        comet.set_data(hlon[max(0, i - 70):i + 1], hlat[max(0, i - 70):i + 1])
        head.set_offsets([[hlon[i], hlat[i]]])
        pprog.set_data(hkm[:i + 1], helv[:i + 1])
        pdot.set_offsets([[hkm[i], helv[i]]])
        hud.set_text(f"{hkm[i]:4.0f} km  ·  {helv[i]:4.0f} m")
        t_gal.set_visible(show_galati and i >= gi)
        t_sea.set_visible(i >= P - 1)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        if f % 30 == 0:
            print(f"frame {f}/{FRAMES}")

    fig.savefig(os.path.join(OUT_DIR, "day26_water_final.png"), facecolor=BG)
    out = os.path.join(OUT_DIR, "day26_water.mp4")
    imageio.mimsave(out, frames, fps=FPS, quality=8, macro_block_size=1)
    print("saved", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
#30DayMapChallenge 2026 | Day 28: Senses: Feeling
"Stars turning over Focsani" - a calm night-sky animation.

The real sky above Focsani (45.70 N, 27.19 E) rotating around Polaris for a
winter night, with star trails slowly building up above a quiet Carpathian
horizon. Star positions come from the HYG catalogue (Hipparcos / Yale / Gliese)
and are computed properly (sidereal time -> hour angle -> alt/az), then drawn
in a stereographic projection looking north.

Needs:  pip install numpy pandas pillow matplotlib imageio-ffmpeg
Run:    python day28_stars_over_focsani.py              # full video (16 s)
        python day28_stars_over_focsani.py --seconds 3  # quick test
Output: day28_stars_over_focsani.mp4  +  day28_stars_over_focsani.png (still)
"""
import argparse
import datetime as dt
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import imageio_ffmpeg
from matplotlib import font_manager
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- #
# SETTINGS - tweak these
# --------------------------------------------------------------------------- #
LAT, LON = 45.6960, 27.1860                 # Focsani
START_UTC = dt.datetime(2026, 12, 13, 17, 30)   # 19:30 local (EET, UTC+2)
HOURS = 7.0                                 # simulated night length
UTC_OFFSET = 2                              # Romania in winter

W, H = 1080, 1350                           # 4:5 like Day 16
SS = 2                                      # supersampling (2 = smooth lines)
FPS = 30
ALT0, AZ0 = 35.0, 0.0                       # view centre: looking north, 35 deg up
SCALE = 722.0                               # px per stereographic unit (zoom)
MAG_LIMIT = 6.3

TRAIL_GAIN = 0.34                           # trail brightness vs. the star itself
GLOW = 0.75                                 # soft bloom amount
FADE_IN, FADE_OUT = 1.0, 1.6                # seconds

TITLE = "Focșani"
SUBTITLE = "45.70°N   27.19°E"
CREDIT = "#30DayMapChallenge  |  Day 28: Senses: Feeling  |  Sara AHARON"
DATA = "Data: HYG star catalogue v4.1 (Hipparcos, Yale BSC, Gliese)  ·  horizon illustrative"

CATALOG_URL = ("https://raw.githubusercontent.com/astronexus/HYG-Database/"
               "main/hyg/CURRENT/hygdata_v41.csv")
CATALOG = Path("hygdata_v41.csv")

# colours (RGB)
SKY_STOPS_ALT = [0, 6, 20, 45, 90]
SKY_STOPS_RGB = [(44, 88, 118), (26, 52, 90), (11, 22, 54), (6, 10, 30), (3, 5, 18)]
RIDGES = [  # (seed, base alt deg, amplitude deg, colour)
    (11, 7.5, 3.2, (26, 44, 78)),     # far, hazy
    (23, 4.0, 2.6, (14, 25, 52)),     # middle
    (37, 1.3, 1.7, (6, 10, 22)),      # near, almost black
]
MIST = (60, 105, 135)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_stars():
    if not CATALOG.exists():
        print("Downloading HYG star catalogue (one time, ~34 MB)...")
        urllib.request.urlretrieve(CATALOG_URL, CATALOG)
    d = pd.read_csv(CATALOG, usecols=["id", "proper", "ra", "dec", "mag", "ci"])
    d = d[(d["id"] != 0) & (d["mag"] < MAG_LIMIT)].reset_index(drop=True)  # drop the Sun
    d["ci"] = d["ci"].fillna(0.6)
    return d


def bv_to_rgb(ci):
    """Colour index (B-V) -> soft star colour."""
    stops = [-0.3, 0.0, 0.4, 0.8, 1.4, 2.0]
    cols = np.array([(150, 185, 255), (190, 212, 255), (255, 250, 240),
                     (255, 226, 170), (255, 178, 118), (255, 140, 90)], float)
    return np.stack([np.interp(ci, stops, cols[:, k]) for k in range(3)], axis=1)


def camera():
    a0, z0 = np.radians(ALT0), np.radians(AZ0)
    f = np.array([np.sin(z0) * np.cos(a0), np.cos(z0) * np.cos(a0), np.sin(a0)])
    r = np.array([np.cos(z0), -np.sin(z0), 0.0])
    u = np.cross(r, f)
    return f, r, u


def project(p, f, r, u):
    """unit vectors (east, north, up) -> stereographic screen px at SS resolution."""
    d = p @ f
    k = 2.0 / (1.0 + d + 1e-9)
    X, Y = k * (p @ r), k * (p @ u)
    return W * SS / 2 + X * SCALE * SS, H * SS / 2 - Y * SCALE * SS, d


def pixel_directions(f, r, u):
    """Altitude and azimuth (deg) for every pixel of the SS-sized frame."""
    ix = (np.arange(W * SS, dtype=np.float32) + 0.5 - W * SS / 2) / (SCALE * SS)
    iy = (H * SS / 2 - (np.arange(H * SS, dtype=np.float32) + 0.5)) / (SCALE * SS)
    X, Y = np.meshgrid(ix, iy)
    rho2 = X * X + Y * Y
    rho = np.sqrt(rho2) + 1e-9
    cos_t = (4 - rho2) / (4 + rho2)
    sin_t = 4 * rho / (4 + rho2)
    k = sin_t / rho
    d = [cos_t * f[i] + k * (X * r[i] + Y * u[i]) for i in range(3)]  # E, N, Up
    alt = np.degrees(np.arcsin(np.clip(d[2], -1, 1)))
    az = np.degrees(np.arctan2(d[0], d[1]))
    return alt.astype(np.float32), az.astype(np.float32)


def ridge_profile(az, seed, base, amp):
    rng = np.random.default_rng(seed)
    h = np.zeros_like(az)
    for freq, weight in [(0.018, 1.0), (0.045, 0.7), (0.11, 0.38), (0.27, 0.16), (0.6, 0.06)]:
        h += weight * np.sin(2 * np.pi * freq * az + rng.uniform(0, 2 * np.pi))
    return base + amp * h / 1.6


def build_sky_and_ridges(alt, az):
    """Static layers: sky gradient (SS res) and opaque ridge image + mask."""
    sky = np.stack([np.interp(alt, SKY_STOPS_ALT, [c[k] for c in SKY_STOPS_RGB])
                    for k in range(3)], axis=-1).astype(np.float32)
    ridge_rgb = np.zeros_like(sky)
    ridge_mask = np.zeros(alt.shape, bool)
    for seed, base, amp, col in RIDGES:  # far -> near
        top = ridge_profile(az, seed, base, amp)
        m = alt < top
        mist = (np.clip(1 - (alt + 6) / (top + 6), 0, 1) *
                np.clip((alt + 10) / 6, 0, 1))[..., None] * 0.38   # mist hugs the horizon only
        c = np.array(col, np.float32)[None, None, :] * (1 - mist) + np.array(MIST, np.float32) * mist
        ridge_rgb[m] = c[m] if c.shape[:2] == alt.shape else c
        ridge_mask |= m
    return sky.astype(np.uint8), ridge_rgb.astype(np.uint8), ridge_mask


def spaced_text(draw, xy, text, font, fill, spacing=0):
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + spacing


def text_width(font, text, spacing=0):
    return sum(font.getlength(ch) + spacing for ch in text) - spacing


def build_compass(fpath, polaris_alt):
    """Static RGBA overlay: compass ring with degree ticks, view wedge and
    'where to look' text. Drawn at 4x and downsampled for smooth lines."""
    S = 4
    OW, OH = 380, 290
    R, cx, cy = 74, OW - 112, 118
    img = Image.new("RGBA", (OW * S, OH * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ring = (120, 146, 188, 200)
    soft = (130, 170, 230)
    gold = (232, 200, 140, 255)
    pale = (172, 192, 222, 255)
    dim = (134, 156, 192, 235)

    def pt(az, rad):
        a = np.radians(az)
        return ((cx + rad * np.sin(a)) * S, (cy - rad * np.cos(a)) * S)

    # field-of-view wedge (what the frame covers, looking north)
    half = 40
    arc = [pt(a, R) for a in np.linspace(-half, half, 60)]
    d.polygon([(cx * S, cy * S)] + arc, fill=soft + (48,))
    d.line([(cx * S, cy * S), pt(-half, R)], fill=soft + (120,), width=S)
    d.line([(cx * S, cy * S), pt(half, R)], fill=soft + (120,), width=S)

    # ring + ticks
    d.ellipse([(cx - R) * S, (cy - R) * S, (cx + R) * S, (cy + R) * S], outline=ring, width=S)
    for az in range(0, 360, 5):
        ln = 11 if az % 30 == 0 else (7 if az % 10 == 0 else 3.5)
        d.line([pt(az, R), pt(az, R - ln)], fill=ring, width=S if az % 30 == 0 else max(1, S // 2))

    # degree numbers every 30 deg (inside), cardinals outside
    f_num = ImageFont.truetype(fpath, 11 * S)
    f_card = ImageFont.truetype(fpath, 19 * S)
    for az in range(0, 360, 30):
        if az % 90 == 0:
            continue
        x, y = pt(az, R - 22)
        d.text((x, y), f"{az}", font=f_num, fill=dim, anchor="mm")
    for az, name in [(0, "N"), (90, "E"), (180, "S"), (270, "W")]:
        x, y = pt(az, R + 17)
        d.text((x, y), name, font=f_card, fill=gold if az == 0 else pale, anchor="mm")
    for az in (90, 180, 270):
        x, y = pt(az, R - 20)
        d.text((x, y), f"{az}°", font=f_num, fill=dim, anchor="mm")

    # looking-direction needle (north)
    d.line([(cx * S, cy * S), pt(0, R - 14)], fill=gold, width=2 * S)
    d.polygon([pt(0, R - 4), pt(-4.5, R - 17), pt(4.5, R - 17)], fill=gold)
    d.ellipse([(cx - 3) * S, (cy - 3) * S, (cx + 3) * S, (cy + 3) * S], fill=gold)

    # text, right-aligned
    f1 = ImageFont.truetype(fpath, 21 * S)
    f2 = ImageFont.truetype(fpath, 16 * S)
    right = OW - 6
    lines = [
        ("FACE NORTH  ·  AZIMUTH 0°", f1, gold, 3, 222),
        (f"TILT UP 35°  ·  POLARIS ≈ {polaris_alt:.0f}° UP", f2, pale, 2, 252),
    ]
    for txt, fnt, colr, sp, y in lines:
        w = text_width(ImageFont.truetype(fnt.path, fnt.size // S), txt, sp)
        x = right - w
        for ch in txt:
            d.text((x * S, y * S), ch, font=fnt, fill=colr)
            x += ImageFont.truetype(fnt.path, fnt.size // S).getlength(ch) + sp
    return img.resize((OW, OH), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--out", default="day28_stars_over_focsani")
    args = ap.parse_args()
    N = max(int(args.seconds * FPS), 10)

    stars = load_stars()
    n_stars = len(stars)
    print(f"{n_stars} stars, {N} frames")

    ra = np.radians(stars["ra"].values * 15.0)
    dec = np.radians(stars["dec"].values)
    mag = stars["mag"].values
    col0 = bv_to_rgb(stars["ci"].values)
    lat = np.radians(LAT)

    # brightness / size from magnitude
    bright0 = np.clip(1.22 - 0.165 * mag, 0.16, 1.0)
    radius = np.clip(0.7 + 0.42 * (MAG_LIMIT - mag), 0.8, 4.4)
    rng = np.random.default_rng(5)
    tw_f = rng.uniform(0.3, 1.1, n_stars)
    tw_p = rng.uniform(0, 2 * np.pi, n_stars)

    # ---- sky positions for every frame -------------------------------------
    jd0 = (START_UTC - dt.datetime(2000, 1, 1, 12)).total_seconds() / 86400.0 + 2451545.0
    jd = jd0 + np.linspace(0, HOURS / 24.0, N)
    gmst = np.radians((280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360.0)
    lst = gmst + np.radians(LON)
    Hang = lst[:, None] - ra[None, :]                                  # (N, stars)
    east = -np.cos(dec) * np.sin(Hang)
    north = np.sin(dec) * np.cos(lat) - np.cos(dec) * np.sin(lat) * np.cos(Hang)
    up = np.sin(dec) * np.sin(lat) + np.cos(dec) * np.cos(lat) * np.cos(Hang)
    P = np.stack([east, north, up], axis=-1).astype(np.float32)        # (N, stars, 3)

    f, r, u = camera()
    px, py, dd = project(P.reshape(-1, 3), f, r, u)
    px, py = px.reshape(N, n_stars), py.reshape(N, n_stars)
    margin = 20 * SS
    visible = ((up > 0.0) & (dd.reshape(N, n_stars) > -0.2) &
               (px > -margin) & (px < W * SS + margin) &
               (py > -margin) & (py < H * SS + margin))
    alt_deg = np.degrees(np.arcsin(np.clip(up, -1, 1)))
    ext = np.clip(alt_deg / 10.0, 0, 1) ** 0.8                          # horizon extinction

    # ---- static layers ------------------------------------------------------
    alt, az = pixel_directions(f, r, u)
    sky_bg, ridge_rgb, ridge_mask = build_sky_and_ridges(alt, az)
    del alt, az
    sky_bg16 = sky_bg.astype(np.uint16)

    # Polaris
    pol = stars.index[stars["proper"] == "Polaris"]
    pol = int(pol[0]) if len(pol) else None

    # fonts
    fpath = font_manager.findfont("DejaVu Sans")
    f_title = ImageFont.truetype(fpath, 56)
    f_sub = ImageFont.truetype(fpath, 24)
    f_clock = ImageFont.truetype(fpath, 22)
    f_small = ImageFont.truetype(fpath, 19)
    f_tiny = ImageFont.truetype(fpath, 16)

    # compass overlay (upper right): Polaris altitude ~ latitude
    pol_alt = float(alt_deg[0, pol]) if pol is not None else LAT
    compass = build_compass(fpath, pol_alt)
    compass_rgb, compass_a = compass.convert("RGB"), compass.split()[3]
    compass_xy = (W - 64 - compass.width, 40)

    trail_img = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
    trail_draw = ImageDraw.Draw(trail_img)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out_mp4 = f"{args.out}.mp4"
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "16", "-preset", "medium",
         "-movflags", "+faststart", out_mp4],
        stdin=subprocess.PIPE)

    blur = ImageFilter.GaussianBlur(radius=3 * SS)

    for i in range(N):
        t = i / FPS
        # --- trails: extend each visible star's path since last frame
        if i > 0:
            idx = np.nonzero(visible[i] & visible[i - 1])[0]
            cols = (col0[idx] * (bright0[idx] * ext[i, idx] * TRAIL_GAIN)[:, None]).astype(np.uint8)
            for j, c in zip(idx, cols.tolist()):
                trail_draw.line([(px[i - 1, j], py[i - 1, j]), (px[i, j], py[i, j])],
                                fill=tuple(c), width=2 if mag[j] < 2.5 else 1)

        # --- stars (fresh layer each frame, with gentle twinkle)
        star_img = Image.new("RGB", (W * SS, H * SS), (0, 0, 0))
        sd = ImageDraw.Draw(star_img)
        idx = np.nonzero(visible[i])[0]
        tw = 1 + 0.10 * np.sin(2 * np.pi * t * tw_f[idx] + tw_p[idx])
        cols = np.clip(col0[idx] * (bright0[idx] * ext[i, idx] * tw)[:, None], 0, 255).astype(np.uint8)
        for j, c in zip(idx, cols.tolist()):
            rr = radius[j] * (0.9 + 0.1 * ext[i, j])
            sd.ellipse([px[i, j] - rr, py[i, j] - rr, px[i, j] + rr, py[i, j] + rr], fill=tuple(c))

        layer = ImageChops.add(trail_img, star_img)
        glow = layer.filter(blur).point(lambda v: int(v * GLOW))
        layer = ImageChops.add(layer, glow)

        sky = np.minimum(sky_bg16 + np.asarray(layer, np.uint16), 255).astype(np.uint8)
        full = np.where(ridge_mask[..., None], ridge_rgb, sky)
        frame = Image.fromarray(full).reduce(SS)                      # downsample -> antialias

        # --- text / annotations (1x)
        fade = min(1.0, t / FADE_IN) if FADE_IN else 1.0
        fade_out = min(1.0, (args.seconds - t) / FADE_OUT) if FADE_OUT else 1.0
        ann = min(1.0, max(0.0, (t - 1.5) / 1.5))                     # annotations ease in

        draw = ImageDraw.Draw(frame)
        def tone(rgb, a=1.0):
            return tuple(int(c * a) for c in rgb)

        spaced_text(draw, (64, 64), TITLE.upper(), f_title, tone((214, 226, 244), ann), spacing=9)
        spaced_text(draw, (68, 134), SUBTITLE, f_sub, tone((150, 172, 204), ann), spacing=4)
        local = START_UTC + dt.timedelta(hours=HOURS * i / (N - 1) + UTC_OFFSET)
        spaced_text(draw, (68, 172), local.strftime("%H:%M  ·  %d %b %Y").upper(),
                    f_clock, tone((232, 200, 140), ann), spacing=3)

        if pol is not None and visible[i, pol]:
            cx, cy = px[i, pol] / SS, py[i, pol] / SS
            draw.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=tone((232, 200, 140), 0.55 * ann), width=1)
            spaced_text(draw, (cx + 26, cy - 11), "POLARIS", f_small, tone((232, 200, 140), 0.8 * ann), spacing=3)

        frame.paste(compass_rgb, compass_xy, compass_a.point(lambda v: int(v * ann)))

        spaced_text(draw, (64, H - 96), CREDIT, f_small, tone((140, 162, 196), ann), spacing=1)
        spaced_text(draw, (64, H - 68), DATA, f_tiny, tone((104, 126, 160), ann), spacing=0.5)

        arr = np.asarray(frame)
        if i == N - 1:
            Image.fromarray(arr).save(f"{args.out}.png")              # unfaded still

        k = fade * fade_out
        if k < 1.0:
            arr = (arr * k).astype(np.uint8)
        proc.stdin.write(arr.tobytes())

        if i % 30 == 0:
            print(f"  frame {i}/{N}")

    proc.stdin.close()
    proc.wait()
    print(f"Done -> {out_mp4} and {args.out}.png")


if __name__ == "__main__":
    main()

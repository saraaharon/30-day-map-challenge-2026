#!/usr/bin/env python3
"""
VRANCEA HUMS
#30DayMapChallenge | Day 11: Senses: Sound | Sara AHARON

Turns Vrancea earthquakes (1977 -> today) into an animated map WITH sound.
  - depth     -> pitch  (deep = low, shallow = high, snapped to A minor pentatonic)
  - magnitude -> loudness, note length, size of the ripple
  - longitude -> stereo pan (west = left, east = right)
  - quake rate-> a low "earth rumble" bed that swells when things get busy
  - M>=5.5    -> extra sub-bass thud

Output: 1080x1350 mp4 (with audio)

Install:
    pip install pandas numpy matplotlib scipy imageio imageio-ffmpeg
    pip install geopandas        # optional, only for the Romania border

Run:
    python vrancea_hums.py                     # downloads USGS catalog, full render
    python vrancea_hums.py --minmag 3.5        # denser, more notes
    python vrancea_hums.py --seconds 30 --fps 24
    python vrancea_hums.py --demo --seconds 4  # offline test with fake data
"""
import argparse
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, fftconvolve
import imageio.v2 as imageio
import imageio_ffmpeg

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DAY, THEME, AUTHOR = 11, "Senses: Sound", "Sara AHARON"
TITLE = "Vrancea Hums"

# Map window (Vrancea seismic zone + surroundings)
LON0, LON1 = 25.5, 27.9
LAT0, LAT1 = 45.0, 46.4
DEPTH_MAX = 180

FOCSANI = (27.1786, 45.6969)
CITIES = {"Brașov": (25.5887, 45.6427), "Buzău": (26.8333, 45.1500)}

START = pd.Timestamp("1900-01-01", tz="UTC")   # catalog start
SPLIT = pd.Timestamp("1977-01-01", tz="UTC")   # modern catalog starts here (the big one is 4 Mar 1977)

# synthwave palette
BG = "#0b0720"
GRID = "#2a1a5e"
MUTED = "#8a93b0"
PINK = "#ff2e97"
CYAN = "#05d9e8"
CMAP = LinearSegmentedColormap.from_list("depth", [PINK, "#b967ff", CYAN])

NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_50m_admin_0_countries.geojson")

SR = 44100
RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------
def usgs_url(minmag, end):
    return ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
            f"&starttime={START:%Y-%m-%d}&endtime={end:%Y-%m-%d}"
            f"&minlatitude={LAT0}&maxlatitude={LAT1}"
            f"&minlongitude={LON0}&maxlongitude={LON1}"
            f"&minmagnitude={minmag}&orderby=time-asc&limit=20000")


def make_demo(n=500):
    """Fake catalog so you can test the pipeline offline."""
    t = START + pd.to_timedelta(np.sort(RNG.uniform(0, 365.25 * 126, n)), unit="D")
    df = pd.DataFrame({
        "time": t,
        "longitude": RNG.normal(26.6, 0.25, n).clip(LON0 + .1, LON1 - .1),
        "latitude": RNG.normal(45.7, 0.2, n).clip(LAT0 + .1, LAT1 - .1),
        "depth": np.where(RNG.random(n) < .85, RNG.normal(100, 25, n), RNG.uniform(5, 40, n)).clip(2, 175),
        "mag": 4 + RNG.exponential(0.45, n),
    })
    df.loc[n // 4, ["mag", "depth"]] = [7.7, 150]
    df.loc[3, ["mag", "depth"]] = [7.4, 94]
    df.loc[n // 2, ["mag", "depth"]] = [6.9, 91]
    return df


def load_quakes(args):
    if args.demo:
        df = make_demo()
    else:
        end = datetime.now(timezone.utc)
        if os.path.exists(args.csv):
            df = pd.read_csv(args.csv)
        else:
            print("Downloading USGS catalog ...")
            df = pd.read_csv(usgs_url(min(args.minmag, 3.0), end))
            df.to_csv(args.csv, index=False)
            print(f"Saved {args.csv} ({len(df)} events)")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df[["time", "latitude", "longitude", "depth", "mag"]].dropna(subset=["time", "latitude", "longitude", "mag"])
    df["depth"] = df["depth"].fillna(100.0)  # historical events often lack depth; Vrancea is mostly ~100 km
    df = df[df.longitude.between(LON0, LON1) & df.latitude.between(LAT0, LAT1)]
    # before 1977 only large quakes are reliably recorded
    df = df[np.where(df.time < SPLIT, df.mag >= args.premag, df.mag >= args.minmag)]
    df = df.sort_values("time").reset_index(drop=True)
    print(f"{len(df)} earthquakes ({(df.time < SPLIT).sum()} before 1977)")
    return df


def video_time(ts, args, end_ts):
    """catalog time -> video seconds (1900-1976 is a short, slower-to-skim intro; 1977+ gets the long part)"""
    pre = (ts - START).dt.total_seconds() / (SPLIT - START).total_seconds()
    post = (ts - SPLIT).dt.total_seconds() / (end_ts - SPLIT).total_seconds()
    return np.where(ts < SPLIT,
                    args.lead + pre * args.pre_seconds,
                    args.lead + args.pre_seconds + post * args.seconds)


def date_at(t, args, end_ts):
    """video seconds -> catalog date (for the year counter)"""
    t0 = args.lead
    if t <= t0:
        return START
    if t < t0 + args.pre_seconds:
        return START + (SPLIT - START) * ((t - t0) / args.pre_seconds)
    q = np.clip((t - t0 - args.pre_seconds) / args.seconds, 0, 1)
    return SPLIT + (end_ts - SPLIT) * q


# ----------------------------------------------------------------------------
# AUDIO
# ----------------------------------------------------------------------------
def midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def lowpass(x, cutoff):
    sos = butter(4, cutoff, btype="low", fs=SR, output="sos")
    return sosfilt(sos, x)


def build_audio(vt, lon, depth, mag, total_sec, minmag):
    n_total = int((total_sec + 2) * SR)
    L, R = np.zeros(n_total), np.zeros(n_total)

    # A minor pentatonic, A1 (55 Hz) -> ~A5
    offs = [0, 3, 5, 7, 10]
    notes = np.array([33 + 12 * o + s for o in range(4) for s in offs] + [81])

    def add(start, sig, gl, gr):
        if start >= n_total:
            return
        end = min(n_total, start + len(sig))
        L[start:end] += sig[: end - start] * gl
        R[start:end] += sig[: end - start] * gr

    for i in range(len(vt)):
        m, d = mag[i], depth[i]
        idx = int(round((1 - np.clip(d, 0, DEPTH_MAX) / DEPTH_MAX) * (len(notes) - 1)))
        f = float(midi_to_hz(notes[idx]))
        dur = float(np.clip(0.35 + 0.55 * (m - minmag), 0.35, 3.0))
        n = int((dur + 0.4) * SR)
        t = np.arange(n) / SR
        env = (1 - np.exp(-t / 0.004)) * np.exp(-t / (dur / 3.0))
        w = (np.sin(2 * np.pi * f * t)
             + 0.35 * np.sin(2 * np.pi * 2 * f * t + 0.3)
             + 0.15 * np.sin(2 * np.pi * 3.01 * f * t))
        if d > 70:  # deep quakes get a sub layer
            w += 0.5 * np.sin(2 * np.pi * f * 0.5 * t)
        amp = 0.05 * 10 ** (0.18 * (m - minmag))
        sig = amp * env * w

        pan = (lon[i] - LON0) / (LON1 - LON0)
        gl, gr = np.cos(pan * np.pi / 2), np.sin(pan * np.pi / 2)
        start = int(vt[i] * SR)
        add(start, sig, gl, gr)

        if m >= 5.5:  # big ones: rumble thud
            nr = int(4 * SR)
            tr = np.arange(nr) / SR
            thud = lowpass(RNG.standard_normal(nr), 90)
            thud /= np.abs(thud).max() + 1e-9
            thud *= (1 - np.exp(-tr / 0.05)) * np.exp(-tr / 1.2) * 0.35 * min(1.0, (m - 5) / 2)
            add(start, thud, 1.0, 1.0)

    # rumble bed that follows the quake rate
    bins = np.arange(0, total_sec + 0.25, 0.25)
    rate, _ = np.histogram(vt, bins=bins)
    rate = np.convolve(rate.astype(float), np.ones(8) / 8, mode="same")
    rate = rate / (rate.max() + 1e-9)
    env_bed = 0.010 + 0.035 * rate
    centers = (bins[:-1] + bins[1:]) / 2
    env_full = np.interp(np.arange(n_total) / SR, centers, env_bed)
    brown = lowpass(RNG.standard_normal(n_total), 70)
    brown /= brown.std() + 1e-9
    L += brown * env_full
    R += brown * env_full

    # light reverb
    ir_len = int(1.4 * SR)
    ir = RNG.standard_normal(ir_len) * np.exp(-np.arange(ir_len) / SR / 0.45)
    ir /= np.sqrt((ir ** 2).sum())
    L = L + 0.35 * fftconvolve(L, ir)[:n_total]
    R = R + 0.35 * fftconvolve(R, ir)[:n_total]

    out = np.stack([L, R], axis=1)
    out = out[: int(total_sec * SR)]
    out *= 0.89 / (np.abs(out).max() + 1e-9)
    fade = int(1.5 * SR)
    out[-fade:] *= np.linspace(1, 0, fade)[:, None]
    return (out * 32767).astype(np.int16)


# ----------------------------------------------------------------------------
# VISUALS
# ----------------------------------------------------------------------------
def draw_borders(ax):
    try:
        import geopandas as gpd
        from shapely.geometry import box
        gdf = gpd.read_file(NE_URL).clip(box(LON0 - 1, LAT0 - 1, LON1 + 1, LAT1 + 1))
        gdf.boundary.plot(ax=ax, color="#6a3fb5", linewidth=1.1, alpha=0.9)
    except Exception as e:  # border is optional
        print("Border skipped:", e)


def render(args, ev, vt, total_sec, out_path, end_ts):
    fps = args.fps
    n_frames = int(total_sec * fps)
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    lon, lat = ev.longitude.values, ev.latitude.values
    dep, mag = ev.depth.values, ev.mag.values
    base = CMAP(np.clip(dep / 170.0, 0, 1))
    s_core = 6 * (mag - args.minmag + 1) ** 1.5
    life = np.where(mag >= 6, 3.0, np.where(mag >= 5, 1.8, 1.1))
    big = np.where(mag >= 6.0)[0]

    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)

    # ---- title block (project layout standard)
    fig.text(0.06, 0.93, TITLE, color="white", size=26, weight="bold")
    fig.text(0.06, 0.895, "Earthquakes as sound: deep = low, shallow = high, big = loud",
             color=CYAN, size=14)
    fig.text(0.06, 0.045, f"#30DayMapChallenge | Day {DAY}: {THEME} | {AUTHOR}",
             color=MUTED, size=11)
    fig.text(0.06, 0.022,
             f"Data: USGS ComCat, {START.year}-{end_ts.year} (M>={args.minmag:g}; M>={args.premag:g} before 1977) "
             f"| Sound: synthesized in Python | Basemap: Natural Earth",
             color=MUTED, size=9)

    # ---- map
    ax = fig.add_axes([0.12, 0.31, 0.82, 0.56], facecolor=BG)
    for x in np.arange(LON0, LON1 + 0.01, 0.5):
        ax.axvline(x, color=GRID, lw=0.6)
    for y in np.arange(LAT0, LAT1 + 0.01, 0.25):
        ax.axhline(y, color=GRID, lw=0.6)
    draw_borders(ax)
    ax.set_xlim(LON0, LON1)
    ax.set_ylim(LAT0, LAT1)
    ax.set_aspect(1 / np.cos(np.radians(45.7)), adjustable="box")
    ax.set_xticks(np.arange(LON0, LON1 + 0.01, 0.5))
    ax.set_xticklabels([f"{x:.1f}°E" for x in np.arange(LON0, LON1 + 0.01, 0.5)])
    ax.set_yticks(np.arange(LAT0, LAT1 + 0.01, 0.25))
    ax.set_yticklabels([f"{y:.2f}°N" for y in np.arange(LAT0, LAT1 + 0.01, 0.25)])
    ax.tick_params(colors=MUTED, labelsize=10, length=0)
    ax.set_xlabel("Geodetic longitude", color=MUTED, size=12, labelpad=8)
    ax.set_ylabel("Geodetic latitude", color=MUTED, size=12, labelpad=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)

    for name, (cx, cy) in CITIES.items():
        ax.plot(cx, cy, "o", color="white", ms=3, alpha=0.8, zorder=5)
        ax.text(cx + 0.03, cy + 0.02, name, color="white", size=9, alpha=0.8, zorder=5)
    ax.plot(*FOCSANI, marker="*", color="white", ms=15, zorder=6)
    ax.text(FOCSANI[0] - 0.04, FOCSANI[1] + 0.03, "Focșani", color="white", size=11,
            weight="bold", ha="right", zorder=6)

    empty = np.empty((0, 2))
    glow = ax.scatter(empty[:, 0], empty[:, 1], s=[], linewidths=0, zorder=2)
    core = ax.scatter(empty[:, 0], empty[:, 1], s=[], linewidths=0, zorder=3)
    ring = ax.scatter(empty[:, 0], empty[:, 1], s=[], facecolors="none", linewidths=1.3, zorder=4)

    txt_year = ax.text(0.97, 0.035, "", transform=ax.transAxes, ha="right", va="bottom",
                       color="white", size=44, weight="bold", alpha=0.92, zorder=7)
    txt_count = ax.text(0.97, 0.125, "", transform=ax.transAxes, ha="right", va="bottom",
                        color=PINK, size=12, zorder=7)
    txt_big = ax.text(0.03, 0.965, "", transform=ax.transAxes, ha="left", va="top",
                      color=PINK, size=15, weight="bold", zorder=7)

    # ---- depth cross-section (longitude vs depth)
    ax2 = fig.add_axes([0.12, 0.10, 0.82, 0.155], facecolor=BG)
    ax2.set_xlim(LON0, LON1)
    ax2.set_ylim(DEPTH_MAX, 0)
    ax2.set_xticks([])
    ax2.set_yticks([0, 50, 100, 150])
    ax2.set_yticklabels(["0 km", "50 km", "100 km", "150 km"])
    ax2.tick_params(colors=MUTED, labelsize=11, length=0)
    for y in (50, 100, 150):
        ax2.axhline(y, color=GRID, lw=0.6)
    ax2.axvline(FOCSANI[0], color=GRID, lw=1, ls="--")
    for sp in ax2.spines.values():
        sp.set_color(GRID)
    ax2.text(0.01, 0.06, "DEPTH PROFILE (longitude vs depth)", transform=ax2.transAxes,
             color=MUTED, size=10, va="bottom")
    glow2 = ax2.scatter(empty[:, 0], empty[:, 1], s=[], linewidths=0)
    core2 = ax2.scatter(empty[:, 0], empty[:, 1], s=[], linewidths=0)

    tmp = out_path.replace(".mp4", "_silent.mp4")
    writer = imageio.get_writer(tmp, fps=fps, codec="libx264", quality=8,
                                pixelformat="yuv420p", macro_block_size=1)

    for i in range(n_frames):
        t = i / fps
        k = int(np.searchsorted(vt, t, side="right"))
        age = t - vt[:k]

        # catalog clock
        cur = date_at(t, args, end_ts)
        txt_year.set_text(str(cur.year))
        txt_count.set_text(f"{k} quakes M≥{args.minmag:g}")

        if k:
            xy = np.c_[lon[:k], lat[:k]]
            xy2 = np.c_[lon[:k], dep[:k]]

            c = base[:k].copy()
            c[:, 3] = 0.22 + 0.78 * np.exp(-age / 1.8)
            g = base[:k].copy()
            g[:, 3] = 0.05 + 0.25 * np.exp(-age / 1.2)

            core.set_offsets(xy); core.set_sizes(s_core[:k]); core.set_facecolor(c)
            glow.set_offsets(xy); glow.set_sizes(s_core[:k] * 5); glow.set_facecolor(g)
            core2.set_offsets(xy2); core2.set_sizes(s_core[:k] * 0.8); core2.set_facecolor(c)
            glow2.set_offsets(xy2); glow2.set_sizes(s_core[:k] * 4); glow2.set_facecolor(g)

            live = np.where(age < life[:k])[0]
            if len(live):
                frac = age[live] / life[live]
                e = base[live].copy()
                e[:, 3] = (1 - frac) ** 1.5
                ring.set_offsets(xy[live])
                ring.set_sizes(s_core[live] * (1 + 45 * frac))
                ring.set_edgecolor(e)
            else:
                ring.set_offsets(empty); ring.set_sizes([])
        else:
            for a in (core, glow, ring, core2, glow2):
                a.set_offsets(empty); a.set_sizes([])

        # big-quake label + background flash
        flash = 0.0
        label = ""
        a_lab = 0.0
        for j in big:
            a = t - vt[j]
            if 0 <= a < 3.5:
                label = (f"{ev.time[j]:%d %b %Y}   M{mag[j]:.1f}   {dep[j]:.0f} km deep")
                a_lab = 1 - a / 3.5
                flash = max(flash, max(0.0, 1 - a / 0.8))
        txt_big.set_text(label)
        txt_big.set_alpha(a_lab if label else 0)
        bg = np.array(matplotlib.colors.to_rgb(BG))
        ax.set_facecolor(bg + (np.array(matplotlib.colors.to_rgb("#3a0f4a")) - bg) * flash)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame.copy())
        if i % 50 == 0:
            print(f"  frame {i}/{n_frames}")

    writer.close()
    plt.close(fig)
    return tmp


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="vrancea_quakes_1900.csv", help="cache file for the USGS catalog")
    ap.add_argument("--minmag", type=float, default=4.0)
    ap.add_argument("--seconds", type=float, default=40, help="length of the 1977->today timeline")
    ap.add_argument("--pre-seconds", dest="pre_seconds", type=float, default=12, help="length of the 1900-1976 intro")
    ap.add_argument("--premag", type=float, default=5.0, help="min magnitude before 1977 (older catalog is incomplete)")
    ap.add_argument("--lead", type=float, default=1.5, help="seconds before the timeline starts")
    ap.add_argument("--tail", type=float, default=3.0, help="seconds after the timeline ends")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--out", default="vrancea_hums.mp4")
    ap.add_argument("--demo", action="store_true", help="fake data, no internet needed")
    ap.add_argument("--max-frames", type=int, default=0, help="render only first N frames (testing)")
    args = ap.parse_args()

    ev = load_quakes(args)
    end_ts = pd.Timestamp.now(tz="UTC")
    vt = video_time(ev.time, args, end_ts)
    total_sec = args.lead + args.pre_seconds + args.seconds + args.tail

    print("Synthesizing audio ...")
    audio = build_audio(vt, ev.longitude.values, ev.depth.values, ev.mag.values, total_sec, args.minmag)
    wav_path = args.out.replace(".mp4", ".wav")
    wavfile.write(wav_path, SR, audio)

    print("Rendering frames ...")
    silent = render(args, ev, vt, total_sec, args.out, end_ts)

    print("Muxing audio + video ...")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, "-y", "-i", silent, "-i", wav_path, "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", args.out],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.remove(silent)
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()

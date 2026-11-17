"""
#30DayMapChallenge | Day 12: Power  (animation)
Energy packets travel along the 220/400 kV lines while the plants pulse.
Seamless loop. Uses the cached OSM files from day12_power_grid.py (no new Overpass calls).

Put this file next to day12_power_grid.py, then:
    python day12_power_animation.py              -> day12_power.mp4 (1080x1350)
    python day12_power_animation.py --gif        -> day12_power.gif instead
Needs: pip install imageio imageio-ffmpeg   (ffmpeg only for mp4)
"""
import argparse

import matplotlib

matplotlib.use("Agg")

import imageio.v2 as imageio
import numpy as np
import shapely

import day12_power_grid as base

FRAMES = 120          # loop length (120 frames @ 30 fps = 4 s)
FPS = 30
LOOP_KM = 70          # distance a packet travels per loop (sets the flow speed)
TRAIL = (0.0, 0.010, 0.020, 0.030)   # packet head + 3 trailing dots (fraction of line)
TRAIL_SIZE = (1.0, 0.7, 0.45, 0.25)

PACKET = {400: "#ffd9fb", 220: "#d2c8ff"}


def make_packets(ax, lines, seed=12):
    """One packet per line segment. Returns the data needed to move them each frame."""
    rng = np.random.default_rng(seed)
    groups = []
    for kv, sub in lines.groupby("kv"):
        geoms = np.array(sub.geometry.values)
        length_km = shapely.length(geoms) / 1000
        keep = length_km > 1.0
        geoms, length_km = geoms[keep], length_km[keep]
        # whole number of laps per loop -> seamless; longer lines move faster, roughly constant speed
        cycles = np.maximum(1, np.round(length_km / LOOP_KM)).astype(int)
        offset = rng.random(len(geoms))
        base_s = 26 if kv == 400 else 14
        arts = []
        for d, sz in zip(TRAIL, TRAIL_SIZE):
            glow = ax.scatter([], [], s=base_s * sz * 9, color=PACKET[kv], alpha=0.10,
                              linewidths=0, zorder=3.6)
            core = ax.scatter([], [], s=base_s * sz, color=PACKET[kv], alpha=0.95 * sz,
                              linewidths=0, zorder=3.7)
            arts.append((d, glow, core))
        groups.append({"geoms": geoms, "cycles": cycles, "offset": offset, "arts": arts})
    return groups


def update(groups, pulse, t):
    for g in groups:
        head = (g["offset"] + g["cycles"] * t) % 1.0
        for d, glow, core in g["arts"]:
            frac = np.clip(head - d, 0, 1)
            pts = shapely.line_interpolate_point(g["geoms"], frac, normalized=True)
            xy = shapely.get_coordinates(pts)
            glow.set_offsets(xy)
            core.set_offsets(xy)
    for coll, sizes, ph in pulse:
        coll.set_sizes(sizes * (1 + 0.35 * np.sin(2 * np.pi * (t + ph))))


def render(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", action="store_true", help="write a GIF instead of mp4")
    args = ap.parse_args()

    plants, lines, ro = base.load_plants(), base.load_lines(), base.load_boundary()
    fig, pulse = base.build(plants, lines, ro)
    fig.set_dpi(70 if args.gif else 100)
    ax = fig.axes[0]
    groups = make_packets(ax, lines)

    frames = []
    for k in range(FRAMES):
        update(groups, pulse, k / FRAMES)
        frames.append(render(fig))
        if k % 10 == 0:
            print(f"frame {k + 1}/{FRAMES}")

    if args.gif:
        imageio.mimsave("day12_power.gif", frames, duration=1 / FPS, loop=0)
        print("saved day12_power.gif")
    else:
        imageio.mimsave("day12_power.mp4", frames, fps=FPS, codec="libx264",
                        quality=8, macro_block_size=1)
        print("saved day12_power.mp4")

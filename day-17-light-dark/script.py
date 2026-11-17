"""
Day 17 | #30DayMapChallenge | Light & dark
The Sunrise Race: isochrones of sunrise across Romania, every day of 2026.
Map by Sara AHARON.

Inputs:
  Romania boundary GeoJSON (GADM 4.1 level 0 works: gadm41_ROU_0.json).
  Put it next to this script, or pass --boundary path. If missing, the script
  tries to download it from GADM.
Output:
  day17_sunrise_race.mp4  (1080x1350, 30 fps)

Run:   python day17_sunrise_race.py                 # full year video (365 frames + hold)
       python day17_sunrise_race.py still           # two preview PNGs (17 Nov, 21 Jun)
       python day17_sunrise_race.py --boundary C:\\path\\gadm41_ROU_0.json
Needs: pip install numpy matplotlib scipy imageio imageio-ffmpeg

Sunrise = NOAA solar-position approximation (sun's upper limb on a flat horizon, standard
refraction, no terrain), shown in Romanian civil time: EET (UTC+2), EEST (UTC+3) from 29 Mar to 24 Oct 2026.
"""
import os, sys, json, math, datetime as dt, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.ndimage import binary_dilation

YEAR = 2026
OUT = 'day17_sunrise_race.mp4'
FPS, HOLD = 30, 1.5
GADM_URL = 'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_ROU_0.json.zip'

BG = '#0b0b1a'; BOX = '#0e0e22'; GREY = '#8a93b0'
CYAN, PINK, YEL = '#00f5d4', '#ff2e97', '#ffd166'
OUTLINE = '#b8c0ff'
CMAP = LinearSegmentedColormap.from_list('sunrise', ['#ffe08a', '#f77f00', '#ff2e97', '#9b5de5', '#2a1b6e', '#14123a'])

CITIES = [('Sulina', 29.6667, 45.1583, (-0.20, 0.26), 'right'),
          ('Constanța', 28.6348, 44.1598, (-0.12, -0.28), 'right'),
          ('Iași', 27.5766, 47.1585, (-0.12, 0.17), 'right'),
          ('București', 26.1025, 44.4268, (0.0, -0.30), 'center'),
          ('Cluj-Napoca', 23.5914, 46.7712, (0.0, 0.17), 'center'),
          ('Timișoara', 21.2272, 45.7489, (0.0, -0.30), 'center')]
SUL, TIM = CITIES[0], CITIES[5]


# ---------------- boundary ----------------
def rdp(pts, eps):
    n = len(pts)
    keep = np.zeros(n, bool); keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1: continue
        a, b = pts[i], pts[j]
        ab = b - a; L = np.hypot(*ab)
        seg = pts[i + 1:j] - a
        d = np.hypot(seg[:, 0], seg[:, 1]) if L == 0 else np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return pts[keep]


def load_rings(path):
    if path is None:
        for cand in ('gadm41_ROU_0.json', 'gadm41_ROU_0.geojson', 'romania.geojson', 'ROU.geojson'):
            if os.path.exists(cand): path = cand; break
    if path is None:
        import urllib.request, zipfile
        print('downloading boundary from GADM ...')
        urllib.request.urlretrieve(GADM_URL, 'gadm41_ROU_0.json.zip')
        with zipfile.ZipFile('gadm41_ROU_0.json.zip') as z:
            z.extract('gadm41_ROU_0.json'); path = 'gadm41_ROU_0.json'
    gj = json.load(open(path, encoding='utf-8'))
    geoms = [f['geometry'] for f in gj['features']] if gj['type'] == 'FeatureCollection' else [gj.get('geometry', gj)]
    rings = []
    for g in geoms:
        polys = [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
        for poly in polys:
            r = np.array(poly[0])[:, :2]
            if np.ptp(r[:, 0]) > 0.03 or np.ptp(r[:, 1]) > 0.03:
                rings.append(rdp(r, 0.002))
    return rings


# ---------------- sunrise maths ----------------
def last_sunday(y, m):
    d = dt.date(y, m + 1, 1) - dt.timedelta(days=1) if m < 12 else dt.date(y, 12, 31)
    return d - dt.timedelta(days=(d.weekday() + 1) % 7)

DST_ON, DST_OFF = last_sunday(YEAR, 3), last_sunday(YEAR, 10)
DATES = [dt.date(YEAR, 1, 1) + dt.timedelta(days=i) for i in range(365)]

def tz_of(d): return 3 if DST_ON <= d < DST_OFF else 2

def sunrise_min(doy, lon, lat, tz):
    """Sunrise in minutes after local midnight (NOAA approximation)."""
    g = 2 * np.pi / 365 * (doy - 1)
    eq = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    de = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
          + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    la = np.radians(lat)
    c = np.cos(np.radians(90.833)) / (np.cos(la) * np.cos(de)) - np.tan(la) * np.tan(de)
    H = np.degrees(np.arccos(np.clip(c, -1, 1)))
    return 720 - 4 * (lon + H) - eq + tz * 60

def hm(m):
    m = int(round(m)); return f'{m // 60:02d}:{m % 60:02d}'


# ---------------- setup ----------------
ap = argparse.ArgumentParser()
ap.add_argument('mode', nargs='?', default='video')
ap.add_argument('--boundary', default=None)
args = ap.parse_args()

rings = load_rings(args.boundary)
allp = np.vstack(rings)
LON0, LON1 = allp[:, 0].min() - 0.15, allp[:, 0].max() + 0.15
LAT0, LAT1 = allp[:, 1].min() - 0.15, allp[:, 1].max() + 0.15
LAT_MID = (LAT0 + LAT1) / 2

gx = np.arange(LON0, LON1, 0.04); gy = np.arange(LAT0, LAT1, 0.04)
GX, GY = np.meshgrid(gx, gy)
pts = np.c_[GX.ravel(), GY.ravel()]
inside = np.zeros(GX.shape, bool)
for r in rings:
    inside |= Path(r).contains_points(pts).reshape(GX.shape)
near = binary_dilation(inside, iterations=2)

# global colour scale: widest east-west gap between first and last sunrise inside the country
span = max(np.ptp(sunrise_min(i + 1, GX, GY, tz_of(DATES[i]))[inside]) for i in range(0, 365, 3))
STEP = 3
VMAX = math.ceil(span / STEP) * STEP
LEVELS = np.arange(0, VMAX + STEP, STEP)

# year curves for the chart
doys = np.arange(1, 366)
tzs = np.array([tz_of(d) for d in DATES])
c_sul = np.array([sunrise_min(i, SUL[1], SUL[2], t) for i, t in zip(doys, tzs)])
c_tim = np.array([sunrise_min(i, TIM[1], TIM[2], t) for i, t in zip(doys, tzs)])

# ---------------- figure ----------------
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
canvas = FigureCanvasAgg(fig)

ax = fig.add_axes([0.06, 0.395, 0.88, 0.465], facecolor=BOX)
ax.set_xlim(LON0, LON1); ax.set_ylim(LAT0, LAT1)
ax.set_aspect(1 / np.cos(np.radians(LAT_MID)), adjustable='box')
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values(): sp.set_color('#2a2a4a'); sp.set_linewidth(1)
for x in range(int(math.ceil(LON0)), int(LON1) + 1): ax.axvline(x, color='#171738', lw=0.6, zorder=0)
for y in range(int(math.ceil(LAT0)), int(LAT1) + 1): ax.axhline(y, color='#171738', lw=0.6, zorder=0)

codes, verts = [], []
for r in rings:
    verts += list(r) + [r[0]]
    codes += [Path.MOVETO] + [Path.LINETO] * (len(r) - 1) + [Path.CLOSEPOLY]
CLIP = PathPatch(Path(verts, codes), transform=ax.transData, fc='none', ec='none')
for r in rings:
    for lw, al in ((9, 0.05), (5, 0.10), (1.6, 0.95)):
        ax.plot(r[:, 0], r[:, 1], color=OUTLINE, lw=lw, alpha=al, zorder=6, solid_joinstyle='round')

for name, lo, la, (dx, dy), ha in CITIES:
    hi = name in ('Sulina', 'Timișoara')
    col = CYAN if name == 'Sulina' else PINK if name == 'Timișoara' else 'white'
    ax.plot(lo, la, 'o', color=col, ms=8 if hi else 4.5, mec='#0b0b1a', mew=1.5, zorder=8)
    ax.text(lo + dx, la + dy, name, color='white' if not hi else col, fontsize=10 if hi else 8.5,
            fontweight='bold' if hi else 'normal', ha=ha, va='center', zorder=8)

fig.text(0.06, 0.93, 'The Sunrise Race', color='white', fontsize=26, fontweight='bold')
fig.text(0.06, 0.895, f'Where the sun rises first in Romania, every day of {YEAR}', color=CYAN, fontsize=14)

# colour legend
cax = fig.add_axes([0.06, 0.368, 0.25, 0.007]); cax.imshow(np.linspace(0, 1, 256)[None, :], aspect='auto', cmap=CMAP)
cax.axis('off')
fig.text(0.06, 0.352, 'first light', color=GREY, fontsize=8)
fig.text(0.31, 0.352, f'+{VMAX} min', color=GREY, fontsize=8, ha='right')
fig.text(0.34, 0.366, 'isolines every 5 min, local time', color=GREY, fontsize=8, va='center')

# left panel
fig.text(0.06, 0.322, 'DATE', color=GREY, fontsize=10, fontweight='bold')
t_date = fig.text(0.06, 0.282, '', color='white', fontsize=34, fontweight='bold')
t_tz = fig.text(0.06, 0.255, '', color=PINK, fontsize=11)
fig.text(0.06, 0.215, 'THE RACE', color=GREY, fontsize=10, fontweight='bold')
rows = []
for i, (nm, col) in enumerate((('Sulina', CYAN), ('Timișoara', PINK))):
    y = 0.185 - i * 0.032
    fig.text(0.06, y, '●', color=col, fontsize=12, va='center')
    fig.text(0.085, y, nm, color='white', fontsize=13, va='center')
    rows.append(fig.text(0.30, y, '', color=col, fontsize=17, fontweight='bold', va='center'))
t_lead = fig.text(0.06, 0.118, '', color=YEL, fontsize=11)

# chart
fig.text(0.52, 0.322, 'SUNRISE THROUGH THE YEAR', color=GREY, fontsize=10, fontweight='bold')
cx = fig.add_axes([0.52, 0.085, 0.42, 0.215], facecolor=BOX)
for sp in cx.spines.values(): sp.set_color('#2a2a4a')
def broken(c):
    y = c.astype(float).copy(); j = np.where(np.abs(np.diff(y)) > 30)[0]; y[j] = np.nan; return y
cx.plot(doys, broken(c_tim), color=PINK, lw=1.8); cx.plot(doys, broken(c_sul), color=CYAN, lw=1.8)
cx.set_xlim(1, 365)
cx.set_ylim(290, 485); cx.set_yticks(range(300, 481, 60)); cx.set_yticklabels([hm(m) for m in range(300, 481, 60)], color=GREY, fontsize=8)
mstart = [dt.date(YEAR, m, 1).timetuple().tm_yday for m in range(1, 13)]
cx.set_xticks(mstart); cx.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], color=GREY, fontsize=8)
cx.tick_params(length=2, color='#2a2a4a'); cx.grid(color='#171738', lw=0.6)
d17 = dt.date(YEAR, 11, 17).timetuple().tm_yday
cx.axvline(d17, color=YEL, lw=0.8, ls=':', alpha=0.6)
cx.text(d17 - 3, 294, '17 Nov', color=YEL, fontsize=7.5, ha='right', va='bottom', alpha=0.8)
cur = cx.axvline(1, color='white', lw=1.2, alpha=0.8)
dots = [cx.plot([1], [0], 'o', color=c, ms=7, mec=BG, mew=1.2, zorder=5)[0] for c in (CYAN, PINK)]

# credits, bottom-left corner
fig.text(0.06, 0.045, '#30DayMapChallenge | Day 17: Light & dark | Map: Sara AHARON', color=GREY, fontsize=8)
fig.text(0.06, 0.030, 'Data: sunrise computed with the NOAA solar-position algorithm (upper limb, standard refraction, no terrain) | '
         'Boundary: GADM 4.1 | Time: EET/EEST', color=GREY, fontsize=7)

dyn = []
def clear():
    while dyn:
        a = dyn.pop()
        try: a.remove()
        except Exception: pass

def clip(cs):
    try: cs.set_clip_path(CLIP)
    except Exception:
        for c in cs.collections: c.set_clip_path(CLIP)

def frame(i):
    clear()
    d = DATES[i]; tz = tz_of(d); doy = i + 1
    T = sunrise_min(doy, GX, GY, tz)
    rel = T - T[inside].min()
    cs = ax.contourf(GX, GY, rel, levels=LEVELS, cmap=CMAP, extend='both', zorder=2); clip(cs); dyn.append(cs)
    Tm = np.ma.masked_where(~near, T)
    lo, hi = T[inside].min(), T[inside].max()
    l5 = np.arange(math.ceil(lo / 5) * 5, hi + 1, 5)
    l10 = [m for m in l5 if m % 10 == 0]
    c5 = ax.contour(GX, GY, Tm, levels=l5, colors='white', linewidths=0.6, alpha=0.45, zorder=4); clip(c5); dyn.append(c5)
    if len(l10):
        c10 = ax.contour(GX, GY, Tm, levels=l10, colors='white', linewidths=1.1, alpha=0.9, zorder=5); clip(c10)
        lbl = ax.clabel(c10, fmt=lambda m: hm(m), fontsize=8, colors='white', inline=True, inline_spacing=6)
        dyn.append(c10)
    t_date.set_text(f'{d.day} {d.strftime("%b").upper()}')
    t_tz.set_text(f'day {doy} of 365  ·  {"EEST (UTC+3)" if tz == 3 else "EET (UTC+2)"}')
    ts, tm_ = c_sul[i], c_tim[i]
    rows[0].set_text(hm(ts)); rows[1].set_text(hm(tm_))
    t_lead.set_text(f'Sulina sees the sun {int(round(tm_ - ts))} min before Timișoara')
    cur.set_xdata([doy, doy])
    dots[0].set_data([doy], [ts]); dots[1].set_data([doy], [tm_])
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

if __name__ == '__main__':
    if args.mode == 'still':
        for name, d in (('nov17', dt.date(YEAR, 11, 17)), ('jun21', dt.date(YEAR, 6, 21))):
            plt.imsave(f'preview_{name}.png', frame((d - dt.date(YEAR, 1, 1)).days))
        print('VMAX', VMAX, 'rings', len(rings), 'pts', sum(len(r) for r in rings))
    else:
        import imageio
        w = imageio.get_writer(OUT, fps=FPS, codec='libx264', macro_block_size=None,
                               pixelformat='yuv420p', ffmpeg_params=['-crf', '18'])
        for i in range(365):
            w.append_data(frame(i))
            if i % 30 == 0: print(i, 365, flush=True)
        last = frame(364)
        for _ in range(int(HOLD * FPS)): w.append_data(last)
        w.close(); print('DONE', OUT, flush=True)

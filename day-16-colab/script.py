"""
Day 16 | #30DayMapChallenge | Collaborative map
Focsani, Tree by Tree: animated 3x3 grid: all seven parks (7th under the 6th).
Tree survey by David TICU. Map by Sara AHARON.

Inputs (put next to this script):
  trees_fcs.geojson   David's tree points (Name = "Romanian name (Latin name)")
  roads.geojson       OSM roads (optional; drawn faintly if present)
  VN.geojson          Vrancea UAT boundaries (auto-downloaded if missing; optional)
Output:
  day16_focsani_trees_grid_roads.mp4  (1080x1350, 30 fps)

Run:   python day16_focsani_trees.py          # full video (~3 min)
       python day16_focsani_trees.py still    # two preview PNGs
Needs: pip install numpy matplotlib scikit-learn scipy imageio imageio-ffmpeg
"""
TREES = 'trees_fcs.geojson'
ROADS = 'roads.geojson'
BOUNDARY = 'VN.geojson'
BOUNDARY_URL = 'https://raw.githubusercontent.com/akaleeroy/romania-uat/master/geojson/VN.geojson'
OUT = 'day16_focsani_trees_grid_roads.mp4'
import json, collections, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgb
from sklearn.cluster import DBSCAN
import os


def load_boundary():
    """Focsani municipality outline for the locator (skipped if unavailable)."""
    try:
        if not os.path.exists(BOUNDARY):
            import urllib.request
            urllib.request.urlretrieve(BOUNDARY_URL, BOUNDARY)
        vn = json.load(open(BOUNDARY))
        return np.array([f for f in vn['features'] if f['properties']['name'] == 'Focșani'][0]['geometry']['coordinates'][0])[:, :2]
    except Exception as e:
        print('boundary unavailable, skipping locator:', e)
        return None

EN = {
'Thuja plicata':'Western red cedar','Acer platanoides':'Norway maple','Aesculus hippocastanum':'Horse chestnut',
'Tilia cordata':'Small-leaved lime','Fraxinus excelsior':'European ash','Pinus sylvestris':'Scots pine',
'Betula pendula':'Silver birch','Carpinus betulus':'European hornbeam','Morus':'Mulberry',
'Prunus cerasifera „Nigra”':'Purple-leaf plum','Prunus serrulata':'Japanese flowering cherry',
'Catalpa bignonioides':'Southern catalpa','Salix alba':'White willow','Ligustrum vulgare':'Common privet',
'Syringa vulgaris':'Common lilac','Tamarix ramosissima':'Tamarisk','rosa L.':'Rose','Robinia pseudoacacia':'Black locust',
'Rhus typhina':'Staghorn sumac','Buxus sempervirens':'Common box','Cornus mas':'Cornelian cherry',
'Yucca filamentosa':"Adam's needle yucca",'Hibiscus syriacus':'Rose of Sharon','Koelreuteria paniculata':'Golden rain tree',
'Juniperus pfitzeriana':'Pfitzer juniper','Quercus robur':'English oak','Magnolia soulangeana':'Saucer magnolia',
'Buddleja davidii':'Butterfly bush','Viburnum opulus':'Guelder rose','Malus sylvestris':'Crab apple',
'Berberis vulgaris':'Common barberry','Picea pungens':'Blue spruce','Prunus avium':'Wild cherry',
'Forsythia suspensa':'Weeping forsythia','Cortaderia selloana':'Pampas grass','Lavandula angustifolia':'English lavender',
'Juglans regia':'Common walnut','Cercis siliquastrum':'Judas tree','Abies alba':'Silver fir',
'Liquidambar styraciflua':'Sweetgum','Lonicera caprifolium':'Italian honeysuckle','Pyracantha coccinea':'Scarlet firethorn',
'Cornus alba':'White dogwood','Photinia × fraseri':'Red robin','Populus nigra':'Black poplar',
'Hydrangea paniculata':'Panicle hydrangea','Hedera helix':'Common ivy','Sambucus nigra':'Elder',
'Gleditsia triacanthos':'Honey locust','Ailanthus altissima':'Tree of heaven','Crataegus monogyna':'Common hawthorn',
'Prunus laurocerasus':'Cherry laurel','Albizia julibrissin':'Persian silk tree','Hosta plantaginea':'Fragrant hosta',
'Cornus sanguinea':'Common dogwood','Populus tremula':'Aspen','Prunus cerasifera':'Cherry plum','Picea abies':'Norway spruce',
'Cotinus coggygria':'Smoke bush','Prunus domestica':'European plum','Larix decidua':'European larch',
'Pyrus calleryana':'Callery pear','Elaeagnus angustifolia':'Russian olive','Ginkgo biloba':'Ginkgo',
'Taxus baccata':'English yew','Wisteria sinensis':'Chinese wisteria','Cydonia oblonga':'Quince'}

d = json.load(open(TREES))
F = [x for x in d['features'] if x['geometry']]
ro_all, la_all = [], []
for x in F:
    m = re.match(r'\s*(.*?)\s*\(\s*(.*?)\s*\)\s*$', x['properties']['Name'])
    ro_all.append(m.group(1)); la_all.append(m.group(2))
ro_all = np.array(ro_all); la_all = np.array(la_all)
ll = np.array([x['geometry']['coordinates'][:2] for x in F])
lat0 = ll[:, 1].mean()
LON0, LAT0 = ll[:, 0].min(), ll[:, 1].min()
PX = (ll[:, 0] - ll[:, 0].min()) * np.cos(np.radians(lat0)) * 111320
PY = (ll[:, 1] - ll[:, 1].min()) * 110570
P = np.c_[PX, PY]
lab = DBSCAN(eps=80, min_samples=3).fit_predict(P)
ks = sorted([(int((lab == k).sum()), k) for k in set(lab) if k >= 0], reverse=True)[:7]
parks = [k for _, k in ks]
keep = np.isin(lab, parks)
P = P[keep]; ll = ll[keep]; ro = ro_all[keep]; la = la_all[keep]
park_of = np.array([parks.index(k) for k in lab[keep]])
N = len(P)
cnt = collections.Counter(la)
order = [k for k, _ in cnt.most_common()]
NS = len(cnt)

NAMES = ['Nicolae Bălcescu Park', 'Public Garden & Schuman Park', 'Theatre Park', "Heroes' Mausoleum", 'Black Park', 'Velo Park', 'Bahne Park']
NEON = ['#00f5d4', '#ff2e97', '#ffd166', '#9b5de5', '#4cc9f0', '#f77f00', '#80ed99', '#ff6b6b']
FPS = 30; T0, HOLD = 1.5, 3.5
start, dur = {}, {}
t = T0
for r, s in enumerate(order):
    dur[s] = 0.9 if r < 8 else (0.4 if r < 25 else 0.2)
    start[s] = t; t += dur[s]
T_FIN = t + 0.4
T_END = t + HOLD
rng = np.random.default_rng(16)
rev = np.zeros(N); col = np.zeros((N, 3))
for r, s in enumerate(order):
    m = np.where(la == s)[0]
    perm = rng.permutation(len(m))
    rev[m] = start[s] + perm / len(m) * dur[s] * 0.75
    col[m] = to_rgb(NEON[r % 8])

BG = '#0b0b1a'; BOX = '#0e0e22'
fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=BG)
canvas = FigureCanvasAgg(fig)

cw = 0.25; ch = cw * 1080 / 1350
xs = [0.06 + i * (cw + 0.045) for i in range(3)]
ys = [0.838 - ch - r * 0.256 for r in range(3)]
boxes = []
for k in range(7):
    r, c = divmod(k, 3)
    if k == 6: r, c = 2, 2   # Bahne Park sits under park 6; bottom-left holds the stats panel
    ax = fig.add_axes([xs[c], ys[r], cw, ch], facecolor=BOX)
    idx = np.where(park_of == k)[0]
    q = P[idx]; mid = (q.max(0) + q.min(0)) / 2; ext = (q.max(0) - q.min(0)).max()
    half = ext / 2 * 1.2
    ax.set_xlim(mid[0] - half, mid[0] + half); ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color('#2a2a4a'); sp.set_linewidth(1)
    base = 12 * (240 / max(ext, 90)) ** 0.5
    h = ax.scatter(q[:, 0], q[:, 1], s=0, lw=0, zorder=3)
    cr = ax.scatter(q[:, 0], q[:, 1], s=0, lw=0, zorder=4)
    L = 50 if ext > 150 else 25
    bx = mid[0] - half * 0.88; by = mid[1] - half * 0.86
    ax.plot([bx, bx + L], [by, by], color='white', lw=1.6, zorder=5)
    ax.text(bx + L / 2, by + half * 0.05, f'{L} m', color='white', ha='center', fontsize=7.5, zorder=5)
    nsp = len(set(la[idx]))
    fig.text(xs[c], ys[r] + ch + 0.024, f'{k + 1}', color='#ff2e97', fontsize=12, fontweight='bold')
    fig.text(xs[c] + 0.022, ys[r] + ch + 0.024, NAMES[k], color='white', fontsize=10.5, fontweight='bold')
    fig.text(xs[c] + 0.022, ys[r] + ch + 0.008, f'{len(idx)} trees · {nsp} species', color='#8a93b0', fontsize=9)
    boxes.append((idx, h, cr, base))


from matplotlib.collections import LineCollection
HAS_ROADS = os.path.exists(ROADS)
if HAS_ROADS:
    rd = json.load(open(ROADS))
    MAJOR = {'primary', 'primary_link', 'secondary', 'secondary_link', 'tertiary'}
    MINOR = {'footway', 'track', 'path', 'pedestrian', 'steps', 'cycleway'}
    groups = {'major': [], 'street': [], 'minor': []}
    for f_ in rd['features']:
        g = f_.get('geometry')
        if not g: continue
        hw = f_['properties'].get('highway')
        k_ = 'major' if hw in MAJOR else ('minor' if hw in MINOR else 'street')
        lines = [g['coordinates']] if g['type'] == 'LineString' else (g['coordinates'] if g['type'] == 'MultiLineString' else [])
        for ln in lines:
            a_ = np.array(ln)[:, :2]
            groups[k_].append(np.c_[(a_[:, 0] - LON0) * np.cos(np.radians(lat0)) * 111320, (a_[:, 1] - LAT0) * 110570])
    style = {'minor': ('#232850', 0.6), 'street': ('#2e3463', 1.0), 'major': ('#4a5190', 1.8)}
    for idx, h, cr, base in boxes:
        ax_ = h.axes
        xl, yl = ax_.get_xlim(), ax_.get_ylim()
        for k_ in ('minor', 'street', 'major'):
            ax_.add_collection(LineCollection(groups[k_], colors=style[k_][0], linewidths=style[k_][1], zorder=1))
        ax_.set_xlim(xl); ax_.set_ylim(yl)

# locator
poly = load_boundary()
if poly is not None:
    ax2 = fig.add_axes([0.06, 0.118, 0.17, 0.100], facecolor=BG)
    ax2.fill(poly[:, 0], poly[:, 1], color='#1a1a38', zorder=1)
    ax2.plot(poly[:, 0], poly[:, 1], color='#4a4f7a', lw=0.9, zorder=2)
    for idx, *_ in boxes:
        q = ll[idx]
        ax2.plot(q[:, 0].mean(), q[:, 1].mean(), 's', color='#ff2e97', ms=3.5, zorder=3)
    ax2.set_aspect(1 / np.cos(np.radians(lat0))); ax2.axis('off')
    fig.text(0.06, 0.226, 'FOCȘANI', color='#8a93b0', fontsize=9, fontweight='bold')
    fig.text(0.06, 0.099, 'Pink = the 7 parks', color='#ff2e97', fontsize=8.5)

fig.text(0.06, 0.93, 'Focșani, Tree by Tree', color='white', fontsize=26, fontweight='bold')
fig.text(0.06, 0.895, 'Seven parks, tracked by hand by David TICU', color='#00f5d4', fontsize=14)
fig.text(0.06, 0.055, '#30DayMapChallenge | Day 16: Collaborative map | Sara AHARON × David TICU', color='#8a93b0', fontsize=8)
fig.text(0.06, 0.04, 'Data: field survey by David TICU, parks of Focșani | Boundary: geo-spatial.org' + (' | Roads: © OpenStreetMap contributors' if HAS_ROADS else '') + ' | Map: Sara AHARON', color='#8a93b0', fontsize=8)
if HAS_ROADS:
    fig.add_artist(plt.Line2D([0.06, 0.09], [0.080, 0.080], color='#5a6099', lw=1.4))
    fig.text(0.10, 0.080, 'Roads', color='#8a93b0', fontsize=9, va='center')

fig.text(0.06, 0.345, 'TREES MAPPED', color='#8a93b0', fontsize=10, fontweight='bold')
t_total = fig.text(0.06, 0.303, '0', color='white', fontsize=34, fontweight='bold')
t_sp = fig.text(0.06, 0.274, '', color='#ff2e97', fontsize=13)
cx = 0.31
t_lab = fig.text(cx, 0.345, 'NOW GROWING', color='#8a93b0', fontsize=10, fontweight='bold')
t_en = fig.text(cx, 0.308, '', color='white', fontsize=21, fontweight='bold')
t_la = fig.text(cx, 0.281, '', color='#00f5d4', fontsize=11, style='italic')
t_ro = fig.text(cx, 0.259, '', color='#8a93b0', fontsize=10)
t_n = fig.text(cx, 0.234, '', color='white', fontsize=12)
ex = cx
fig.text(ex, 0.200, 'EARLIER', color='#8a93b0', fontsize=10, fontweight='bold')
tick = []
for i in range(4):
    y = 0.172 - i * 0.027
    tick.append((fig.text(ex, y, '●', fontsize=11, va='center'), fig.text(ex + 0.025, y, '', fontsize=10, va='center')))

def frame(t):
    age = t - rev
    vis = age >= 0
    a = np.where(vis, age, 99)
    glow = np.exp(-a / 0.9)
    white = 0.65 * np.exp(-a / 0.12)
    c = col * (1 - white[:, None]) + white[:, None]
    pop = 1 + 1.6 * np.exp(-a / 0.18)
    ac = np.where(vis, 0.5 + 0.5 * glow, 0)
    ah = np.where(vis, 0.10 + 0.22 * glow, 0)
    for idx, h, cr, base in boxes:
        cr.set_facecolor(np.c_[c[idx], ac[idx]]); cr.set_sizes(base * pop[idx])
        h.set_facecolor(np.c_[col[idx], ah[idx]]); h.set_sizes(base * pop[idx] * 6)
    t_total.set_text(f'{int(vis.sum()):,}')
    done = [s for s in order if t >= start[s]]
    t_sp.set_text(f'{len(done)} / {NS} species' if done else f'{NS} species')
    t_lab.set_text('NOW GROWING' if t < T_FIN else 'IN TOTAL')
    if t >= T_FIN:
        t_en.set_text('Focșani parks'); t_en.set_fontsize(21); t_en.set_color('white')
        t_la.set_text(f'{N:,} trees · {NS} species'); t_ro.set_text('Parks of Focșani'); t_n.set_text('')
    elif done:
        s = done[-1]; r = order.index(s)
        en = EN[s]
        t_en.set_text(en); t_en.set_fontsize(21 if len(en) <= 17 else 16); t_en.set_color(NEON[r % 8])
        t_la.set_text(s); t_ro.set_text(ro[np.where(la == s)[0][0]])
        t_n.set_text(f'{cnt[s]} tree' + ('s' if cnt[s] > 1 else ''))
    else:
        for tt in (t_en, t_la, t_ro, t_n): tt.set_text('')
    prev = done[:-1][::-1][:4]
    for i, (dot, txt) in enumerate(tick):
        if i < len(prev):
            s = prev[i]; r = order.index(s); al = 1 - i * 0.14
            dot.set_color(NEON[r % 8]); dot.set_alpha(al)
            txt.set_text(f'{EN[s]}  ·  {cnt[s]}'); txt.set_color('white'); txt.set_alpha(al)
        else:
            dot.set_alpha(0); txt.set_text('')
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()

if __name__ == '__main__':
    import imageio
    if len(sys.argv) > 1 and sys.argv[1] == 'still':
        for name, tt in (('mid', start[order[10]] + 0.3), ('end', T_END - 0.1)):
            imageio.imwrite(f'preview_{name}.png', frame(tt))
        print('N', N, 'NS', NS, 'T_END', T_END, [len(b[0]) for b in boxes])
    else:
        w = imageio.get_writer(OUT, fps=FPS, codec='libx264',
                               macro_block_size=None, pixelformat='yuv420p', ffmpeg_params=['-crf', '18'])
        nf = int(T_END * FPS)
        for i in range(nf):
            w.append_data(frame(i / FPS))
            if i % 60 == 0: print(i, nf, flush=True)
        w.close(); print('DONE', flush=True)

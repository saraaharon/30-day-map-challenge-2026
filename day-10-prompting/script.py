
# Day 10 - #30DayMapChallenge: Prompting only
# "Everything Is Close to Home" - Romania by straight-line distance from Focsani (animated)
# pip install geopandas matplotlib scipy imageio imageio-ffmpeg
import os, urllib.request, geopandas as gpd

URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces.geojson"
if not os.path.exists("ro.geojson"):
    urllib.request.urlretrieve(URL, "ne_admin1.geojson")          # ~40 MB, one-time download
    adm = gpd.read_file("ne_admin1.geojson")
    adm[adm.iso_a2 == "RO"].to_file("ro.geojson", driver="GeoJSON")  # Romanian counties only

OUT = "day10_prompting_only.mp4"
import numpy as np, geopandas as gpd, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from shapely.geometry import Point
import matplotlib.patheffects as pe

BG="#0b0b1f"
g=gpd.read_file("ro.geojson").to_crs(3844)
ro=g.union_all()
foc=gpd.GeoSeries([Point(27.1833,45.6967)],crs=4326).to_crs(3844).iloc[0]
cities={"Focșani":(27.1833,45.6967),"București":(26.1025,44.4268),"Iași":(27.5911,47.1585),
        "Cluj-Napoca":(23.5890,46.7712),"Timișoara":(21.2087,45.7489),"Constanța":(28.6348,44.1598)}
cp={k:gpd.GeoSeries([Point(v)],crs=4326).to_crs(3844).iloc[0] for k,v in cities.items()}

x0,y0,x1,y1=ro.bounds; pad=40000
xs=np.linspace(x0-pad,x1+pad,900); ys=np.linspace(y0-pad,y1+pad,700)
X,Y=np.meshgrid(xs,ys); D=np.hypot(X-foc.x,Y-foc.y)/1000

cmap=LinearSegmentedColormap.from_list("syn",["#ffe66d","#ff9e3d","#ff2e93","#a12bff","#3a2bff","#12123a"])
fig=plt.figure(figsize=(8,10),dpi=135,facecolor=BG)
ax=fig.add_axes([0.02,0.22,0.96,0.64]); ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")

RGB=cmap(np.clip(D/520,0,1))
im=ax.imshow(RGB,extent=[xs[0],xs[-1],ys[0],ys[-1]],origin="lower",zorder=2)

verts=[];codes=[]
for poly in (ro.geoms if hasattr(ro,"geoms") else [ro]):
    for ring in [poly.exterior,*poly.interiors]:
        c=np.array(ring.coords); verts+=list(c); codes+=[Path.MOVETO]+[Path.LINETO]*(len(c)-2)+[Path.CLOSEPOLY]
def clipp(): return PathPatch(Path(verts,codes),transform=ax.transData)
im.set_clip_path(clipp())
gpd.GeoSeries([ro],crs=3844).boundary.plot(ax=ax,color="#00e5ff",linewidth=2.4,alpha=.9,zorder=5)
g.boundary.plot(ax=ax,color=BG,linewidth=.5,alpha=.55,zorder=3)
ax.set_xlim(xs[0],xs[-1]); ax.set_ylim(ys[0],ys[-1]); ax.set_anchor("C")
ax.set_position([0.02,0.22,0.96,0.64])

# static rings + labels (shown as the wave passes)
ring_art={}
for r in range(100,600,100):
    c=Circle((foc.x,foc.y),r*1000,fill=False,ec="white",lw=.9,ls=(0,(2,4)),alpha=.65,zorder=4); ax.add_patch(c); c.set_clip_path(clipp())
    t=ax.text(foc.x+r*1000*np.cos(np.radians(172)),foc.y+r*1000*np.sin(np.radians(172)),f"{r} km",color="white",fontsize=7,ha="center",va="center",zorder=6,bbox=dict(fc=BG,ec="none",pad=1.2,alpha=.7))
    ring_art[r]=(c,t)
# wavefront
wf=Circle((foc.x,foc.y),1,fill=False,ec="white",lw=2.2,alpha=.95,zorder=6); ax.add_patch(wf); wf.set_clip_path(clipp())
wg=Circle((foc.x,foc.y),1,fill=False,ec="#ffe66d",lw=7,alpha=.25,zorder=5); ax.add_patch(wg); wg.set_clip_path(clipp())

off={"București":(-30000,-20000,"right"),"Iași":(-25000,15000,"right"),"Cluj-Napoca":(0,30000,"center"),
     "Timișoara":(0,-38000,"center"),"Constanța":(-20000,-32000,"right")}
city_art={}
for k,p in cp.items():
    if k=="Focșani": continue
    km=np.hypot(p.x-foc.x,p.y-foc.y)/1000
    m=ax.scatter(p.x,p.y,s=18,c="white",zorder=7,edgecolor=BG,lw=.6)
    dx,dy,ha=off[k]
    t=ax.text(p.x+dx,p.y+dy,f"{k} · {km:.0f} km",color="white",fontsize=7.5,ha=ha,va="center",zorder=8,path_effects=[pe.withStroke(linewidth=2.5,foreground=BG)])
    city_art[k]=(m,t,km)
pul=[ax.scatter(foc.x,foc.y,s=s,c="#ffe66d",alpha=a,zorder=8,lw=0) for s,a in [(700,.12),(380,.22),(170,.4)]]
ax.scatter(foc.x,foc.y,s=45,c="white",edgecolor="#ff2e93",lw=1.5,zorder=9)
ax.text(foc.x-18000,foc.y+30000,"FOCȘANI",color="white",fontsize=10,fontweight="bold",ha="right",zorder=9,path_effects=[pe.withStroke(linewidth=3,foreground=BG)])

cax=fig.add_axes([0.06,0.075,0.32,0.012])
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,520)); cb=fig.colorbar(sm,cax=cax,orientation="horizontal")
cb.set_ticks([0,100,200,300,400,500]); cb.outline.set_visible(False); cb.ax.tick_params(colors="#8a93b0",labelsize=7,length=0)
fig.text(0.06,0.098,"Straight-line distance from Focșani (km)",color="#8a93b0",fontsize=7)
fact1=fig.text(0.06,0.20,"Iași and București are both 165 km from Focșani.",color="white",fontsize=11,fontweight="bold")
fact2=fig.text(0.06,0.178,"Timișoara, at 465 km, is the farthest of the six cities marked.",color="#8a93b0",fontsize=9)
counter=fig.text(0.94,0.93,"",color="#ffe66d",fontsize=26,fontweight="bold",ha="right")
fig.text(0.94,0.895,"from Focșani",color="#ff2e93",fontsize=12,ha="right")
fig.text(0.06,0.93,"Everything Is Close to Home",color="white",fontsize=26,fontweight="bold")
fig.text(0.06,0.895,"Romania, measured in kilometres from Focșani",color="#ff2e93",fontsize=14)
fig.text(0.06,0.04,"#30DayMapChallenge | Day 10: Prompting only | Sara AHARON",color="#8a93b0",fontsize=8)
fig.text(0.06,0.022,"Data: Natural Earth admin-1 boundaries · Built end-to-end by prompting Claude (no manual coding)",color="#8a93b0",fontsize=8)

import imageio.v2 as imageio
N_GROW,N_HOLD=110,40
frames=[]
def ease(u): return u*u*(3-2*u)
for i in range(N_GROW+N_HOLD):
    u=min(i/(N_GROW-1),1); r=ease(u)*540
    soft=25
    a=np.clip((r-D)/soft,0,1)
    RGBA=RGB.copy(); RGBA[...,3]=a; im.set_data(RGBA)
    wf.set_radius(max(r*1000,1)); wg.set_radius(max(r*1000,1))
    vis=u<1; wf.set_visible(vis); wg.set_visible(vis)
    for R,(c,t) in ring_art.items():
        on=r>=R; c.set_visible(on); t.set_visible(on)
    for k,(m,t,km) in city_art.items():
        on=r>=km; m.set_visible(on); t.set_visible(on)
    counter.set_text(f"{min(r,520):.0f} km" if u<1 else "")
    hold=i-N_GROW
    show=u>=1
    fact1.set_visible(show); fact2.set_visible(show)
    if show:
        p=0.5+0.5*np.sin(hold/6)
        for k,art in enumerate(pul): art.set_alpha([.12,.22,.4][k]*(0.6+0.8*p))
    fig.canvas.draw()
    frames.append(np.asarray(fig.canvas.buffer_rgba())[...,:3].copy())
w=imageio.get_writer(OUT,fps=30,codec="libx264",quality=8,macro_block_size=2)
for f in frames: w.append_data(f)
w.close()
imageio.imwrite("day10_prompting_only.png",frames[-1])  # final still
print(frames[0].shape,len(frames))

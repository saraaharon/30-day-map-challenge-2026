import numpy as np, cv2, sys, subprocess
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import matplotlib.patheffects as pe

W,H=1080,1350; FPS=20; DUR=35
PX0,PX1,PY0,PY1=65,765,170,994          # painting box on canvas
IMW,IMH=1117,1315
SX=(PX1-PX0)/IMW; SY=(PY1-PY0)/IMH
M_PER_PX=10000/IMW
E0,N0=5200000,2600000
BG='#0b0d1e'; CY='#00e5ff'; MG='#ff2d95'; YL='#ffd23f'; GR='#8a93b0'; LIM='#39ff88'
AUTHOR='Sara AHARON'

im=cv2.cvtColor(cv2.imread('/mnt/user-data/uploads/unnamed.jpg'),cv2.COLOR_BGR2RGB)
lbl=np.load('lbl.npy'); rgb=np.load('rgb.npy')
names={4:'Basalt / bare rock',7:'Shadowed slopes',5:'Burnt field',1:'Glacial water',
       3:'Lichen & scree',6:'Dry steppe',2:'Snow & ice',0:'Sulphur crust'}
cnt=np.bincount(lbl.ravel(),minlength=8)/lbl.size*100
order=np.argsort(-cnt)

# ---------- vectorize ----------
def polys(cls,n,prefix):
    m=(lbl==cls).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((17,17),np.uint8))
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cs=sorted(cs,key=cv2.contourArea,reverse=True)[:n]
    out=[]
    for i,c in enumerate(cs):
        a=cv2.contourArea(c); ap=cv2.approxPolyDP(c,2.5,True)[:,0,:].astype(float)
        M=cv2.moments(c); cx,cy=M['m10']/M['m00'],M['m01']/M['m00']
        out.append(dict(id=f'{prefix}{i+1}',cls=names[cls],area_ha=a*M_PER_PX**2/1e4,
            perim=cv2.arcLength(c,True)*M_PER_PX,E=E0+cx*M_PER_PX,N=N0+(IMH-cy)*M_PER_PX,
            xy=np.c_[PX0+ap[:,0]*SX,PY0+ap[:,1]*SY],lab=(PX0+cx*SX,PY0+cy*SY),prefix=prefix))
    return out
feats=polys(4,5,'R')+polys(0,3,'S')
for f in feats: print(f['id'],round(f['area_ha'],1),round(f['perim']),int(f['E']),int(f['N']))

# ---------- figure ----------
fig=plt.figure(figsize=(W/100,H/100),dpi=100,facecolor=BG)
ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off'); ax.set_facecolor(BG)
ext=(PX0,PX1,PY1,PY0)
ax.imshow(im,extent=ext,zorder=1,aspect='auto')
clsim=ax.imshow(rgb[lbl],extent=ext,zorder=2,aspect='auto',alpha=0.9)
scan,=ax.plot([PX0,PX1],[PY0,PY0],color=CY,lw=2.5,zorder=9,alpha=0)
ax.add_patch(Rectangle((PX0,PY0),PX1-PX0,PY1-PY0,fill=False,ec=CY,lw=1,zorder=8))

ax.text(65,72,'Is this a map?',color='white',fontsize=26,fontweight='bold',va='center')
ax.text(65,110,'Landscapes II, Untitled VI  —  classified, vectorized, georeferenced',color=MG,fontsize=13,va='center')
ft='#30DayMapChallenge | Day 25: Is this a map? | '+AUTHOR
ax.text(65,1296,ft,color=GR,fontsize=8,va='center')
ax.text(65,1312,'Artwork: Landscapes II, Untitled VI, 2021, acrylic on canvas, 70 × 60 cm, Cristina DIMUȘ',color=GR,fontsize=8,va='center')
ax.text(65,1328,'Method: k-means (k=8, Lab) · contours via OpenCV · pseudo-georeferenced, fictional location',color=GR,fontsize=8,va='center')

# graticule
grat=[]
for k in range(0,11000,2000):
    x=PX0+k*0.07
    if 0<k<10000: grat.append(ax.plot([x,x],[PY0,PY1],color=CY,lw=0.8,alpha=0,zorder=6)[0])
    grat.append(ax.text(x,163,f'{E0+k:,}',color=CY,fontsize=6.5,ha='center',va='bottom',alpha=0,zorder=6))
for k in range(0,12000,2000):
    y=PY1-k*0.07
    if k>0: grat.append(ax.plot([PX0,PX1],[y,y],color=CY,lw=0.8,alpha=0,zorder=6)[0])
    grat.append(ax.text(PX0-4,y,f'{N0+k:,}',color=CY,fontsize=6.5,ha='right',va='center',alpha=0,zorder=6))

# polygons
polyA=[]; 
for f in feats:
    col=MG if f['prefix']=='R' else CY
    p=Polygon(f['xy'],closed=True,fc=(*matplotlib.colors.to_rgb(col),0),ec=(*matplotlib.colors.to_rgb(col),0),lw=1.8,zorder=7); ax.add_patch(p)
    t=ax.text(*f['lab'],f['id'],color='white',fontsize=9,fontweight='bold',ha='center',va='center',zorder=10,alpha=0)
    t.set_path_effects([pe.withStroke(linewidth=2.5,foreground='#0b0d1e')])
    polyA.append((p,t,col))

# meter
RX=790; RR=1015
ax.text(RX,188,'IS THIS A MAP?',color=GR,fontsize=9,va='center')
word=ax.text(RX,238,'NO',color=MG,fontsize=44,fontweight='bold',va='center')
ax.add_patch(Rectangle((RX,266),RR-RX,9,fc='#1b1f3a',ec='none'))
bar=Rectangle((RX,266),0,9,fc=MG,ec='none'); ax.add_patch(bar)
ax.text(RX,292,'MAP-NESS',color=GR,fontsize=8,va='center')
pct=ax.text(RR,292,'0%',color='white',fontsize=9,fontweight='bold',ha='right',va='center')

# legend
leg=[]
leg.append(ax.text(RX,345,'LEGEND',color=CY,fontsize=9,fontweight='bold',va='center',alpha=0))
leg.append(ax.text(RX,362,'Land cover · k-means, k=8',color=GR,fontsize=7.5,va='center',alpha=0))
for i,c in enumerate(order):
    y=392+i*28
    col=tuple(rgb[c]/255)
    r=Rectangle((RX,y-9),18,18,fc=col,ec='white',lw=0.5,alpha=0); ax.add_patch(r); leg.append(r)
    leg.append(ax.text(RX+26,y,names[c],color='white',fontsize=8.5,va='center',alpha=0))
    leg.append(ax.text(RR,y,f'{cnt[c]:.1f}%',color=GR,fontsize=8,ha='right',va='center',alpha=0))
for j,(col,tx) in enumerate([(MG,'Rock polygons (R)'),(CY,'Sulphur polygons (S)')]):
    y=392+8*28+10+j*24
    r=Rectangle((RX,y-8),18,16,fc=col,ec=col,lw=1.5,alpha=0); r.set_facecolor((*matplotlib.colors.to_rgb(col),0)); ax.add_patch(r); leg.append(r)
    leg.append(ax.text(RX+26,y,tx,color='white',fontsize=8.5,va='center',alpha=0))

# table
tab=[]
cols=[(70,'left'),(125,'left'),(560,'right'),(690,'right'),(850,'right'),(1010,'right')]
hdr=['FID','Class','Area (ha)','Perim. (m)','Centroid E','Centroid N']
th=ax.text(65,1018,'ATTRIBUTE TABLE  ·  vectorized features',color=CY,fontsize=9,fontweight='bold',va='center',alpha=0)
hline=ax.plot([65,1015],[1050,1050],color=CY,lw=1,alpha=0)[0]
hrow=[ax.text(x,1036,h,color=GR,fontsize=8,ha=al,va='center',alpha=0,family='DejaVu Sans Mono') for (x,al),h in zip(cols,hdr)]
rows=[]
for i,f in enumerate(feats):
    y=1068+i*24
    vals=[f['id'],f['cls'],f"{f['area_ha']:,.1f}",f"{f['perim']:,.0f}",f"{f['E']:,.0f}",f"{f['N']:,.0f}"]
    col=MG if f['prefix']=='R' else CY
    rows.append([ax.text(x,y,v,color=(col if j==0 else 'white'),fontsize=8.5,ha=al,va='center',alpha=0,family='DejaVu Sans Mono') for j,((x,al),v) in enumerate(zip(cols,vals))])

# scale bar (inside painting, bottom-right)
sb=[]
box=Rectangle((72,936),170,52,fc=BG,ec='none',alpha=0,zorder=11); ax.add_patch(box); sb.append(box)
for i in range(2):
    r=Rectangle((89+i*70,954),70,7,fc=('white' if i==0 else '#0b0d1e'),ec='white',lw=0.8,alpha=0,zorder=12); ax.add_patch(r); sb.append(r)
for xx,l in [(89,'0'),(159,'1'),(229,'2 km')]:
    sb.append(ax.text(xx,974,l,color='white',fontsize=7.5,ha='center',va='center',alpha=0,zorder=12))

# north arrow
na=[]
cx0,cy0=RX+112,730
arr=Polygon([[cx0,cy0-38],[cx0+14,cy0+22],[cx0,cy0+10],[cx0-14,cy0+22]],closed=True,fc='white',ec='white',lw=1,zorder=5); arr.set_alpha(0); ax.add_patch(arr); na.append(arr)
na.append(ax.text(cx0,cy0-48,'N',color='white',fontsize=11,fontweight='bold',ha='center',va='center',alpha=0))

# projection note
proj=[ax.text(RX,790+i*19,s,color=(CY if i==0 else GR),fontsize=7.5,va='center',alpha=0,family='DejaVu Sans Mono') for i,s in enumerate(
 ['PROJECTION','CRS    ETRS89-LAEA (EPSG:3035)','Extent 10.0 × 11.8 km','Scale  ~1:16,700 on canvas','Datum  pseudo (fictional)','Source acrylic on canvas'])]

cap=ax.text(RX,1000,'A map is a claim\nabout a place.\nThis one claims\na place that\ndoesn\'t exist.',color='white',fontsize=11,style='italic',va='top',alpha=0,linespacing=1.5)

def seg(t,a,b):
    x=min(1,max(0,(t-a)/(b-a))); return x*x*(3-2*x)
WT=dict(grat=14,cls=20,leg=10,poly=14,tab=12,scale=9,north=9,proj=12)

def frame(t):
    fade=seg(t,27.5,30.5); k=1-fade
    a=dict(grat=seg(t,2,4.5)*k, cls=seg(t,5,8.5)*k, leg=seg(t,9,10.5)*k,
           scale=seg(t,19.5,20.5)*k, north=seg(t,21,22)*k, proj=seg(t,23,24.5)*k)
    pa=[seg(t,11+i*0.35,11.8+i*0.35)*k for i in range(len(feats))]
    ta=[seg(t,15+ (0 if i<0 else 0.5+i*0.45),15.8+0.5+i*0.45)*k for i in range(len(feats))]
    th_a=seg(t,15,15.8)*k
    a['poly']=np.mean(pa); a['tab']=np.mean(ta)*0.6+th_a*0.4
    m=sum(WT[n]*a[n] for n in WT)
    # grid
    for g in grat: g.set_alpha(0.55*a['grat'] if hasattr(g,'get_xdata') else a['grat'])
    # classification wipe
    w=seg(t,5,8.5)
    h=(PY1-PY0)*w
    clsim.set_alpha(0.9*k)
    clsim.set_clip_path(Rectangle((PX0,PY0),PX1-PX0,max(h,0.01),transform=ax.transData))
    scan.set_ydata([PY0+h]*2); scan.set_alpha(0.9*(1 if 0<w<1 else 0))
    for o in leg:
        if isinstance(o,Rectangle): o.set_alpha(a['leg'])
        else: o.set_alpha(a['leg'])
    for (p,tx,col),al in zip(polyA,pa):
        p.set_facecolor((*matplotlib.colors.to_rgb(col),0.18*al)); p.set_edgecolor((*matplotlib.colors.to_rgb(col),al)); tx.set_alpha(al)
    th.set_alpha(th_a); hline.set_alpha(th_a)
    for o in hrow: o.set_alpha(th_a)
    for r,al in zip(rows,ta):
        for o in r: o.set_alpha(al)
    for o in sb:
        o.set_alpha(a['scale']*(0.6 if o is box else 1))
    for o in na: o.set_alpha(a['north'])
    for o in proj: o.set_alpha(a['proj'])
    # meter
    final=t>=31.5
    yes=(m>=80) if not final else None
    if final:
        word.set_text('BOTH'); word.set_color('white'); word.set_fontsize(40); pct.set_text('—'); col='white'; m_show=100
    else:
        word.set_text('YES' if yes else 'NO'); word.set_color(LIM if yes else MG); word.set_fontsize(44)
        pct.set_text(f'{m:.0f}%'); col=LIM if yes else MG; m_show=m
    bar.set_width((RR-RX)*m_show/100); bar.set_facecolor(col)
    cap.set_alpha(seg(t,31.5,33))
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()

if __name__=='__main__':
    if len(sys.argv)>1:
        for tt in map(float,sys.argv[1:]):
            plt.imsave(f'test_{tt:.0f}.png',frame(tt))
    else:
        import imageio
        wr=imageio.get_writer('day25_is_this_a_map.mp4',fps=FPS,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p')
        n=FPS*DUR
        for i in range(n):
            wr.append_data(frame(i/FPS))
            if i%50==0: print(i,n,flush=True)
        wr.close()
        plt.imsave('day25_final_still.png',frame(26.5)); print('done')

import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
BLUE="#2a78d6"; ORANGE="#eb6834"; GREY="#8a8a8a"; LIGHT="#cde2fb"; INK="#222222"
plt.rcParams.update({"font.family":"serif","font.size":9})
fig=plt.figure(figsize=(6.85,3.6))
# ---------------- (a) motif and event trains
ax=fig.add_axes([0.01,0.08,0.27,0.84]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
P={"a":(2.0,8.6),"b":(0.6,6.0),"c":(3.4,6.0)}
for (u,v),lab,off in [(("a","b"),"e1",(-0.55,0.1)),(("a","c"),"e2",(0.45,0.1)),(("b","c"),"e3",(0,-0.45))]:
    ax.plot([P[u][0],P[v][0]],[P[u][1],P[v][1]],color=INK,lw=1.2,zorder=1)
    ax.text((P[u][0]+P[v][0])/2+off[0],(P[u][1]+P[v][1])/2+off[1],lab,ha="center",fontsize=9,style="italic")
for n,(x,y) in P.items():
    ax.scatter(x,y,s=260,color="white",edgecolor=INK,lw=1.2,zorder=2); ax.text(x,y,n,ha="center",va="center",fontsize=9,zorder=3)
rng=np.random.default_rng(3)
base=np.sort(rng.uniform(4.6,9.7,6)); co=[5.3,7.1,8.6]
trains={"T1":np.r_[base[:3],co],"T2":np.r_[rng.uniform(4.6,9.7,3),np.array(co)+0.08],"T3":np.r_[rng.uniform(4.6,9.7,3),np.array(co)-0.1]}
for k,(name,tt) in enumerate(trains.items()):
    y=4.3-k*1.25; ax.plot([4.4,9.9],[y,y],color=GREY,lw=0.6)
    ax.text(4.2,y,name.replace("T","T_e"),ha="right",va="center",fontsize=9,style="italic")
    for t in tt:
        c=ORANGE if any(abs(t-x)<0.15 for x in co) else INK
        ax.plot([t,t],[y-0.32,y+0.32],color=c,lw=1.6)
for x in co: ax.add_patch(Rectangle((x-0.22,0.9),0.44,3.8,color=ORANGE,alpha=0.12,lw=0))
ax.text(7.15,0.35,"time →",ha="center",fontsize=9.5,color=GREY)
fig.text(0.01,0.97,"(a) Motif and event trains",fontsize=9.5,weight="bold",va="top")
ax.text(4.9,8.4,"Orange: events\nthat co-occur\nacross edges\nwithin δ",fontsize=9,color=ORANGE,va="top")
# ---------------- (b) conditional null for one event
ax=fig.add_axes([0.33,0.08,0.36,0.82]); ax.set_xlim(0,10); ax.set_ylim(1.2,9.2); ax.axis("off")
fig.text(0.345,0.97,"(b) Local null for one event t",fontsize=9.5,weight="bold",va="top")
Ti=[1.6,3.9,6.3,8.4]; d=0.35
yTi,yTj,yA=7.6,5.9,3.6
for y,lab in [(yTi,"T_i"),(yTj,"T_j"),(yA,"A_j")]:
    ax.plot([0.9,9.8],[y,y],color=GREY,lw=0.6); ax.text(0.7,y,lab,ha="right",va="center",style="italic")
for x in Ti:
    ax.plot([x,x],[yTi-0.3,yTi+0.3],color=INK,lw=1.6)
    for y0,y1 in [(yTi-0.45,yTi+0.45),(yA-0.5,yA+0.5)]:
        ax.add_patch(Rectangle((x-d,y0),2*d,y1-y0,color=LIGHT,lw=0,zorder=0))
t=6.45; ax.plot([t,t],[yTj-0.3,yTj+0.3],color=ORANGE,lw=2.2); ax.text(t+0.2,yTj-0.75,"t",ha="center",color=ORANGE,style="italic")
for x in [2.4,4.8,8.9]: ax.plot([x,x],[yTj-0.3,yTj+0.3],color=INK,lw=1.6)
ax.annotate("",xy=(6.3,yTi-0.5),xytext=(t,yTj+0.35),arrowprops=dict(arrowstyle="->",color=ORANGE,lw=0.9))
ax.text(6.55,6.75,"hit: within δ of T_i",fontsize=9,color=ORANGE)
r=np.random.default_rng(7); pool=np.sort(np.r_[r.uniform(1.0,9.7,26),r.normal(6.2,0.7,14)]); pool=pool[(pool>1)&(pool<9.7)]
K=12; q=int(np.searchsorted(pool,t)); lo_i=int(np.clip(q-K//2,0,len(pool)-K)); near=pool[lo_i:lo_i+K]   # rank window, as in NRC
for x in pool:
    isn=x in near; c=BLUE if isn else GREY
    ax.plot([x,x],[yA-0.3,yA+0.3],color=c,lw=1.3 if isn else 0.9)
lo,hi=near.min(),near.max()
ax.annotate("",xy=(lo,yA-0.75),xytext=(hi,yA-0.75),arrowprops=dict(arrowstyle="<->",color=BLUE,lw=0.9))
ax.text((lo+hi)/2,yA-1.0,"rank window of k\npool events around t",ha="center",va="top",fontsize=9,color=BLUE)
# ---------------- (c) per-scale evidence (real data)
Z=json.load(open("zscale.json"))["joint_bursts"]
ax=fig.add_axes([0.8,0.2,0.19,0.6])
lab=["1m","5m","15m","1h","4h"]; x=np.arange(5); w=0.38
ax.bar(x-w/2-0.01,Z["norm"],w,color=GREY,label="Normal motifs")
ax.bar(x+w/2+0.01,Z["anom"],w,color=BLUE,label="Anomalous motifs")
ax.axhline(0,color=INK,lw=0.6); ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=9,rotation=45)
ax.set_xlabel("Time scale δ"); ax.set_ylabel("Mean excess Z(δ)")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",color="#e5e5e5",lw=0.6); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=9,loc="lower center",bbox_to_anchor=(0.45,1.0),ncol=1)
fig.text(0.72,0.97,"(c) Evidence per scale",fontsize=9.5,weight="bold",va="top")
fig.savefig("../figures/fig_nrc_method.png",dpi=300); fig.savefig("../figures/fig_nrc_method.pdf"); print("ok")

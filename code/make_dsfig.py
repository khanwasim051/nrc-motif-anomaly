import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
DS=["uci","email_eu","mathoverflow"]; DN={"uci":"UCI","email_eu":"Email-Eu","mathoverflow":"MathOverflow","thiers13":"Thiers13"}
VN={"anchored":"Anchored","joint_bursts":"Joint bursts"}; DL=[60.0,600.0,3600.0,14400.0]
S={}
for d in DS:
    G=defaultdict(list)
    for l in open(f"ds_{d}.jsonl"):
        r=json.loads(l); G[(r["delta"],r["variant"],r["rho"])].append(r)
    for k,v in G.items():
        S[(d,)+k]=dict(naive=np.mean([x["naive_best5"] for x in v]),nrc=np.mean([x["nrc"] for x in v]),
                       matched=np.mean([x["single"][str(int(min([60,300,900,3600,14400],key=lambda s:abs(np.log(s/x["delta"])))))] for x in v]),
                       naive_ap=np.mean([x["naive_best5_ap"] for x in v]),nrc_ap=np.mean([x["nrc_ap"] for x in v]))
SD={}
for d in DS:
    G=defaultdict(list)
    for l in open(f"ds_{d}.jsonl"):
        r=json.loads(l); G[(r["delta"],r["variant"],r["rho"])].append(r)
    for k,v in G.items():
        SD[(d,)+k]=dict(naive=np.std([x["naive_best5"] for x in v],ddof=1),nrc=np.std([x["nrc"] for x in v],ddof=1),
                        matched=np.std([x["single"][str(int(min([60,300,900,3600,14400],key=lambda s:abs(np.log(s/x["delta"])))))] for x in v],ddof=1))
json.dump({"|".join(map(str,k)):v for k,v in S.items()},open("dsweep_summary.json","w"))
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,ax=plt.subplots(2,3,figsize=(6.85,5.0),sharex=True)
for i,v in enumerate(VN):
    for j,d in enumerate(DS):
        a=ax[i,j]; x=DL
        for k,lab,col,fmt,off in [("naive","Best naive synchrony (best of 6 scales, oracle)","#eb6834","s--",0.93),("nrc","NRC, sum over scales (default)","#2a78d6","o-",1.0),("matched","NRC, single scale nearest δ (δ known)","#1baf7a","^:",1.07)]:
            a.errorbar([q*off for q in x],[S[(d,dl,v,0.25)][k] for dl in DL],yerr=[SD[(d,dl,v,0.25)][k] for dl in DL],fmt=fmt,color=col,lw=1.8,ms=5.5,capsize=2.5,elinewidth=1,label=lab,mfc=("white" if k=="naive" else col),mew=1.4)
        a.axhline(0.5,color="#999",lw=0.8,ls=":"); a.set_xscale("log"); a.set_xticks(DL); a.set_xticklabels(["1m","10m","1h","4h"]); a.minorticks_off()
        a.grid(axis="y",color="#e5e5e5",lw=0.6); a.set_title(f"{DN[d]} – {VN[v]}",fontsize=9.5)
        if j==0: a.set_ylabel("AUROC")
        if i==1: a.set_xlabel("Injected window δ")
h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.01),fontsize=9.5)
fig.tight_layout(rect=(0,0.1,1,1)); fig.savefig("../figures/fig_delta.png",dpi=300); fig.savefig("../figures/fig_delta.pdf"); print("ok")
for d in DS:
    g=[S[(d,dl,v,r)]["nrc"]-S[(d,dl,v,r)]["naive"] for dl in DL for v in VN for r in [0.1,0.25]]
    print(d,'NRC-naive5 by delta:',{int(dl):round(np.mean([S[(d,dl,v,r)]["nrc"]-S[(d,dl,v,r)]["naive"] for v in VN for r in [0.1,0.25]]),3) for dl in DL},
          'matched-NRC:',{int(dl):round(np.mean([S[(d,dl,v,r)]["matched"]-S[(d,dl,v,r)]["nrc"] for v in VN for r in [0.1,0.25]]),3) for dl in DL})

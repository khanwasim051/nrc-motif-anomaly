import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import defaultdict
S=[json.loads(l) for l in open("sens_seeds.jsonl")]
DN={"uci":"UCI","email_eu":"Email-Eu","mathoverflow":"MathOverflow"}; VN={"anchored":"Anchored","joint_bursts":"Joint bursts"}
plt.rcParams.update({"font.family":"serif","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
G=defaultdict(list)
for r in S: G[(r["data"],r["variant"],r["rho"])].append(r)
RH=[0.05,0.1,0.25,0.5]
for metric,key,fn,yl,chance in [("AUROC","auroc","fig_auroc_rho","AUROC",0.5),("AP","ap","fig_ap_rho","Average precision",0.05)]:
    fig,ax=plt.subplots(2,3,figsize=(6.85,4.7),sharex=True)
    for i,v in enumerate(VN):
        for j,d in enumerate(DN):
            a=ax[i,j]
            for name,lab,col,mk,ls,off in [("naive","Best naive synchrony","#eb6834","s","--",0.97),("nrc","NRC (ours)","#2a78d6","o","-",1.03)]:
                m=[np.mean([x[f"{name}_{key}"] for x in G[(d,v,r)]]) for r in RH]; sd=[np.std([x[f"{name}_{key}"] for x in G[(d,v,r)]],ddof=1) for r in RH]
                a.errorbar([r*off for r in RH],m,yerr=sd,fmt=mk+ls,color=col,lw=1.8,ms=5.5,capsize=2.5,elinewidth=1,label=lab,mfc=("white" if name=="naive" else col),mew=1.4)
            a.axhline(chance,color="#999",lw=0.8,ls=":"); a.grid(axis="y",color="#e5e5e5",lw=0.6)
            a.set_xscale("log"); a.set_xticks(RH); a.set_xticklabels(["0.05","0.10","0.25","0.50"]); a.minorticks_off()
            a.set_title(f"{DN[d]} – {VN[v]}",fontsize=9.5)
            if j==0: a.set_ylabel(yl)
            if i==1: a.set_xlabel("ρ (log scale)")
    h,l=ax[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=2,frameon=False,bbox_to_anchor=(0.5,-0.01))
    fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(f"../figures/{fn}.png",dpi=300); fig.savefig(f"../figures/{fn}.pdf"); plt.close(fig)
print("ok", len(S))

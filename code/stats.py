import json, numpy as np
from scipy.stats import wilcoxon
rows=json.load(open('paper_rows.json'))
BL={}
for f in ['baselines_means.jsonl']:
    for l in open(f): r=json.loads(l); BL[(r['data'],r['variant'],r['rho'])]=r
nrc=[];nv=[];iff=[];best=0;tied=0;gap=[]
for r in rows:
    b=BL[(r[0],r[1],r[2])]
    others=[r[3],r[4],r[9],r[5]]+[b[k+' AUROC'] for k in ['MIDAS','MIDAS-R','burstiness','IForest(features)']]
    nrc.append(r[6]); nv.append(r[5]); iff.append(b['IForest(features) AUROC'])
    m=max(others)
    if r[6]>=m: best+=1
    elif m-r[6]<0.005: tied+=1
    if r[1]!='periodic': gap.append(r[6]-b['IForest(features) AUROC'])
    if r[6]<m: print('NRC not best:',r[:3],round(r[6],4),'best other',round(m,4))
def wtl(a,b):
    d=np.array(a)-np.array(b); return [int((d>=0.005).sum()),int((abs(d)<0.005).sum()),int((d<=-0.005).sum())]
def p(a,b): return f"{wilcoxon(a,b).pvalue:.3g}"
act={}
for l in open('actstats.jsonl'):
    r=json.loads(l); act[r['data']]={k:(f"{r[k]:g}" if k!='pool' else f"{r[k]:,.0f}") for k in ['events_per_edge','pool','pool_per_active_day']}
ST=dict(nrc_best=best,nrc_tied=tied,w_naive=wtl(nrc,nv),p_naive=p(nrc,nv),w_if=wtl(nrc,iff),p_if=p(nrc,iff),
        if_gap_min=f"{min(gap):.3f}",if_gap_max=f"{max(gap):.3f}",act=act,mo_span="2,350")
json.dump(ST,open('stats.json','w'),indent=1); print(json.dumps(ST))

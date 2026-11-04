import json, numpy as np
R=[json.loads(l) for d in ['uci','email_eu','mathoverflow'] for l in open(f'ov_{d}.jsonl')]
R=[r for r in R if r['mode']=='standard']
DS=['uci','email_eu','mathoverflow']; VS=['anchored','joint_bursts','periodic']; RH=[0.1,0.25]
cells=[(v,r) for v in VS for r in RH]
def arr(d,a,b,met):
    # shape cells x seeds
    return np.array([[ [x for x in R if x['data']==d and x['variant']==v and x['rho']==r and x['seed']==s][0][f'{a} {met}']-
                       [x for x in R if x['data']==d and x['variant']==v and x['rho']==r and x['seed']==s][0][f'{b} {met}'] for s in range(3)] for v,r in cells])
rng=np.random.default_rng(0); B=20000
def boot2(D, sub=None):        # two-level: cells then seeds
    D=D if sub is None else D[sub]
    nc,ns=D.shape; ci=rng.integers(0,nc,(B,nc)); si=rng.integers(0,ns,(B,nc,ns))
    m=D[ci[:,:,None],si].mean((1,2)); return np.quantile(m,[.025,.975])
def boot3(Ds):                  # three-level: datasets, cells, seeds
    out=[]
    for _ in range(B//4):
        ds=rng.integers(0,len(Ds),len(Ds)); vals=[]
        for k in ds:
            D=Ds[k]; c=rng.integers(0,D.shape[0],D.shape[0]); s=rng.integers(0,3,(D.shape[0],3))
            vals.append(D[c[:,None],s].mean())
        out.append(np.mean(vals))
    return np.quantile(out,[.025,.975])
res={}
for comp in [('NRC','naive'),('NRC','IForest')]:
    for met in ['AUROC','AP']:
        Ds=[arr(d,*comp,met) for d in DS]; key=f'{comp[0]}-{comp[1]} {met}'; res[key]={}
        for d,D in zip(DS,Ds):
            np_=np.array([0,1,2,3])     # non-periodic cells
            res[key][d]=dict(mean=D.mean(), ci=boot2(D).tolist(), mean_np=D[np_].mean(), ci_np=boot2(D,np_).tolist(),
                             seed_sd=float(np.sqrt(D.var(1,ddof=1).mean())), pos_cells=int((D.mean(1)>0).sum()),
                             pos_runs=int((D>0).sum()), cell_mean=D.mean(1).tolist(), cell_sd=D.std(1,ddof=1).tolist())
        res[key]['all']=dict(mean=float(np.mean([D.mean() for D in Ds])), ci=boot3(Ds).tolist())
        print(key)
        for d in DS+['all']:
            x=res[key][d]; print('  ',d, round(x['mean'],4), np.round(x['ci'],4), *( [round(x['mean_np'],4),np.round(x['ci_np'],4),'seedsd',round(x['seed_sd'],4),'poscells',x['pos_cells'],'posruns',x['pos_runs']] if d!='all' else []))
json.dump(res,open('stats2.json','w'),default=float)

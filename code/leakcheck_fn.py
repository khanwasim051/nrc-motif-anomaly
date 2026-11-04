import numpy as np
def activity_novelty(pool, ev2, nidx, W=86400.0):
    out=[]
    for a,b,c in pool:
        nodes=[a,b,c]; vals=[]
        for (u,v) in [(a,b),(a,c),(b,c)]:
            P=np.sort(np.concatenate([nidx[x][0][~np.isin(nidx[x][1],nodes)] for x in (u,v)]))
            t=ev2[(u,v)][1:]; cnt=np.searchsorted(P,t+W,side='right')-np.searchsorted(P,t-W); vals.extend(-np.log1p(cnt))
        out.append(np.mean(vals) if vals else 0.0)
    return np.array(out)

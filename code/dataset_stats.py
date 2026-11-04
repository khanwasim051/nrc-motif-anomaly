"""Table 2: nodes, events, time span, triangles and eligible triangles (>= 3 events per edge) per graph."""
import sys, json, numpy as np
from motif_bench import load, build, triangles
for name in (sys.argv[1:] or ["uci", "email_eu", "mathoverflow", "btc_alpha", "btc_otc"]):
    E = load(name); mult, birth, adj = build(E); tri = triangles(adj)
    elig = sum(1 for t in tri if min(mult[(t[0], t[1])], mult[(t[0], t[2])], mult[(t[1], t[2])]) >= 3)
    ts = np.array([e[2] for e in E])
    print(json.dumps(dict(data=name, nodes=len(adj), events=len(E), span_days=round(float(ts.max() - ts.min()) / 86400, 1), triangles=len(tri), eligible=elig)), flush=True)

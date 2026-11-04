"""Compare rebuilt summaries with the shipped ones: every number must agree to 1e-9."""
import json, sys, os
new_dir, ref_dir = sys.argv[1], sys.argv[2]
def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items(): yield from flat(v, f"{p}/{k}")
    elif isinstance(d, list):
        for i, v in enumerate(d): yield from flat(v, f"{p}[{i}]")
    else: yield p, d
ok = True
for f in sorted(os.listdir(ref_dir)):
    if not f.endswith(".json"): continue
    a = dict(flat(json.load(open(os.path.join(new_dir, f))))); b = dict(flat(json.load(open(os.path.join(ref_dir, f)))))
    bad = [k for k in b if k not in a or (isinstance(b[k], (int, float)) and not isinstance(b[k], bool) and abs(a[k] - b[k]) > 1e-9) or
           (not isinstance(b[k], (int, float)) and a[k] != b[k])]
    print(f"{'OK  ' if not bad and set(a) == set(b) else 'DIFF'} {f}: {len(b)} values" + (f", {len(bad)} differ, e.g. {bad[:3]}" if bad else ""))
    ok &= not bad and set(a) == set(b)
print("ALL SUMMARIES REPRODUCED" if ok else "SOME SUMMARIES DIFFER"); sys.exit(0 if ok else 1)

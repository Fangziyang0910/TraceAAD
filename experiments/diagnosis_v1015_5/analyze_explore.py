"""Summarize the Explore paired study: python -m experiments.diagnosis_v1015_5.analyze_explore TASK [LIMIT]"""
import json
import random
import statistics as st
import sys
from collections import defaultdict

task = sys.argv[1]
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
recs = [json.loads(l) for l in open(f"experiments_result/diagnosis_v1015_5/explore_study/{task}.jsonl")]
higher = recs[0]["higher"]


def good(r):  # valid within the nominal search limit
    return r["status"] == "valid" and r.get("eval_seconds", 1e9) <= limit


def gain(r, ref):  # relative improvement over ref, positive = better
    s = r["score"]
    return (s - ref) / abs(ref) if higher else (ref - s) / abs(ref)


arms = sorted({r["arm"] for r in recs}, key=lambda a: ["cur", "cur_t1", "cur_time", "v1", "v1_time", "restr", "restr_time", "restr_time_t1"].index(a))
ctx = defaultdict(dict)
for r in recs:
    ctx[(r["run"], r["parent_id"])].setdefault(r["arm"], []).append(r)
print(f"{task}: {len(ctx)} parents, limit {limit:g}s")
print("arm             valid  timeout>lim  dup  beat-parent  beat-best  gain>=1%  median-gain-vs-best  best-of-arm-wins  sim-parent  out-tok")
wins = defaultdict(float)
for key, d in ctx.items():
    # best child per arm (failures = -inf); ties split
    vals = {a: max([gain(r, r["best"]) for r in d.get(a, []) if good(r)], default=-1e9) for a in arms}
    top = max(vals.values())
    winners = [a for a, v in vals.items() if v == top]
    for a in winners:
        wins[a] += 1 / len(winners)
for a in arms:
    rs = [r for r in recs if r["arm"] == a]
    v = [r for r in rs if good(r)]
    slow = [r for r in rs if r["status"] == "valid" and r.get("eval_seconds", 0) > limit or r["status"] == "timeout"]
    dup = [r for r in rs if r["status"] == "duplicate"]
    bp = [r for r in v if gain(r, r["parent_score"]) > 1e-9]
    bb = [r for r in v if gain(r, r["best"]) > 1e-9]
    big = [r for r in v if gain(r, r["best"]) >= 0.01]
    mg = st.median([gain(r, r["best"]) for r in v]) if v else float("nan")
    sims = [r["sim_parent"] for r in rs if "sim_parent" in r]
    tok = [r["output_tokens"] for r in rs if r.get("output_tokens")]
    n = len(rs)
    print(f"{a:15s} {len(v)/n:5.0%} {len(slow)/n:8.0%} {len(dup)/n:9.0%} {len(bp)/n:9.0%} {len(bb)/n:10.0%} {len(big)/n:8.0%} "
          f"{mg*100:15.1f}% {wins[a]:14.1f} {st.median(sims) if sims else float('nan'):10.2f} {st.median(tok):7.0f}")
# paired bootstrap of beat-best rate difference vs cur
rng = random.Random(0)
keys = list(ctx)
for a in arms:
    if a == "cur":
        continue
    diffs = []
    for _ in range(2000):
        sample = [rng.choice(keys) for _ in keys]
        num = 0
        for k in sample:
            d = ctx[k]
            fa = sum(good(r) and gain(r, r["best"]) > 1e-9 for r in d.get(a, [])) / max(1, len(d.get(a, [])))
            fc = sum(good(r) and gain(r, r["best"]) > 1e-9 for r in d.get("cur", [])) / max(1, len(d.get("cur", [])))
            num += fa - fc
        diffs.append(num / len(sample))
    diffs.sort()
    print(f"beat-best rate {a} - cur: {st.mean(diffs)*100:+.1f} pp  95% CI [{diffs[50]*100:+.1f}, {diffs[1949]*100:+.1f}]")

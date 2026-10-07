"""Transfer of training progress to the selection set along each run's training frontier.

usage: python -m experiments.diagnosis_v1016_frontier.analyze
"""
import json
import statistics
from collections import defaultdict

OUT = "experiments_result/diagnosis_v1016_frontier"


def rel(a, b):
    """Relative improvement of fitness a over b (fitness: lower is better)."""
    return (b - a) / abs(b)


def main():
    programs = json.load(open(f"{OUT}/frontier.json"))
    sel = {(r["run"], r["id"]): r["selection_fitness"] for r in map(json.loads, open(f"{OUT}/selection.jsonl"))}
    runs = defaultdict(list)
    for p in programs:
        p["sel"] = sel.get((p["run"], p["id"]))
        runs[p["run"]].append(p)
    steps = []  # successive frontier programs, both evaluated
    at = defaultdict(dict)  # run -> checkpoint -> selection fitness of the training best then
    for run, ps in runs.items():
        ps.sort(key=lambda p: p["t"])
        ok = [p for p in ps if p["sel"] is not None]
        for a, b in zip(ok, ok[1:]):
            steps.append({"ver": b["ver"], "task": b["task"], "t": b["t"],
                          "dtrain": rel(b["fitness"], a["fitness"]), "dsel": rel(b["sel"], a["sel"])})
        if ps[0]["n_att"] < 1000 or len(ok) < len(ps):
            continue
        for cut in (250, 500, 750, 1000):
            last = [p for p in ps if p["t"] <= cut]
            if last:
                at[run][cut] = (last[-1]["sel"], last[-1]["fitness"], ps[0]["ver"], ps[0]["task"])
        best_sel = min(p["sel"] for p in ok)
        at[run]["best_sel"] = best_sel

    print("successive training-frontier steps: share whose selection score also improves")
    for ver in ("V15", "V16"):
        for lo, hi in ((0, 250), (250, 500), (500, 750), (750, 1000)):
            s = [x for x in steps if x["ver"] == ver and lo < x["t"] <= hi]
            if s:
                up = sum(x["dsel"] > 1e-12 for x in s)
                down = sum(x["dsel"] < -1e-12 for x in s)
                print(f"  {ver} t {lo:4}-{hi:4}: n={len(s):3}  sel up {up:3}  flat {len(s) - up - down:3}  down {down:3}  "
                      f"median dtrain {100 * statistics.median(x['dtrain'] for x in s):.3f}%  "
                      f"median dsel {100 * statistics.median(x['dsel'] for x in s):+.3f}%")
    print("\nby task (all steps after t=300): sel up / flat / down")
    for task in sorted({x["task"] for x in steps}):
        s = [x for x in steps if x["task"] == task and x["t"] > 300]
        up = sum(x["dsel"] > 1e-12 for x in s)
        down = sum(x["dsel"] < -1e-12 for x in s)
        print(f"  {task:20} n={len(s):3}  {up}/{len(s) - up - down}/{down}  "
              f"sum dsel {100 * sum(x['dsel'] for x in s):+.2f}%  sum dtrain {100 * sum(x['dtrain'] for x in s):+.2f}%")
    print("\ncompleted runs: selection score of the training best at each checkpoint, relative to t=250")
    for run, c in sorted(at.items()):
        if 250 not in c:
            continue
        base = c[250][0]
        row = "  ".join(f"{cut}:{100 * rel(c[cut][0], base):+6.2f}%/{100 * rel(c[cut][1], c[250][1]):+5.2f}%"
                        for cut in (500, 750, 1000) if cut in c)
        print(f"  {c[250][2]} {c[250][3]:18} {run[-4:]}  sel/train  {row}   frontier best sel vs t=1000 "
              f"{100 * rel(c['best_sel'], c[1000][0]):+.2f}%")


if __name__ == "__main__":
    main()

"""Summary of the equal-development study (see develop.py).

usage: python -m experiments.diagnosis_v1016_develop.analyze

Gains are relative to the run's training best before development (fitness: higher is
better), so 0 means "reached the search best", and a new design above the ``best``
arm's result means equal development took it past further development of the best.
"""
import json
from collections import defaultdict
from pathlib import Path

OUT = Path("experiments_result/diagnosis_v1016_develop")


def rel(a, b):
    return 100 * (a - b) / abs(b)


def main():
    records = [json.loads(p.read_text()) for p in sorted((OUT / "arms").glob("*.json"))]
    runs = defaultdict(dict)
    for r in records:
        runs[(r["task"], r["run"])][r["arm"]] = r
    print("training fitness relative to the search best before development (%), start -> reached; "
          "selection set likewise relative to the search best's selection score")
    for (task, run), arms in sorted(runs.items()):
        if "best" not in arms:
            continue
        ref = arms["best"]["selection"]["start"]
        cells = []
        for arm in ("best", "new1", "new2"):
            if arm not in arms:
                continue
            s = arms[arm]["selection"]
            valid = sum(x["status"] == "valid" for x in arms[arm]["steps"])
            sel = (f"{rel(s['reached']['selection'], ref['selection']):+.2f}"
                   if s["reached"]["selection"] is not None and ref["selection"] is not None else "fail")
            cells.append(f"{arm}: train {rel(s['start']['training'], ref['training']):+6.2f} -> "
                         f"{rel(s['reached']['training'], ref['training']):+6.2f}  sel {sel}  "
                         f"valid {valid}/{arms[arm]['generations']}")
        print(f"{task:18} {run[-4:]}  " + " | ".join(cells))
    new = [(r, runs[(r["task"], r["run"])].get("best")) for r in records if r["arm"].startswith("new")]
    new = [(r, b) for r, b in new if b is not None]
    above = sum(r["selection"]["reached"]["training"] > b["selection"]["reached"]["training"] for r, b in new)
    past = sum(r["selection"]["reached"]["training"] > b["selection"]["start"]["training"] for r, b in new)
    print(f"\nnew designs: {len(new)}; passed the search best: {past}; "
          f"ended above the developed best: {above}")
    for label, group in (("best", [r for r in records if r["arm"] == "best"]),
                         ("new", [r for r, _ in new])):
        gains = [rel(r["selection"]["reached"]["training"], r["selection"]["start"]["training"]) for r in group]
        if gains:
            gains.sort()
            print(f"{label:4} training gain from its own start over the arm: median {gains[len(gains) // 2]:+.2f}%  "
                  f"improved {sum(g > 1e-9 for g in gains)}/{len(gains)}")


if __name__ == "__main__":
    main()

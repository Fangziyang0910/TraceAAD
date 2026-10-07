"""Summarize the V10.20 fixed-parent replay (results.jsonl from replay.py).

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1020_replay.analyze

Per task and arm: valid children, children better than the parent (against the parent
measured again in the same pool), the median relative change of the score, the child's
evaluation time relative to the parent's, and the code similarity to the parent (whether
the parent's computation is kept). Fitness is higher-is-better on every task, so a positive
change is an improvement.
"""

import json
import statistics

from traceaad.common.canonical import similarity
from traceaad.common.state import Facts
from experiments.diagnosis_v1020_replay.replay import OUT, run_dir


def summary():
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    parents = json.load(open(OUT / "parents.json"))
    table = {}
    for task, info in parents.items():
        measured = [r for r in rows if r["task"] == task and r["arm"] == "parent" and r["status"] == "valid"]
        if not measured:
            continue
        parent_fitness = statistics.fmean(r["fitness"] for r in measured)
        parent_seconds = statistics.fmean(r["eval_seconds"] for r in measured)
        code = Facts(run_dir(task)).programs[info["parent_id"]]["code"]
        table[task] = {"parent": {"fitness": parent_fitness, "archived_fitness": info["fitness"],
                                  "eval_seconds": parent_seconds}}
        for arm in ("R0", "R1", "D"):
            children = [r for r in rows if r["task"] == task and r["arm"] == arm]
            valid = [r for r in children if r.get("status") == "valid"]
            change = [(r["fitness"] - parent_fitness) / abs(parent_fitness) for r in valid]
            ratio = [r["eval_seconds"] / parent_seconds for r in valid]
            kept = [similarity(r["code"], code) for r in children if r.get("code")]
            table[task][arm] = {
                "children": len(children), "valid": len(valid),
                "statuses": {s: sum(r.get("status") == s for r in children)
                             for s in sorted({r.get("status") for r in children})},
                "better": sum(c > 1e-9 for c in change),
                "change_median": statistics.median(change) if change else None,
                "change_best": max(change) if change else None,
                "time_ratio_median": statistics.median(ratio) if ratio else None,
                "time_ratio_above_5": sum(x > 5 for x in ratio),
                "similarity_to_parent_median": statistics.median(kept) if kept else None}
    return table


def main():
    table = summary()
    print(json.dumps(table, indent=1))
    with open(OUT / "summary.json", "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()

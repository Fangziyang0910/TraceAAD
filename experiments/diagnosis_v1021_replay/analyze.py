"""Summarize the V10.21 goal replay (results.jsonl from replay.py).

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_v1021_replay.analyze

Per arm, pooled over decision states and by task: valid children, children better than
their starting program (measured under the same execution), the median relative change,
children better than the search's best at that time, code similarity to the starting
program and, for Crossover, to the reference. Fitness is lower-is-better.
"""

import json
import statistics
from collections import defaultdict

from traceaad.common.canonical import similarity
from traceaad.common.state import Facts
from experiments.diagnosis_v1021_replay.replay import OUT, run_dir


def sim(a, b):
    try:
        return similarity(a, b)
    except Exception:  # unparseable source (invalid_source children)
        return None


def stats(rows, states, codes):
    valid = [r for r in rows if r.get("status") == "valid"]
    change = [(states[r["state"]]["parent_fitness"] - r["fitness"]) / abs(states[r["state"]]["parent_fitness"]) for r in valid]
    with_code = [r for r in rows if r.get("code")]
    to_parent = [x for x in (sim(r["code"], codes[r["state"]][0]) for r in with_code) if x is not None]
    out = {"children": len(rows), "valid": len(valid),
           "better": sum(c > 1e-9 for c in change),
           "better_than_search_best": sum(r["fitness"] < states[r["state"]]["search_best"] - 1e-9 for r in valid),
           "change_median": statistics.median(change) if change else None,
           "change_mean": statistics.fmean(change) if change else None,
           "similarity_to_parent_median": statistics.median(to_parent) if to_parent else None,
           "near_copies": sum(s >= 0.95 for s in to_parent)}
    refs = [x for x in (sim(r["code"], codes[r["state"]][1]) for r in with_code if codes[r["state"]][1]) if x is not None]
    if refs:
        out["similarity_to_reference_median"] = statistics.median(refs)
    return out


def summary():
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    states = json.load(open(OUT / "states.json"))
    codes = {}
    for name, s in states.items():
        task, rep, _ = name.split("/")
        programs = Facts(run_dir(task, int(rep.removeprefix("rep")))).programs
        codes[name] = (programs[s["parent_id"]]["code"], programs[s["reference_id"]]["code"] if s["reference_id"] else None)
    table = {"pooled": {}, "by_task": defaultdict(dict)}
    for arm in sorted({r["arm"] for r in rows}):
        arm_rows = [r for r in rows if r["arm"] == arm]
        table["pooled"][arm] = stats(arm_rows, states, codes)
        for task in sorted({r["task"] for r in arm_rows}):
            table["by_task"][task][arm] = stats([r for r in arm_rows if r["task"] == task], states, codes)
    return table


def main():
    table = summary()
    print(json.dumps(table["pooled"], indent=1))
    for task, arms in table["by_task"].items():
        print(task, {arm: (s["valid"], s["better"], None if s["change_median"] is None else round(100 * s["change_median"], 2))
                     for arm, s in arms.items()})
    with open(OUT / "summary.json", "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()

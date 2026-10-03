"""Paired summary of the profile replay (see replay.py).

usage: python -m experiments.diagnosis_v1016_profile.analyze

A child counts as within the search limit when its local time, relative to its parent's
mean local time in the same pool, scaled to the parent's logged search time, is at most
30 s. "Improved" means valid, within the limit and a better training score than the parent.
Intervals are 95% bootstrap intervals over items (prompts), the unit of pairing.
"""
import json
import random
import statistics
from collections import defaultdict

OUT = "experiments_result/diagnosis_v1016_profile"
SEARCH_LIMIT = 30.0


def main():
    items = {f"r{i['rep']}a{i['attempt_id']}": i for i in json.load(open(f"{OUT}/items_all.json"))}
    rows = [json.loads(line) for line in open(f"{OUT}/results.jsonl")]
    parent_seconds = defaultdict(list)
    for r in rows:
        if r["arm"] == "parent" and r["status"] == "valid":
            parent_seconds[r["item"]].append(r["seconds"])
    outcome = defaultdict(lambda: defaultdict(list))  # arm -> item -> outcomes
    gaps = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["arm"] == "parent" or r["item"] not in parent_seconds:
            continue
        item = items[r["item"]]
        local = statistics.fmean(parent_seconds[r["item"]])
        if r["status"] == "invalid":
            kind = "invalid"
        elif r["status"] == "runtime_error":
            kind = "error"
        elif r["status"] == "timeout" or r["seconds"] / local * item["parent_seconds"] > SEARCH_LIMIT:
            kind = "over_limit"
        elif r["score"] > item["parent_fitness"] + 1e-9:
            kind = "improved"
        else:
            kind = "not_better"
        outcome[r["arm"]][r["item"]].append(kind)
        if kind in ("improved", "not_better"):
            gaps[r["arm"]][r["item"]].append((r["score"] - item["parent_fitness"]) / abs(item["parent_fitness"]))

    kinds = ("improved", "not_better", "over_limit", "error", "invalid")
    print(f"items with a timed parent: {len(parent_seconds)}")
    for arm in ("base", "profile"):
        flat = [k for ks in outcome[arm].values() for k in ks]
        print(f"{arm:8} n={len(flat):4} " + "  ".join(f"{k}={100 * flat.count(k) / max(len(flat), 1):5.1f}%" for k in kinds))

    paired = [i for i in outcome["base"] if i in outcome["profile"]]
    rng = random.Random(0)

    def rate(arm, i, kind):
        return outcome[arm][i].count(kind) / len(outcome[arm][i])

    for kind in ("improved", "over_limit", "error"):
        diffs = [rate("profile", i, kind) - rate("base", i, kind) for i in paired]
        boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(5000))
        print(f"profile - base, {kind:10}: {100 * statistics.fmean(diffs):+5.1f} pp "
              f"[{100 * boots[125]:+.1f}, {100 * boots[4874]:+.1f}] over {len(paired)} items")
    for arm in ("base", "profile"):
        anyimp = sum("improved" in outcome[arm][i] for i in paired)
        print(f"{arm:8} items with at least one improved child: {anyimp}/{len(paired)}")
    for arm in ("base", "profile"):
        best = [max(gaps[arm][i]) for i in paired if gaps[arm][i]]
        if best:
            print(f"{arm:8} best valid within-limit child vs parent, median relative gain: "
                  f"{100 * statistics.median(best):+.2f}% over {len(best)} items")


if __name__ == "__main__":
    main()

"""Summarize the ACO seed-noise audit (results.jsonl from audit.py).

usage: PYTHONPATH=. uv run python -m experiments.diagnosis_aco_noise.analyze

- noise: the standard deviation of one program's training score across the eight seeds;
- luck: the search-seed score minus the mean of the seven other seeds, relative to that mean,
  for frontier programs (chosen because their search-seed score set a new best) and for
  controls (drawn at random);
- real progress: for each run, the seed-mean of the best-so-far program at the end versus
  at budget 250 and 500, next to the archived (search-seed) change.
Fitness is lower-is-better (CVRP fitness is the route length).
"""

import json
import statistics as st

from experiments.diagnosis_aco_noise.audit import OUT


def main():
    rows = [r for r in map(json.loads, open(OUT / "results.jsonl")) if r["failure"] is None and len(r["scores"]) == 8]
    out = {}
    for task in sorted({r["task"] for r in rows}):
        items = [r for r in rows if r["task"] == task]
        noise, luck = [], {"frontier": [], "control": []}
        for r in items:
            values = list(r["scores"].values())
            others = [v for s, v in r["scores"].items() if s != "730241"]
            mean = st.fmean(others)
            noise.append(st.stdev(values) / abs(st.fmean(values)))
            luck[r["kind"]].append((mean - r["scores"]["730241"]) / abs(mean))
        progress = []
        for run in sorted({r["run"] for r in items}):
            frontier = sorted((r for r in items if r["run"] == run and r["kind"] == "frontier"), key=lambda r: r["id"])
            if not frontier:
                continue
            def at(budget):
                before = [r for r in frontier if r["id"] <= budget]
                return before[-1] if before else None
            last = frontier[-1]
            for budget in (250, 500):
                start = at(budget)
                if start is None or start is last:
                    continue
                seed_mean = lambda r: st.fmean(v for s, v in r["scores"].items() if s != "730241")
                progress.append({"run": run, "from": budget,
                                 "archived_gain": (start["archived"] - last["archived"]) / abs(start["archived"]),
                                 "seed_mean_gain": (seed_mean(start) - seed_mean(last)) / abs(seed_mean(start))})
        out[task] = {
            "programs": len(items),
            "noise_sd_median": st.median(noise),
            "luck_frontier_mean": st.fmean(luck["frontier"]) if luck["frontier"] else None,
            "luck_control_mean": st.fmean(luck["control"]) if luck["control"] else None,
            "luck_frontier_n": len(luck["frontier"]), "luck_control_n": len(luck["control"]),
            "progress": progress,
        }
    print(json.dumps(out, indent=1))
    with open(OUT / "summary.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()

"""Summarize the Explore context replay by arm.

usage: PYTHONPATH=. .venv/bin/python -m experiments.diagnosis_explore_context.analyze
"""

import json
import re
import statistics
from collections import defaultdict

from traceaad.common.canonical import similarity
from .replay import OUT

MST = re.compile(r"MST|1-?tree|spanning|lower bound", re.I)
SCORE_FRAME = re.compile(r"^(Select|Choose|Pick)\b.*\b(minimi[sz]|lowest|smallest|score)", re.I)


def main():
    states = json.load(open(OUT / "states.json"))
    prompts = json.load(open(OUT / "prompts.json"))
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    parent_code = {}
    for name in states:
        text = prompts[f"{name}/A"]
        parent_code[name] = re.search(r"\[Current Algorithm\]\n.*?```python\n(.*?)```", text, re.S).group(1)
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    print("arm  n  valid  timeout  beat_parent  beat_best  MST/1-tree  score-frame  rel_change_median  sim_parent")
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        valid = [r for r in rs if r.get("status") == "valid"]
        beat = [r for r in valid if r["fitness"] < states[r["state"]]["parent_fitness"] - 1e-9]
        best = [r for r in valid if r["fitness"] < states[r["state"]]["search_best"] - 1e-9]
        ideas = [r.get("idea") or "" for r in rs if r.get("idea") is not None]
        rel = [(r["fitness"] - states[r["state"]]["parent_fitness"]) / states[r["state"]]["parent_fitness"] for r in valid]
        sims = [similarity(parent_code[r["state"]], r["code"]) for r in rs if r.get("code")]
        print(f"{arm:3s} {len(rs):3d} {len(valid)/len(rs):6.0%} {sum(r.get('status') == 'timeout' for r in rs):7d}"
              f" {len(beat):5d} ({len(beat)/len(rs):.1%}) {len(best):6d}"
              f" {sum(bool(MST.search(i)) for i in ideas)/len(ideas):10.0%} {sum(bool(SCORE_FRAME.search(i)) for i in ideas)/len(ideas):10.0%}"
              f" {statistics.median(rel):+17.2%} {statistics.median(sims):10.2f}")
    print("\nper state: MST share A/B/C, beat parent A/B/C")
    for name in states:
        cells = []
        for arm in "ABC":
            rs = [r for r in by_arm[arm] if r["state"] == name]
            ideas = [r.get("idea") or "" for r in rs]
            beat = sum(r.get("status") == "valid" and r["fitness"] < states[name]["parent_fitness"] - 1e-9 for r in rs)
            cells.append(f"{sum(bool(MST.search(i)) for i in ideas)}/{len(rs)} b{beat}")
        print(f"{name:20s} " + "  ".join(cells))


if __name__ == "__main__":
    main()

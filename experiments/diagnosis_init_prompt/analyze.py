"""Summarize the Init replay: validity, scores, computation and decision structure by arm.

usage: PYTHONPATH=. .venv/bin/python -m experiments.diagnosis_init_prompt.analyze

Decision structure is first labelled from the code and then checked by reading every
program the pattern flags: ``plan`` returns the first step of a route it built from the
current node; ``simulate`` scores candidates with a constructed completion; ``formula``
scores candidates with closed-form terms.
"""

import json
import re
import statistics
from collections import Counter, defaultdict

from .replay import OUT

PLAN = re.compile(r"(=|return)\s*(int\()?\s*[A-Za-z_]*(path|tour|perm|seq|route|plan|order)[A-Za-z_]*\s*\[\s*[01]\s*\]")
SIMULATE = re.compile(r"while\s+\w*(remaining|unvisited|rem|avail|left)\w*|2-?opt|two_opt|simulat|rollout|nearest.neighbo",
                      re.I)
N_INSTANCES = 16


def label(row):
    code, idea = row.get("code") or "", row.get("idea") or ""
    if PLAN.search(code):
        return "plan?"
    if SIMULATE.search(code) or SIMULATE.search(idea):
        return "simulate"
    return "formula"


def main():
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    print("arm  n  valid  timeout  best  median  per-instance CPU median  labels")
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        valid = [r for r in rs if r.get("status") == "valid"]
        scores = sorted(r["fitness"] for r in valid)
        cpu = [r["function_seconds"] / N_INSTANCES for r in valid if r.get("function_seconds") is not None]
        labels = Counter(label(r) for r in rs if r.get("code"))
        print(f"{arm:3s} {len(rs):3d} {len(valid)/len(rs):6.0%} {sum(r.get('status') == 'timeout' for r in rs):7d}"
              f" {scores[0]:6.3f} {statistics.median(scores):7.3f} {statistics.median(cpu):10.3f} s   {dict(labels)}")
    print("\nprograms flagged as plan?:")
    for r in rows:
        if r.get("code") and label(r) == "plan?":
            lines = [f"{i}: {line.strip()}" for i, line in enumerate(r["code"].splitlines(), 1)
                     if PLAN.search(line) or re.match(r"\s*if (n|len\()", line)]
            print(f"--- {r['arm']} {r['sample']} {r.get('status')} {r.get('fitness')}\n  {r.get('idea')}\n  " + "\n  ".join(lines[:8]))


if __name__ == "__main__":
    main()

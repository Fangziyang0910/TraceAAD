"""Summarize the three-arm E32 initialization comparison."""

import json
from collections import Counter, defaultdict
from statistics import mean

from .launch import SCHEDULE, run_dir


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def report_job(job):
    directory = run_dir(job)
    summary_path = directory / "logs" / "run_summary.json"
    if not summary_path.exists():
        return {**job, "status": "no_summary" if directory.exists() else "not_started"}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    heldout_path = directory / "heldout.json"
    heldout = json.loads(heldout_path.read_text(encoding="utf-8")) if heldout_path.exists() else {}
    events = read_jsonl(directory / "events.jsonl")
    roots = [event for event in events if event["operator"] == "Init"
             and event["status"] == "ok"]
    root_best = max(event["fitness"] for event in roots[:8]) if len(roots) >= 8 else None
    first_eight = roots[7] if len(roots) >= 8 else None
    first_eight_attempt = first_eight["candidate_id"] if first_eight else None
    first_eight_evaluation = first_eight["evaluation_id"] if first_eight else None
    attempted = summary.get("candidate_attempts", len(events))
    return {
        **job,
        "status": summary["status"],
        "candidate_attempts": attempted,
        "evaluation_calls": summary.get("evaluation_calls", summary.get("budget_used")),
        "valid_roots": len(roots),
        "root_best_train_fitness": root_best,
        "root_completion_attempt": first_eight_attempt,
        "root_completion_evaluation": first_eight_evaluation,
        "development_attempts": attempted - first_eight_attempt if first_eight_attempt else 0,
        "best_train_fitness": (summary.get("best") or {}).get("fitness"),
        "heldout_fitness": heldout.get("heldout_fitness"),
        "failures": dict(Counter(event["status"] for event in events
                                 if event["status"] != "ok")),
    }


def main():
    jobs = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    reports = [report_job(job) for job in jobs]
    print(f"initialization E32: planned={len(reports)} "
          f"status={dict(Counter(r['status'] for r in reports))}")
    groups = defaultdict(list)
    for report in reports:
        groups[(report["task"], report["mode"])].append(report)
    for task in sorted({job["task"] for job in jobs}):
        for mode in ("independent", "sequential", "hybrid"):
            rows = groups[(task, mode)]
            complete = [row for row in rows if row["status"] in
                        {"finished", "incomplete_initialization", "call_cap",
                         "generation_stall"}]
            initialized = sum(row["root_completion_attempt"] is not None for row in complete)
            def avg(field):
                values = [row[field] for row in complete if row.get(field) is not None]
                return round(mean(values), 4) if values else None

            print(f"  {task} {mode}: terminal={len(complete)}/{len(rows)} "
                  f"eight_roots={initialized}/{len(complete)} "
                  f"root_best={avg('root_best_train_fitness')} "
                  f"root_eval_cost={avg('root_completion_evaluation')} "
                  f"E32_best={avg('best_train_fitness')} "
                  f"heldout={avg('heldout_fitness')}")
    for task in sorted({job["task"] for job in jobs}):
        by_repeat = defaultdict(dict)
        for row in reports:
            if row["task"] == task:
                by_repeat[row["repeat"]][row["mode"]] = row
        matched = [block for block in by_repeat.values()
                   if all(block.get(mode, {}).get("heldout_fitness") is not None
                          for mode in ("independent", "sequential", "hybrid"))]
        if matched:
            differences = {
                mode: [block[mode]["heldout_fitness"] - block["sequential"]["heldout_fitness"]
                       for block in matched]
                for mode in ("independent", "hybrid")
            }
            print(f"  {task} matched held-out repeats={len(matched)} "
                  f"paired_differences_vs_sequential={differences}")


if __name__ == "__main__":
    main()

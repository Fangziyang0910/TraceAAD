"""Compute search diagnostics from committed program individuals."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
from traceaad.storage import write_json
from traceaad.config import DEFAULT_OPERATOR_WEIGHTS
from traceaad.history import final_program
from traceaad.selection import better
from traceaad.state import Facts

TRIED_OUT = 15
CLOCK = re.compile(
    "\\btime\\.(time|perf_counter|monotonic|process_time)\\s*\\(|\\bfrom\\s+time\\s+import\\b|\\bdatetime\\.now\\s*\\("
)


def computation_stats(facts, attempts, time_limit, final_candidates):
    """How much of the time limit programs use, and what each step does to it.

    For each step, the evaluation time of a valid new program relative to the
    program it started from; for the best program and the top ones, the share
    of the training time limit their evaluation took.
    """
    programs, archive = (facts.programs, facts.valid)
    ranking = sorted(archive.values(), key=lambda n: (n.fitness, n.id))
    share = lambda node: (
        node.eval_seconds / time_limit
        if time_limit and node.eval_seconds is not None
        else None
    )
    ratios = {}
    for attempt in attempts:
        if attempt.repair_of is not None or attempt.status != "valid":
            continue
        child, start = (attempt, programs.get(attempt.parent_id))
        if (
            start is None
            or not start.valid
            or (not start.eval_seconds)
            or (child.eval_seconds is None)
        ):
            continue
        ratios.setdefault(attempt.action, []).append(
            child.eval_seconds / start.eval_seconds
        )
    timeouts = Counter(
        (a.action for a in attempts if a.repair_of is None and a.status == "timeout")
    )
    top = ranking[:final_candidates]
    return {
        "time_limit": time_limit,
        "best_time_share": share(ranking[0]) if ranking else None,
        "best_function_seconds": ranking[0].function_seconds if ranking else None,
        "top_time_share_median": statistics.median(
            [s for s in map(share, top) if s is not None]
        )
        if any((share(n) is not None for n in top))
        else None,
        "time_ratio_to_start_median": {
            action: statistics.median(values) for action, values in ratios.items()
        },
        "time_ratio_to_start_above_5": {
            action: sum((v > 5 for v in values)) for action, values in ratios.items()
        },
        "timeouts_by_action": dict(timeouts),
        "clock_programs": sum((bool(CLOCK.search(n.code)) for n in archive.values())),
        "clock_programs_in_top": sum((bool(CLOCK.search(n.code)) for n in top)),
    }


def diagnostics(facts, budget, init_attempts, final_candidates=5, time_limit=None):
    programs, archive = (facts.programs, facts.valid)
    program_table = facts.programs
    attempts = sorted(program_table.values(), key=lambda a: a.id)
    counts = dict(Counter((a.status for a in attempts)))
    repairs = {a.repair_of: a for a in attempts if a.repair_of is not None}
    actions = {}
    for name in (*DEFAULT_OPERATOR_WEIGHTS, "Repair"):
        proposed = [a for a in attempts if a.action == name]
        new = [a for a in proposed if a.status == "valid"]
        started = [(a, programs.get(a.parent_id)) for a in new]
        improved = sum(
            (
                1
                for a, p in started
                if p is not None and p.valid and better(a.fitness, p.fitness)
            )
        )
        actions[name] = {
            "attempts": len(proposed),
            "new_valid": len(new),
            "improved_over_start": improved,
            "improvement_per_attempt": improved / len(proposed) if proposed else None,
        }
    frontier, frontiers = (math.inf, Counter())
    explore_late = 0
    for node in sorted(archive.values(), key=lambda n: n.id):
        if node.fitness < frontier:
            frontiers[node.action] += 1
            explore_late += node.action == "Explore" and node.id > 300
        frontier = min(frontier, node.fitness)
    tried, improved, on_tried_out = (Counter(), Counter(), 0)
    operator_attempts = [
        a for a in attempts if a.repair_of is None and a.action in DEFAULT_OPERATOR_WEIGHTS
    ]
    for a in operator_attempts:
        source = programs.get(a.parent_id)
        if source is None or not source.valid:
            continue
        on_tried_out += tried[source.id] >= TRIED_OUT and improved[source.id] == 0
        tried[source.id] += 1
        final = final_program(a, program_table)
        result = final
        if (
            final.status == "valid"
            and result is not None
            and better(result.fitness, source.fitness)
        ):
            improved[source.id] += 1
    quarters = []
    for q in range(4):
        group = [
            a
            for a in operator_attempts
            if q * budget // 4 < a.generation <= (q + 1) * budget // 4
        ]
        quarters.append(
            {
                "attempts": len(group),
                "timeout": sum((a.status == "timeout" for a in group)) / len(group)
                if group
                else None,
                "error": sum(
                    (
                        a.status
                        in {"runtime_error", "invalid_output", "invalid_source"}
                        for a in group
                    )
                )
                / len(group)
                if group
                else None,
            }
        )
    search = [e for e in facts.evaluations if e["role"] == "search"]
    cpu = sorted(
        (
            e["cpu_seconds"]
            for e in search
            if e["valid"] and isinstance(e.get("cpu_seconds"), (int, float))
        )
    )
    best = min(archive.values(), key=lambda n: (n.fitness, n.id)) if archive else None
    return {
        "status_counts": counts,
        "actions": actions,
        "repair_success": sum((a.status == "valid" for a in repairs.values())),
        "repair_attempts": len(repairs),
        "new_frontiers_by_action": dict(frontiers),
        "explore_new_frontiers_after_300": explore_late,
        "attempts_on_tried_out_programs": on_tried_out / len(operator_attempts)
        if operator_attempts
        else None,
        "failure_rates_by_quarter": quarters,
        "failed_programs": sum((not p.valid for p in programs.values())),
        "valid_cpu_seconds": {
            "median": cpu[len(cpu) // 2],
            "p95": cpu[int(len(cpu) * 0.95)],
            "max": cpu[-1],
        }
        if cpu
        else None,
        "best_training_depth": best.depth if best else None,
        "computation": computation_stats(facts, attempts, time_limit, final_candidates),
        "too_long_nodes": len((facts.state or {}).get("too_long", [])),
    }


def diagnose(run_dir):
    run_dir = Path(run_dir)
    facts = Facts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = (
        json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if (run_dir / "run_config.json").exists()
        else {}
    )
    if summary.get("method", config.get("method")) != "traceaad":
        raise ValueError(
            "diagnostics require a TraceAAD run; use saved diagnostics or historical source for older runs"
        )
    time_limit = (config.get("task_eval") or {}).get("timeout_seconds")
    instance_record = next(
        (
            r
            for r in facts.evaluations
            if r.get("role") == "search" and r.get("timeout_scope") == "instance"
        ),
        None,
    )
    if instance_record:
        time_limit = instance_record["timeout_seconds"] * instance_record["n_instances"]
    result = diagnostics(
        facts,
        summary["budget"],
        summary["init_attempts"],
        config.get("method_params", {}).get("final_candidates", 5),
        time_limit=time_limit,
    )
    write_json(run_dir / "diagnostics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.run_dir), indent=2))


if __name__ == "__main__":
    main()

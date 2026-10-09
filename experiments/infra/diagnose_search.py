"""Compute search and development diagnostics from committed facts."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from traceaad.common.storage import write_json
from traceaad.common.canonical import similarity
from traceaad.common.selection import OPERATORS
from traceaad.common.history import final_attempt
from traceaad.common.selection import better
from traceaad.common.state import Facts

TRIED_OUT = 15
# The generation steps each method samples; the others write Refine, Explore and Crossover.
METHOD_ACTIONS = {"v1018": ("Refine", "Explore", "Crossover", "Develop"),
                  "v1019": ("Refine", "Explore", "Crossover", "Develop"),
                  "v1020": ("Refine", "Explore", "Crossover", "Deepen"),
                  "v1021": ("Refine", "Explore", "Crossover"),
                  "v1022": ("Refine", "Explore", "Crossover")}
METHOD_ACTIONS["v1024"] = ("Refine", "Explore", "Crossover", "Develop")
# A program whose work depends on the wall clock: its result depends on the host and its load.
CLOCK = re.compile(r"\btime\.(time|perf_counter|monotonic|process_time)\s*\(|"
                   r"\bfrom\s+time\s+import\b|\bdatetime\.now\s*\(")


def computation_stats(facts, attempts, time_limit, final_candidates):
    """How much of the time limit programs use, and what each step does to it.

    For each step, the evaluation time of a valid new program relative to the
    program it started from; for the best program and the top ones, the share
    of the training time limit their evaluation took.
    """
    programs, archive = facts.programs, facts.valid
    ranking = sorted(archive.values(), key=lambda n: (n["fitness"], n["id"]))
    share = (lambda node: node["eval_seconds"] / time_limit
             if time_limit and node.get("eval_seconds") is not None else None)
    ratios = {}
    for attempt in attempts:
        if attempt.get("repair_of") is not None or attempt["status"] != "valid":
            continue
        child, start = programs.get(attempt["program_id"]), programs.get(attempt["parent_id"])
        if start is None or not start.get("valid") or not start.get("eval_seconds") or child.get("eval_seconds") is None:
            continue
        ratios.setdefault(attempt["action"], []).append(child["eval_seconds"] / start["eval_seconds"])
    timeouts = Counter(a["action"] for a in attempts if a.get("repair_of") is None and a["status"] == "timeout")
    top = ranking[:final_candidates]
    return {"time_limit": time_limit,
            "best_time_share": share(ranking[0]) if ranking else None,
            "best_function_seconds": ranking[0].get("function_seconds") if ranking else None,
            "top_time_share_median": statistics.median([s for s in map(share, top) if s is not None])
            if any(share(n) is not None for n in top) else None,
            "time_ratio_to_start_median": {action: statistics.median(values) for action, values in ratios.items()},
            "time_ratio_to_start_above_5": {action: sum(v > 5 for v in values) for action, values in ratios.items()},
            "timeouts_by_action": dict(timeouts),
            "clock_programs": sum(bool(CLOCK.search(n["code"])) for n in archive.values()),
            "clock_programs_in_top": sum(bool(CLOCK.search(n["code"])) for n in top)}


def diagnostics(facts, budget, init_attempts, final_candidates=5, method="v1017", time_limit=None):
    programs, archive = facts.programs, facts.valid
    attempts_table = facts.attempts
    attempts = sorted(attempts_table.values(), key=lambda a: a["id"])
    counts = dict(Counter(a["status"] for a in attempts))
    repairs = {a["repair_of"]: a for a in attempts if a.get("repair_of") is not None}
    actions = {}
    operators = METHOD_ACTIONS.get(method, ("Refine", "Explore", "Crossover"))
    for name in (*operators, "Repair"):
        proposed = [a for a in attempts if a["action"] == name]
        new = [a for a in proposed if a["status"] == "valid"]
        started = [(a, programs.get(a["parent_id"])) for a in new]
        improved = sum(1 for a, p in started if p is not None and p["valid"] and better(programs[a["program_id"]]["fitness"], p["fitness"]))
        actions[name] = {"attempts": len(proposed), "new_valid": len(new), "improved_over_start": improved,
                         "improvement_per_attempt": improved / len(proposed) if proposed else None}
    frontier, frontiers = math.inf, Counter()
    explore_late = 0
    for node in sorted(archive.values(), key=lambda n: n["id"]):
        if node["fitness"] < frontier:
            frontiers[node["action"]] += 1
            explore_late += node["action"] == "Explore" and node["id"] > 300
        frontier = min(frontier, node["fitness"])
    # How much of the budget went to programs that had already been tried
    # many times without improving (the waste experience should remove).
    tried, improved, on_tried_out = Counter(), Counter(), 0
    operator_attempts = [a for a in attempts if a["repair_of"] is None and a["action"] in OPERATORS]
    for a in operator_attempts:
        source = programs.get(a["parent_id"])
        if source is None or not source["valid"]:
            continue
        on_tried_out += tried[source["id"]] >= TRIED_OUT and improved[source["id"]] == 0
        tried[source["id"]] += 1
        final = final_attempt(a, attempts_table)
        result = programs.get(final.get("program_id"))
        if final["status"] == "valid" and result is not None and better(result["fitness"], source["fitness"]):
            improved[source["id"]] += 1
    quarters = []
    for q in range(4):
        group = [a for a in operator_attempts
                 if q * budget // 4 < a["id"] <= (q + 1) * budget // 4]
        quarters.append({"attempts": len(group),
                         "timeout": sum(a["status"] == "timeout" for a in group) / len(group) if group else None,
                         "error": sum(a["status"] in {"runtime_error", "invalid_output", "invalid_source"}
                                      for a in group) / len(group) if group else None})
    search = [e for e in facts.evaluations if e["role"] == "search"]
    cpu = sorted(e["cpu_seconds"] for e in search
                 if e["valid"] and isinstance(e.get("cpu_seconds"), (int, float)))
    best = min(archive.values(), key=lambda n: (n["fitness"], n["id"])) if archive else None
    return {"status_counts": counts, "actions": actions,
            "repair_success": sum(a["status"] == "valid" for a in repairs.values()),
            "repair_attempts": len(repairs),
            "crossover_copy_rate": counts.get("copied_reference", 0) / actions["Crossover"]["attempts"]
            if actions["Crossover"]["attempts"] else None,
            "new_frontiers_by_action": dict(frontiers),
            "explore_new_frontiers_after_300": explore_late,
            "attempts_on_tried_out_programs": on_tried_out / len(operator_attempts) if operator_attempts else None,
            "failure_rates_by_quarter": quarters,
            "failed_programs": sum(not p["valid"] for p in programs.values()),
            "valid_cpu_seconds": {"median": cpu[len(cpu) // 2], "p95": cpu[int(len(cpu) * .95)],
                                  "max": cpu[-1]} if cpu else None,
            "best_training_depth": best["depth"] if best else None,
            "explorations": exploration_stats(facts, attempts, init_attempts, final_candidates, method),
            "computation": computation_stats(facts, attempts, time_limit, final_candidates),
            "too_long_nodes": len((facts.state or {}).get("too_long", []))}

def random_development_stats(facts, attempts, init_attempts, final_candidates):
    """How development used the budget, what it added and where it led."""
    programs = facts.programs
    records = sorted(facts.explorations.values(), key=lambda e: e["id"])
    tagged = {a["id"] for a in attempts if a.get("exploration") and a["exploration"]["step"] > 0}
    inside = [a for a in attempts if a["id"] in tagged or a.get("repair_of") in tagged]
    search = [a for a in attempts if a["id"] > init_attempts]  # initialization comes first
    developed = [e for e in records if e.get("developed")]
    gains, moved = [], []
    for e in developed:
        first, reached = programs[e["proposed_id"]], programs[e["best_id"]]
        gains.append((first["fitness"] - reached["fitness"]) / max(abs(first["fitness"]), 1e-12))
        start = programs.get(e["start_id"])
        if start is not None and reached["id"] != first["id"]:
            moved.append(similarity(reached["code"], start["code"]) - similarity(first["code"], start["code"]))
    produced = {a.get("program_id") for a in inside if a["status"] == "valid"}
    top = [n["id"] for n in sorted(facts.valid.values(), key=lambda n: (n["fitness"], n["id"]))][:final_candidates]
    # V10.20 opens explorations from Explore and Deepen proposals.
    by_action = {}
    for e, gain in zip(developed, gains):
        by_action.setdefault(facts.attempts[e["proposal_attempt"]]["action"], []).append(gain)
    proposals = Counter(facts.attempts[e["proposal_attempt"]]["action"] for e in records)
    return {"count": len(records), "new_programs": sum(e["proposed_id"] is not None for e in records),
            "by_proposal_action": {action: {"count": proposals[action],
                                            "developed": len(by_action.get(action, [])),
                                            "development_improved": sum(g > 0 for g in by_action.get(action, []))}
                                   for action in sorted(proposals)},
            "developed": len(developed),
            "development_generations": len(inside),
            "development_share_of_search_generations": len(inside) / len(search) if search else None,
            "development_gain_median": statistics.median(gains) if gains else None,
            "development_improved": sum(g > 0 for g in gains),
            "similarity_to_start_change_median": statistics.median(moved) if moved else None,
            "top_programs_from_explorations": sum(i in produced for i in top)}



def v1018_development_stats(facts, attempts, init_attempts, final_candidates):
    programs = facts.programs
    ranking = [p["id"] for p in sorted(facts.valid.values(), key=lambda p: (p["fitness"], p["id"]))]
    """How development used the budget, what it changed and where its designs ended."""
    records = sorted(facts.explorations.values(), key=lambda e: e["id"])
    tagged = {a["id"] for a in attempts if a.get("exploration") and a["exploration"]["step"] > 0}
    inside = [a for a in attempts if a["id"] in tagged or a.get("repair_of") in tagged]
    proposals = [a for a in attempts if a.get("exploration") and a["exploration"]["step"] == 0]
    search = [a for a in attempts if a["id"] > init_attempts]  # initialization comes first
    developed = [e for e in records if e["development_attempts"]]
    gains, moved = [], []
    for e in developed:
        first, reached = programs[e["proposed_id"]], programs[e["best_id"]]
        gains.append((first["fitness"] - reached["fitness"]) / max(abs(first["fitness"]), 1e-12))
        start = programs.get(e["start_id"])
        if start is not None and reached["id"] != first["id"]:
            moved.append(similarity(reached["code"], start["code"]) - similarity(first["code"], start["code"]))
    produced = {a.get("program_id") for a in inside if a["status"] == "valid"}
    proposed = {e["proposed_id"] for e in records if e["proposed_id"] is not None}
    top = ranking[:final_candidates]
    steps = [len(e["development_attempts"]) for e in developed]
    return {"count": len(records), "proposal_generations": len(proposals),
            "new_programs": len(proposed), "developed": len(developed),
            "reasons": dict(Counter(e["reason"] for e in records)),
            "competitive_at_proposal": sum(e["reason"] == "competitive" and not e["development_attempts"]
                                           for e in records),
            "competitive_after_development": sum(e["reason"] == "competitive" and bool(e["development_attempts"])
                                                 for e in records),
            "development_steps_mean": statistics.fmean(steps) if steps else None,
            "development_generations": len(inside),
            "development_share_of_search_generations": len(inside) / len(search) if search else None,
            "proposal_share_of_search_generations": len(proposals) / len(search) if search else None,
            "development_gain_median": statistics.median(gains) if gains else None,
            "development_improved": sum(g > 0 for g in gains),
            "similarity_to_start_change_median": statistics.median(moved) if moved else None,
            "top_programs_from_development": sum(i in produced for i in top),
            "top_programs_from_proposals": sum(i in proposed for i in top)}


def v1019_development_stats(facts, attempts, init_attempts, final_candidates):
    programs = facts.programs
    ranking = [p["id"] for p in sorted(facts.valid.values(), key=lambda p: (p["fitness"], p["id"]))]
    """How changes used the budget, what their development did and where they ended."""
    records = sorted(facts.explorations.values(), key=lambda e: e["id"])
    tagged = {a["id"] for a in attempts if a.get("exploration") and a["exploration"]["step"] > 0}
    inside = [a for a in attempts if a["id"] in tagged or a.get("repair_of") in tagged]
    proposals = [a for a in attempts if a.get("exploration") and a["exploration"]["step"] == 0]
    search = [a for a in attempts if a["id"] > init_attempts]  # initialization comes first
    made = [e for e in records if e["proposed_id"] is not None]
    developed = [e for e in made if e["development_attempts"]]
    by_steps = Counter()
    source_similarity, first_similarity, kept = [], [], []
    for e in developed:
        source, best = programs[e["start_id"]], programs[e["best_id"]]
        if e["improved_start"]:
            by_steps[len(e["development_attempts"])] += 1
            first = programs[e["proposed_id"]]
            # Where the successful version sits between the algorithm and the change's first version.
            source_similarity.append(similarity(best["code"], source["code"]))
            first_similarity.append(similarity(best["code"], first["code"]))
    for e in made:
        first, source = programs[e["proposed_id"]], programs.get(e["start_id"])
        if source is not None and source.get("valid"):
            kept.append(similarity(first["code"], source["code"]))
    starts = {e["id"]: e["start_id"] for e in records}
    reverted = sum(1 for a in inside if a.get("exploration") and a["status"] == "duplicate"
                   and a.get("program_id") == starts.get(a["exploration"]["id"]))
    produced = {a.get("program_id") for a in inside if a["status"] == "valid"}
    proposed = {e["proposed_id"] for e in made}
    top = ranking[:final_candidates]
    steps = [len(e["development_attempts"]) for e in developed]
    median = lambda values: statistics.median(values) if values else None
    return {"count": len(records), "proposal_generations": len(proposals),
            "new_programs": len(made), "developed": len(developed),
            "reasons": dict(Counter(e["reason"] for e in records)),
            "first_version_improved_start": sum(e["improved_start"] and not e["development_attempts"]
                                                for e in made),
            "developed_improved_start": sum(e["improved_start"] for e in developed),
            "developed_improved_start_by_steps": dict(sorted(by_steps.items())),
            "development_steps_mean": statistics.fmean(steps) if steps else None,
            "development_generations": len(inside),
            "development_returned_start_code": reverted,
            "development_share_of_search_generations": len(inside) / len(search) if search else None,
            "proposal_share_of_search_generations": len(proposals) / len(search) if search else None,
            "first_version_similarity_to_start_median": median(kept),
            "improved_development_similarity_to_start_median": median(source_similarity),
            "improved_development_similarity_to_first_median": median(first_similarity),
            "top_programs_from_development": sum(i in produced for i in top),
            "top_programs_from_proposals": sum(i in proposed for i in top)}


def exploration_stats(facts, attempts, init_attempts, final_candidates, method):
    if method == "v1024":
        from traceaad.v10_24.diagnostics import development_stats
        return development_stats(facts)
    if method == "v1018":
        return v1018_development_stats(facts, attempts, init_attempts, final_candidates)
    if method == "v1019":
        return v1019_development_stats(facts, attempts, init_attempts, final_candidates)
    return random_development_stats(facts, attempts, init_attempts, final_candidates)


def diagnose(run_dir):
    run_dir = Path(run_dir)
    facts = Facts(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").exists() else {}
    time_limit = (config.get("task_eval") or {}).get("timeout_seconds")
    instance_record = next((r for r in facts.evaluations if r.get("role") == "search"
                            and r.get("timeout_scope") == "instance"), None)
    if instance_record:
        time_limit = instance_record["timeout_seconds"] * instance_record["n_instances"]
    result = diagnostics(facts, summary["budget"], summary["init_attempts"],
                         config.get("method_params", {}).get("final_candidates", 5),
                         summary.get("method", config.get("method", "v1017")),
                         time_limit)
    write_json(run_dir / "diagnostics.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(diagnose(args.run_dir), indent=2))


if __name__ == "__main__":
    main()

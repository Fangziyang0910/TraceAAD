"""Export empirical outcome distributions from complete matched replay cells."""

import experiments  # noqa: F401
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from benchmarks.tasks import training_task
from traceaad.common.state import Facts
from traceaad.common.storage import append_jsonl, committed_rows, write_json
from .collect import METHODS, prompt_identity
from .data import ACTIONS, DEFAULT_SUPPORT, delivered_request, outcome, questions, request_state


def empirical_gold(trials, horizon):
    names = ("repair_used",) if horizon == 2 else ("new_valid", "improve_parent")
    return {
        name: {"probabilities": {"true": sum(t["gold"][str(horizon)][name] for t in trials) / len(trials)}}
        for name in names
    } | {"frontier_gain": {"probabilities": {
        str(i): sum(t["gold"][str(horizon)]["frontier_gain"] == i for t in trials) / len(trials)
        for i in range(5)}}}


def past_tables(facts, cutoff):
    programs, attempts = {}, {}
    for row in committed_rows(facts.run_dir):
        if row.get("kind") == "candidate":
            if row["attempt"]["id"] > cutoff:
                break
            attempts[row["attempt"]["id"]] = row["attempt"]
        if node := row.get("program"):
            programs[node["id"]] = {**node, "code": facts.sources.get(node["key"])}
    return programs, attempts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("collection", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--request-only", action="store_true")
    parser.add_argument("--splits", nargs="+", choices=("train", "calibration", "test"),
                        default=("train", "calibration", "test"))
    args = parser.parse_args()
    horizons = (2,) if args.request_only else (1, 4)
    if args.output.exists():
        parser.error("choose a new output directory")
    metadata = json.loads((args.dataset / "metadata.json").read_text())
    manifest = json.loads((args.collection / "manifest.json").read_text())
    policy = manifest["prompt_policy"]
    if manifest["prompt_sources_sha256"] != prompt_identity(policy):
        raise ValueError("export prompt sources differ from the measured collection")
    if manifest["states_sha256"] != hashlib.sha256((args.dataset / "states.jsonl").read_bytes()).hexdigest():
        raise ValueError("export states differ from the measured collection")
    method = METHODS[policy]
    states = [json.loads(line) for line in (args.dataset / "states.jsonl").read_text().splitlines()]
    groups = defaultdict(list)
    for path in (args.collection / "chains").rglob("result.json"):
        r = json.loads(path.read_text())
        s = r["state"]
        if args.request_only:
            attempts, programs = {}, {}
            for record in committed_rows(path.parent):
                attempt = record.get("attempt")
                if attempt is None or attempt["id"] <= s["cutoff"]:
                    continue
                attempts[attempt["id"]] = attempt
                if program := record.get("program"):
                    programs[program["id"]] = program
                if attempt["id"] >= s["cutoff"] + 2:
                    break
            first_id = s["cutoff"] + 1
            delivery = delivered_request(attempts[first_id], attempts.get(first_id + 1), programs, s["cutoff"] + 4)
            if delivery is None:
                raise ValueError("completed replay has an unfinished initial request")
            labels, gain = outcome(s["parent_fitness"], s["frontier_fitness"], delivery["fitnesses"],
                                   metadata["score_scales"][s["task"]])
            r["gold"]["2"] = {"frontier_gain": labels["frontier_gain"], "repair_used": delivery["candidate_cost"] == 2}
            r["normalized_gain"]["2"] = gain
            r["request_candidate_cost"] = delivery["candidate_cost"]
        groups[s["run_id"], s["state_id"], r["action"]].append(r)
    rows, rewards, comparisons, facts_cache, builders = defaultdict(list), defaultdict(list), [], {}, {}
    for state in states:
        if state["split"] not in args.splits:
            continue
        task, run_id = state["task"], state["run_id"]
        if run_id not in facts_cache:
            facts_cache[run_id] = Facts(args.dataset / state["source"])
        if task not in builders:
            evaluation, _ = training_task(task, condition="traceaad")
            builders[task] = (method.PromptBuilder(None, task, evaluation, {}, {}, method.Config()),
                              str(evaluation.template_program))
        builder, template = builders[task]
        programs, attempts = past_tables(facts_cache[run_id], state["cutoff"])
        comparison = {"run_id": run_id, "state_id": state["state_id"], "task": task,
                      "split": state["split"], "actions": {}, "candidate_costs": {}}
        for action in ACTIONS:
            trials = sorted(groups[run_id, state["state_id"], action], key=lambda t: t["repeat"])
            if [t["repeat"] for t in trials] != list(range(manifest["repeats"])):
                raise ValueError(f"incomplete or duplicated cell: {run_id}/{state['state_id']}/{action}")
            if any(t["state"] != state or t["candidates"] != 4 or t["prompt_policy"] != policy for t in trials):
                raise ValueError("state identity or candidate budget mismatch")
            comparison["actions"][action] = {str(h): [t["normalized_gain"][str(h)] for t in trials] for h in horizons}
            comparison["candidate_costs"][action] = {
                str(h): [t["request_candidate_cost"] if h == 2 else h for t in trials] for h in horizons}
            for horizon in horizons:
                model_state = request_state(task, builder.common[0], template, programs, attempts,
                    state["parent_id"], state["reference_id"], action, state["cutoff"], state["search_budget"],
                    horizon=horizon, score_scale=metadata["score_scales"][task], prompt_policy=policy)
                rows[state["split"]].append({"run_id": run_id,
                    "state_id": f"{state['state_id']}/{action}/h{horizon}", "state": model_state,
                    "questions": questions(horizon), "gold": empirical_gold(trials, horizon),
                    "measurement": {"repeats": len(trials), "normalized_gains": comparison["actions"][action][str(horizon)],
                                    "candidate_costs": comparison["candidate_costs"][action][str(horizon)]}})
                if state["split"] == "train":
                    for trial in trials:
                        rewards[horizon, task, trial["gold"][str(horizon)]["frontier_gain"]].append(trial["normalized_gain"][str(horizon)])
        comparisons.append(comparison)
    args.output.mkdir(parents=True)
    for split in args.splits:
        if not rows[split]:
            raise ValueError(f"empty {split}")
        for row in rows[split]:
            append_jsonl(args.output / f"{split}.jsonl", row)
    for comparison in comparisons:
        append_jsonl(args.output / "comparisons.jsonl", comparison)
    metadata.update(source_prompt_policy=metadata["prompt_policy"], prompt_policy=policy,
        prompt_sources_sha256=manifest["prompt_sources_sha256"],
        horizons=list(horizons), decision_horizon=2 if args.request_only else 1,
        decision_unit="ordinary operator request including its automatic Repair" if args.request_only else "fixed candidate block",
        selection_objective="expected frontier gain per expected candidate cost",
        trained_questions=list(questions(horizons[0])), rows={s: len(values) for s, values in rows.items()},
        label_source="four independent observed trials per action and state", repeats=manifest["repeats"],
        reward_support={str(h): {task: [sum(rewards[h, task, i]) / len(rewards[h, task, i])
                        if rewards[h, task, i] else DEFAULT_SUPPORT[i] for i in range(5)]
                        for task in builders} for h in horizons},
        sha256={f"{s}.jsonl": hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest() for s in rows})
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps({"states": len(states), "rows": metadata["rows"]}, indent=2))


if __name__ == "__main__":
    main()

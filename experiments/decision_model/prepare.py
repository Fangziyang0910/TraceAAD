"""Freeze compatible search records and export measured decision outcomes."""

import experiments  # noqa: F401
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from benchmarks.tasks import CO_TASKS, training_task
from traceaad.common.config import REVISION
from traceaad.common.selection import choose_reference
from traceaad.common.state import Facts
from traceaad.common.storage import Programs, append_jsonl, committed_rows, write_json
from .collect import METHODS, prompt_identity
from .data import ACTIONS, DEFAULT_SUPPORT, gain_level, outcome, questions, request_state

SPLITS = ("train", "calibration", "test")


def compatible(config, prompt_policy="v1022"):
    execution = config.get("evaluation_execution", {})
    return (config.get("revision") == REVISION and config.get("method") == prompt_policy
            and config.get("task") in CO_TASKS and config.get("objective") == "min"
            and execution.get("function_seconds") == 2 and execution.get("timeout_seconds") == 20
            and config.get("method_params", {}).get("evaluation_seeds") == [730241])


def freeze(source, target, prompt_policy="v1022"):
    config = json.loads((source / "run_config.json").read_text())
    if not compatible(config, prompt_policy):
        raise ValueError(f"incompatible source: {source}")
    facts = Facts(source)
    cutoff = max(facts.attempts, default=0)
    if not cutoff:
        raise ValueError(f"empty search: {source}")
    target.mkdir(parents=True)
    sources = Programs(target)
    for program in facts.programs.values():
        if not program.get("code") or sources.add(program["code"]) != program["key"]:
            raise ValueError(f"missing or mismatched source: {source}/{program['id']}")
    for row in committed_rows(source):
        if row.get("kind") == "candidate" and row["attempt"]["id"] > cutoff:
            break
        # Timings for sixteen instances are retained in the original archive.
        row = {k: v for k, v in row.items() if k != "evaluations"}
        append_jsonl(target / "events.jsonl", row)
    write_json(target / "run_config.json", config)
    write_json(target / "snapshot.json", {"source": str(source.resolve()), "last_attempt": cutoff,
        "files_sha256": {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                         for name in ("events.jsonl", "programs.jsonl", "run_config.json")}})
    return config, Facts(target)


def records_for_run(task, run_id, config, facts, task_text, template, score_scale, prompt_policy="v1022"):
    programs, attempts = {}, {}
    for row in committed_rows(facts.run_dir):
        if row.get("kind") != "candidate":
            if node := row.get("program"):
                programs[node["id"]] = {**node, "code": facts.sources.get(node["key"])}
            continue
        attempt = row["attempt"]
        parent = programs.get(attempt.get("parent_id"))
        if attempt["action"] in ACTIONS and not attempt.get("repair_of") and parent and parent["valid"]:
            phase = "development" if (attempt.get("exploration") or {}).get("step", 0) > 0 else "ordinary"
            state = request_state(task, task_text, template, programs, attempts, parent["id"],
                                  attempt.get("reference_id"), attempt["action"], attempt["id"] - 1,
                                  config["budget"], phase=phase, score_scale=score_scale, prompt_policy=prompt_policy)
            new_node = row.get("program") if attempt["status"] == "valid" else None
            if new_node is not None and not new_node["valid"]:
                raise ValueError("valid attempt references an invalid program")
            gold, reward = outcome(parent["fitness"], state["frontier_score_before"],
                                   [new_node["fitness"]] if new_node else [], score_scale)
            yield {"run_id": run_id, "state_id": f"before-{attempt['id']}", "state": state,
                   "questions": questions(1), "gold": gold,
                   "measurement": {"normalized_gain": reward, "status": attempt["status"],
                                   "parent_id": parent["id"], "reference_id": attempt.get("reference_id"),
                                   "attempt_id": attempt["id"], "cutoff": attempt["id"] - 1}}
        if node := row.get("program"):
            programs[node["id"]] = {**node, "code": facts.sources.get(node["key"])}
        attempts[attempt["id"]] = attempt


def choose_states(records, count, rng, included=()):
    ordinary = [r for r in records if r["state"]["decision_phase"] == "ordinary"]
    by_id = {r["state_id"]: r for r in ordinary}
    selected = [by_id[name] for name in included if name in by_id]
    chosen = {r["state_id"] for r in selected}
    parent_counts = Counter(r["state"]["parent"]["code"] for r in selected)
    strata = defaultdict(list)
    for row in ordinary:
        state = row["state"]
        gap = (state["parent"]["fitness"] - state["frontier_score_before"]) / max(abs(state["frontier_score_before"]), 1)
        strata[(state["attempts_used"] // 250, gap <= 0.005)].append(row)
    for values in strata.values():
        rng.shuffle(values)
    while len(selected) < count:
        added = False
        for values in strata.values():
            while values:
                row = values.pop()
                parent_key = row["state"]["parent"]["code"]
                if row["state_id"] in chosen or parent_counts[parent_key] >= 4:
                    continue
                selected.append(row)
                chosen.add(row["state_id"])
                parent_counts[parent_key] += 1
                added = True
                break
            if len(selected) == count:
                break
        if not added:
            raise ValueError(f"only {len(selected)} distinct usable states, requested {count}")
    return selected


def reward_support(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        measured = row["measurement"]
        gains = measured["normalized_gains"] if "normalized_gains" in measured else [measured["normalized_gain"]]
        for gain in gains:
            groups[row["state"]["task"]][gain_level(gain)].append(gain)
    return {task: [sum(levels[i]) / len(levels[i]) if levels[i] else DEFAULT_SUPPORT[i]
                   for i in range(5)] for task, levels in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sources", nargs="+", type=Path, required=True)
    parser.add_argument("--states-per-task", type=int, default=12)
    parser.add_argument("--split-map", type=Path)
    parser.add_argument("--include-states", type=Path)
    parser.add_argument("--score-scales", type=Path)
    parser.add_argument("--prompt-policy", choices=tuple(METHODS), default="v1022")
    args = parser.parse_args()
    if args.output.exists() or args.states_per_task < 3:
        parser.error("choose a new output directory and at least three states per task")
    known_splits = json.loads(args.split_map.read_text()) if args.split_map else {}
    score_scales = json.loads(args.score_scales.read_text())["score_scales"] if args.score_scales else {}
    included = defaultdict(list)
    included_details = {}
    if args.include_states:
        for line in args.include_states.read_text().splitlines():
            r = json.loads(line)
            included[r["run_id"]].append(r["state_id"])
            included_details[r["run_id"], r["state_id"]] = r
    sources = defaultdict(list)
    for root in args.sources:
        for path in sorted(root.glob("*/*/run_config.json")):
            config = json.loads(path.read_text())
            if compatible(config, args.prompt_policy):
                sources[config["task"]].append(path.parent)
    if any(len(sources[task]) < 3 for task in CO_TASKS):
        raise ValueError("need at least three compatible independent runs per task")
    args.output.mkdir(parents=True)
    rows, states, split_map, seen_states, source_audit = {s: [] for s in SPLITS}, [], {}, set(), []
    for task in CO_TASKS:
        evaluation, _ = training_task(task, condition="traceaad")
        method = METHODS[args.prompt_policy]
        builder = method.PromptBuilder(None, task, evaluation, {}, {}, method.Config())
        candidates = sources[task]
        if len({p.name for p in candidates}) != len(candidates):
            raise ValueError(f"duplicate source identity for {task}")
        counts = Counter()
        for index, source in enumerate(candidates):
            run_id = f"{task}/{source.name}"
            default_split = "train" if known_splits else SPLITS[index] if index < 3 else "train"
            split_map[run_id] = known_splits.get(run_id, default_split)
            if split_map[run_id] not in SPLITS:
                raise ValueError("invalid split map")
            counts[split_map[run_id]] += 1
        if any(counts[s] == 0 for s in SPLITS):
            raise ValueError(f"missing split for {task}")
        # Initial three runs: 1/1/1. With two added training runs: 60/20/20 states.
        quota = {s: args.states_per_task * counts[s] // len(candidates) for s in SPLITS}
        quota["train"] += args.states_per_task - sum(quota.values())
        ordinal = Counter()
        frozen = {}
        for source in candidates:
            target = args.output / "snapshots" / task / source.name
            frozen[source.name] = freeze(source, target, args.prompt_policy)
        if task not in score_scales:
            roots = [p["fitness"] for source in candidates if split_map[f"{task}/{source.name}"] == "train"
                     for p in frozen[source.name][1].valid.values() if p["action"] == "Init"]
            score_scales[task] = (100.0 if task in ("fssp_gls", "graph_colouring", "jssp_construct")
                                  else max(1.0, statistics.median(abs(x) for x in roots)))
        for source in candidates:
            run_id = f"{task}/{source.name}"
            split = split_map[run_id]
            target = args.output / "snapshots" / task / source.name
            config, facts = frozen[source.name]
            exported = list(records_for_run(task, run_id, config, facts, builder.common[0],
                                           str(evaluation.template_program), score_scales[task], args.prompt_policy))
            for row in exported:
                digest = hashlib.sha256(json.dumps(row["state"], sort_keys=True).encode()).hexdigest()
                if digest not in seen_states:
                    rows[split].append(row)
                    seen_states.add(digest)
            count = quota[split] // counts[split] + (ordinal[split] < quota[split] % counts[split])
            ordinal[split] += 1
            for record in choose_states(exported, count, random.Random(f"decision-20261009:{run_id}"), included[run_id]):
                m = record["measurement"]
                past = {i: p for i, p in facts.programs.items() if i <= m["cutoff"]}
                parent = past[m["parent_id"]]
                original = included_details.get((run_id, record["state_id"]))
                reference = past.get(original["reference_id"] if original else m["reference_id"])
                if reference is None:
                    reference, _ = choose_reference(parent, {i: p for i, p in past.items() if p["valid"]},
                                                   random.Random(f"decision-ref:{run_id}:{m['cutoff']}"))
                if reference is None:
                    raise ValueError("selected state has no valid Crossover reference")
                states.append({"run_id": run_id, "state_id": record["state_id"], "task": task,
                    "split": split, "source": str(target.relative_to(args.output)), "cutoff": m["cutoff"],
                    "parent_id": parent["id"], "reference_id": reference["id"], "frontier_fitness": record["state"]["frontier_score_before"],
                    "parent_fitness": parent["fitness"], "search_budget": config["budget"]})
            source_audit.append({"run_id": run_id, "split": split, "rows": len(exported),
                                 "last_attempt": max(facts.attempts)})
    for split, values in rows.items():
        for row in values:
            append_jsonl(args.output / f"{split}.jsonl", row)
    for state in states:
        append_jsonl(args.output / "states.jsonl", state)
    write_json(args.output / "split_map.json", split_map)
    metadata = {"format": "traceaad-decision-outcomes-v1", "revision": REVISION,
                "prompt_policy": args.prompt_policy, "prompt_sources_sha256": prompt_identity(args.prompt_policy),
                "horizons": [1], "sources": source_audit,
                "score_scales": score_scales,
                "rows": {s: len(values) for s, values in rows.items()}, "states": len(states),
                "reward_support": {"1": reward_support(rows["train"])},
                "sha256": {f"{s}.jsonl": hashlib.sha256((args.output / f"{s}.jsonl").read_bytes()).hexdigest() for s in SPLITS}}
    write_json(args.output / "metadata.json", metadata)
    print(json.dumps({k: metadata[k] for k in ("rows", "states", "reward_support")}, indent=2))


if __name__ == "__main__":
    main()

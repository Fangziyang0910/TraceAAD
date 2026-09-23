"""Prepare, run, profile and analyze randomized E2-B' two-step paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from experiments.infra.base import BACKENDS, REPO_ROOT, build_llm_client, build_task, resolve_backend
from experiments.traceaad_refine_e1.profile import candidate as profile_candidate
from experiments.traceaad_refine_e1.profile import worker as profile_worker
from experiments.traceaad_refine_e1.profile_core import profile_distance
from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from traceaad.v10_3.schema import Node
from traceaad.v10_6 import prompts
from traceaad.v10_6.traceaad import CODE_RE, SUMMARY_RE, _strip_thinking

PROTOCOL = "traceaad-e2-b-prime-v1"
SOURCE = REPO_ROOT / "experiments/traceaad_e2_a/raw/traceaad_e2_a_20260907"
DEFAULT = REPO_ROOT / "experiments/traceaad_e2_b/raw/traceaad_e2_b_20260907"
CONFIG_PATH = Path(__file__).with_name("e2b_config.json")
TASKS = ("tsp_construct", "online_bin_packing", "vrptw_construct")
STATES = ("experienced", "fresh")
ARMS = {"RR": ("Refine", "Refine"), "PR": ("Pivot", "Refine")}
OUTPUT_TOKENS = 16384
PROMPT_LIMIT = 32768 - OUTPUT_TOKENS - 256
HISTORY_TOKENS = 8192
SUMMARY_TOKENS = 1024
MAX_EVENTS = 8
TRANSPORT_RETRIES = 3


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def node(row: dict[str, Any]) -> Node:
    keys = ("id", "code", "idea", "fitness", "evaluation_id", "parent_id", "operator", "donor_id")
    return Node(**{key: row.get(key) for key in keys})


def depth(nodes: dict[int, dict[str, Any]], node_id: int) -> int:
    value = 0
    parent = nodes[node_id].get("parent_id")
    while parent is not None:
        value += 1
        parent = nodes[parent].get("parent_id")
    return value


def run_pool(run: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    folder = SOURCE / "snapshot" / run["run_name"]
    rows = read_json(folder / "tree_state.json")["nodes"]
    nodes = {int(row["id"]): row for row in rows}
    events = read_jsonl(folder / "events.jsonl")
    born = {int(event["node_id"]): int(event["candidate_id"]) for event in events if event.get("node_id") is not None}
    ids = read_json(SOURCE / "distances" / run["run_name"] / "ids.json")
    index = {int(node_id): i for i, node_id in enumerate(ids)}
    behavior = np.load(SOURCE / "distances" / run["run_name"] / "behavior.npy")
    attempts: dict[int, list[dict[str, Any]]] = defaultdict(list)
    child_metrics: dict[int, list[tuple[float, float]]] = defaultdict(list)
    all_edges: list[tuple[float, float, str]] = []
    for event in events:
        parent_id = event.get("parent_id")
        if parent_id is None:
            continue
        parent_id = int(parent_id)
        attempts[parent_id].append(event)
        child_id = event.get("node_id")
        if child_id not in index or parent_id not in index:
            continue
        earlier = [old for old in ids if born[int(old)] < event["candidate_id"] and int(old) != parent_id]
        revisit = min((behavior[index[int(child_id)], index[int(old)]] for old in earlier), default=math.nan)
        movement = float(behavior[index[parent_id], index[int(child_id)]])
        child_metrics[parent_id].append((movement, float(revisit)))
        all_edges.append((movement, float(revisit), event["operator"]))
    move_median = float(np.nanmedian([edge[0] for edge in all_edges]))
    revisit_median = float(np.nanmedian([edge[1] for edge in all_edges]))
    pivot_move_median = float(np.nanmedian([edge[0] for edge in all_edges if edge[2] == "Pivot"]))
    ranked_rows = sorted(rows, key=lambda item: (item["fitness"], item["id"]))
    ranks = {row["id"]: rank / max(1, len(rows) - 1) for rank, row in enumerate(ranked_rows)}
    pool = []
    for row in rows:
        node_id = int(row["id"])
        local = child_metrics[node_id]
        history = attempts[node_id]
        recent = history[-5:]
        no_recent_gain = bool(recent) and all(
            not event.get("parent_improved") and not event.get("frontier_improved") for event in recent
        )
        median_move = float(np.median([item[0] for item in local])) if local else None
        revisit_rate = float(np.mean([item[1] <= revisit_median for item in local])) if local else None
        state = None
        if (
            len(history) >= 3
            and no_recent_gain
            and len(local) >= 2
            and median_move <= move_median
            and revisit_rate >= 0.5
        ):
            state = "experienced"
        elif not history:
            state = "fresh"
        if state is None:
            continue
        lineage = []
        parent_id = row.get("parent_id")
        while parent_id is not None:
            lineage.append(nodes[int(parent_id)])
            parent_id = nodes[int(parent_id)].get("parent_id")
        pool.append(
            {
                "task": run["task"],
                "run_name": run["run_name"],
                "repeat": run["repeat"],
                "state": state,
                "parent": row,
                "ancestors": lineage,
                "anchor_fitness": float(row["fitness"]),
                "fitness_rank": ranks[node_id],
                "depth": depth(nodes, node_id),
                "development_attempts": len(history),
                "recent_attempts_checked": len(recent),
                "recent_parent_or_frontier_gains": sum(
                    bool(event.get("parent_improved") or event.get("frontier_improved")) for event in recent
                ),
                "valid_behavior_children": len(local),
                "median_child_movement": median_move,
                "child_revisit_rate": revisit_rate,
                "run_move_median": move_median,
                "run_revisit_median": revisit_median,
                "historical_pivot_move_median": pivot_move_median,
            }
        )
    return pool, {
        "move_median": move_median,
        "revisit_median": revisit_median,
        "pivot_move_median": pivot_move_median,
    }


def select_anchors(seed: int, per_state: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    manifest = read_json(SOURCE / "snapshot.json")
    selected, audit = [], {}
    for task in TASKS:
        task_runs = [run for run in manifest["runs"] if run["task"] == task]
        pools = []
        for run in task_runs:
            pool, thresholds = run_pool(run)
            pools.extend(pool)
            audit[run["run_name"]] = {
                "thresholds": thresholds,
                "experienced_pool": sum(row["state"] == "experienced" for row in pool),
                "fresh_pool": sum(row["state"] == "fresh" for row in pool),
            }
        experienced = []
        by_run = {
            run["run_name"]: [
                row
                for row in pools
                if row["run_name"] == run["run_name"] and row["state"] == "experienced"
            ]
            for run in task_runs
        }
        run_order = list(by_run)
        rng.shuffle(run_order)
        while len(experienced) < per_state:
            progressed = False
            for name in run_order:
                choices = [row for row in by_run[name] if row not in experienced]
                if choices and len(experienced) < per_state:
                    choices.sort(key=lambda row: (-row["development_attempts"], row["parent"]["id"]))
                    experienced.append(rng.choice(choices[: min(5, len(choices))]))
                    progressed = True
            if not progressed:
                raise ValueError(f"not enough experienced anchors for {task}")
        fresh_pool = [row for row in pools if row["state"] == "fresh"]
        fresh = []
        for target in experienced:
            choices = [row for row in fresh_pool if row not in fresh]
            choices.sort(
                key=lambda row: (
                    abs(row["anchor_fitness"] - target["anchor_fitness"])
                    / max(abs(target["anchor_fitness"]), 1e-8),
                    abs(row["fitness_rank"] - target["fitness_rank"]),
                    abs(row["depth"] - target["depth"]),
                    row["run_name"] != target["run_name"],
                    row["parent"]["id"],
                )
            )
            match = choices[0]
            match["matched_experienced_source"] = f'{target["run_name"]}:{target["parent"]["id"]}'
            fresh.append(match)
        for row in experienced + fresh:
            row["anchor_id"] = (
                f'{task}:{row["state"]}:{row["run_name"]}:node_{row["parent"]["id"]}'
            )
            selected.append(row)
    return selected, audit


def prepare(out: Path, per_state: int, seed: int) -> None:
    if out.exists():
        raise FileExistsError(out)
    anchors, audit = select_anchors(seed, per_state)
    rng = random.Random(seed + 1)
    blocks = []
    for anchor in anchors:
        arms = list(ARMS)
        rng.shuffle(arms)
        blocks.append(
            {
                "anchor_id": anchor["anchor_id"],
                "task": anchor["task"],
                "state": anchor["state"],
                "arms": arms,
            }
        )
    rng.shuffle(blocks)
    schedule = []
    for order, block in enumerate(blocks):
        arms = block.pop("arms")
        for arm_order, arm in enumerate(arms):
            schedule.append(
                {
                    **block,
                    "block_order": order,
                    "arm_order": arm_order,
                    "arm": arm,
                    "trial_id": f'{block["anchor_id"]}:{arm}',
                    "step1_seed": seed * 1000 + order + 1,
                    "summary_seed": seed * 1000 + 1000 + order + 1,
                    "step2_seed": seed * 1000 + 2000 + order + 1,
                }
            )
    out.mkdir(parents=True)
    for name, rows in (("anchors.jsonl", anchors), ("schedule.jsonl", schedule)):
        with (out / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
    config = read_json(CONFIG_PATH)
    config.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        source_snapshot_frozen_at=read_json(SOURCE / "snapshot.json")["frozen_at"],
        anchors=len(anchors),
        trajectories=len(schedule),
        anchor_selection_audit=audit,
    )
    dump(out / "config.json", config)


class TrialRuntime:
    def __init__(self, task: str, backend: str):
        self.evaluation, self.task_config = build_task(task, None)
        profile = resolve_backend(backend, None, None, None)
        self.llm = build_llm_client(
            base_url=profile.base_url,
            model=profile.model,
            no_proxy=profile.no_proxy,
            max_tokens=OUTPUT_TOKENS,
        )
        self.secure = SecureEvaluator(self.evaluation)
        self.contract = prompts.build_task_contract(self.evaluation)
        template = TextFunctionProgramConverter.text_to_program(self.evaluation.template_program)
        self.template_func = template.functions[0]
        self.builder = prompts.PromptBuilder(
            self.llm,
            self.contract,
            max_tokens=PROMPT_LIMIT,
            history_tokens=HISTORY_TOKENS,
            max_events=MAX_EVENTS,
            summary_tokens=SUMMARY_TOKENS,
        )

    def reset_prompt_cache(self) -> None:
        """Node ids are run-local, so caches cannot cross fixed anchors."""
        self.builder._counts.clear()
        self.builder._views.clear()
        self.builder._summaries.clear()

    def close(self) -> None:
        self.llm.close()

    def parse(self, response: str, finish: str) -> tuple[str, str] | None:
        if finish not in {"stop", "unknown"}:
            return None
        match = CODE_RE.fullmatch(_strip_thinking(response))
        if not match:
            return None
        from traceaad.v10_6.traceaad import TraceAADV106

        parser = type("Parser", (), {"_template_func": self.template_func})()
        parsed = TraceAADV106.parse_response(parser, response, finish)
        return (parsed[0], parsed[1]) if parsed else None

    def call(self, prompt: str, seed: int) -> dict[str, Any]:
        started = time.time()
        for attempt in range(1, TRANSPORT_RETRIES + 1):
            try:
                details = self.llm.draw_sample_with_details(
                    prompt, max_tokens=OUTPUT_TOKENS, seed=seed
                )
                return {**details, "attempts": attempt, "seconds": time.time() - started}
            except Exception as exc:
                if attempt == TRANSPORT_RETRIES:
                    return {
                        "error": repr(exc),
                        "attempts": attempt,
                        "seconds": time.time() - started,
                    }
        raise AssertionError("unreachable")

    def evaluate(self, code: str) -> dict[str, Any]:
        started = time.time()
        outcome = self.secure.evaluate_program_with_details(code)
        score = getattr(outcome.result, "fitness", outcome.result)
        try:
            fitness = float(score)
        except (TypeError, ValueError, OverflowError):
            fitness = None
        if fitness is not None and not math.isfinite(fitness):
            fitness = None
        return {
            "fitness": fitness,
            "failure_kind": outcome.failure_kind,
            "error": outcome.error,
            "seconds": time.time() - started,
        }


def generated_step(
    runtime: TrialRuntime,
    *,
    current: Node,
    ancestors: list[Node],
    operator: str,
    seed: int,
    summary_seed: int | None,
    synthetic_id: int,
) -> tuple[dict[str, Any], Node | None]:
    prompt = runtime.builder.build(current, ancestors, operator)
    call = runtime.call(prompt.text, seed)
    record: dict[str, Any] = {
        "operator": operator,
        "prompt_hash": digest(prompt.text),
        "prompt_tokens": prompt.tokens,
        "history_ids": list(prompt.history_ids),
        "context_omissions": list(prompt.omissions),
        "generation": call,
    }
    if "content" not in call:
        record["status"] = "transport_failed"
        return record, None
    parsed = runtime.parse(call["content"], call.get("finish_reason") or "unknown")
    if not parsed:
        record["status"] = "invalid_output"
        return record, None
    design_idea, code = parsed
    record.update(design_idea=design_idea, code=code, code_hash=digest(code))
    implementation_idea = design_idea
    if summary_seed is not None:
        summary_prompt = prompts.build_summary_prompt(
            runtime.contract, design_idea, code, current
        )
        summary_call = runtime.call(summary_prompt, summary_seed)
        text = _strip_thinking(summary_call.get("content", "")).strip()
        match = SUMMARY_RE.fullmatch(text)
        implementation_idea = (
            match.group(1).strip()
            if match
            and (summary_call.get("finish_reason") or "unknown") in {"stop", "unknown"}
            else ""
        )
        record.update(
            summary_prompt_hash=digest(summary_prompt),
            summary_prompt_tokens=runtime.builder.count(summary_prompt, chat=True),
            summary=summary_call,
            summary_status="present" if implementation_idea else "unavailable",
        )
    evaluation = runtime.evaluate(code)
    record["evaluation"] = evaluation
    if evaluation["fitness"] is None:
        record["status"] = "eval_failed"
        return record, None
    record["status"] = "ok"
    child = Node(
        id=synthetic_id,
        code=code,
        idea=implementation_idea,
        fitness=evaluation["fitness"],
        evaluation_id=None,
        parent_id=current.id,
        operator=operator,
        donor_id=None,
    )
    return record, child


def run_shard(out: Path, backend: str, shard_index: int, num_shards: int) -> None:
    if read_json(out / "config.json")["protocol_id"] != PROTOCOL:
        raise ValueError("protocol mismatch")
    anchors = {row["anchor_id"]: row for row in read_jsonl(out / "anchors.jsonl")}
    schedule = [
        row
        for row in read_jsonl(out / "schedule.jsonl")
        if row["block_order"] % num_shards == shard_index
    ]
    result_path = out / "shards" / f"shard_{shard_index:02d}.jsonl"
    completed = {row["trial_id"] for row in read_jsonl(result_path)}
    runtimes: dict[str, TrialRuntime] = {}
    try:
        for trial in schedule:
            if trial["trial_id"] in completed:
                continue
            anchor = anchors[trial["anchor_id"]]
            task = trial["task"]
            if task not in runtimes:
                runtimes[task] = TrialRuntime(task, backend)
            runtime = runtimes[task]
            runtime.reset_prompt_cache()
            parent = node(anchor["parent"])
            ancestors = [node(row) for row in anchor["ancestors"]]
            base_id = (
                10_000_000 + trial["block_order"] * 10 + trial["arm_order"] * 2
            )
            step1, child1 = generated_step(
                runtime,
                current=parent,
                ancestors=ancestors,
                operator=ARMS[trial["arm"]][0],
                seed=trial["step1_seed"],
                summary_seed=trial["summary_seed"],
                synthetic_id=base_id,
            )
            step2, child2 = None, None
            if child1 is not None:
                step2, child2 = generated_step(
                    runtime,
                    current=child1,
                    ancestors=[parent, *ancestors],
                    operator="Refine",
                    seed=trial["step2_seed"],
                    summary_seed=None,
                    synthetic_id=base_id + 1,
                )
            result = {
                **trial,
                "protocol_id": PROTOCOL,
                "backend": backend,
                "anchor_fitness": anchor["anchor_fitness"],
                "step1": step1,
                "step2": step2,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            append(result_path, result)
            completed.add(trial["trial_id"])
            print(
                f'shard={shard_index} {trial["trial_id"]} '
                f'step1={step1["status"]} step2={step2 and step2["status"]}',
                flush=True,
            )
    finally:
        for runtime in runtimes.values():
            runtime.close()


def load_results(out: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((out / "shards").glob("shard_*.jsonl")):
        rows.extend(read_jsonl(path))
    return rows


def profile_first_steps(out: Path, workers: int) -> None:
    rows = load_results(out)
    anchors = {row["anchor_id"]: row for row in read_jsonl(out / "anchors.jsonl")}
    jobs = {
        row["trial_id"]: (row["task"], {"id": 0, "code": row["step1"]["code"]})
        for row in rows
        if row["arm"] == "PR" and row["step1"]["status"] == "ok"
    }
    profiles = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(profile_worker, job): trial_id
            for trial_id, job in jobs.items()
        }
        for future in as_completed(futures):
            profiles[futures[future]] = future.result()
    output = []
    by_trial = {row["trial_id"]: row for row in rows}
    for trial_id, result in profiles.items():
        row = by_trial[trial_id]
        anchor = anchors[row["anchor_id"]]
        anchor_path = (
            SOURCE
            / "profiles"
            / row["task"]
            / f'{profile_candidate(anchor["parent"])["key"]}.json'
        )
        baseline = read_json(anchor_path)
        distances = [
            profile_distance(result["panels"][panel], baseline["panels"][panel])
            for panel in ("A", "B")
            if result["panels"][panel]["ok"] and baseline["panels"][panel]["ok"]
        ]
        movement = float(np.mean(distances)) if len(distances) == 2 else None
        output.append(
            {
                "trial_id": trial_id,
                "task": row["task"],
                "anchor_id": row["anchor_id"],
                "ok": movement is not None,
                "panel_distances": distances,
                "behavior_movement": movement,
                "historical_pivot_move_median": anchor[
                    "historical_pivot_move_median"
                ],
                "large_move": (
                    movement >= anchor["historical_pivot_move_median"]
                    if movement is not None
                    else None
                ),
            }
        )
    with (out / "first_step_behavior.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(output, key=lambda item: item["trial_id"]):
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )


def trajectory_metrics(row: dict[str, Any]) -> dict[str, Any]:
    baseline = float(row["anchor_fitness"])
    f1 = row["step1"].get("evaluation", {}).get("fitness")
    f2 = (row.get("step2") or {}).get("evaluation", {}).get("fitness")
    best1 = max([baseline] + ([float(f1)] if f1 is not None else []))
    best2 = max([best1] + ([float(f2)] if f2 is not None else []))
    scale = max(abs(baseline), 1e-8)
    return {
        **row,
        "f1": f1,
        "f2": f2,
        "q1": best1 - baseline,
        "q2": best2 - baseline,
        "relative_q1": (best1 - baseline) / scale,
        "relative_q2": (best2 - baseline) / scale,
        "continuation_gain": (
            float(f2) - float(f1) if f1 is not None and f2 is not None else None
        ),
        "relative_continuation_gain": (
            (float(f2) - float(f1)) / scale
            if f1 is not None and f2 is not None
            else None
        ),
        "step1_valid": f1 is not None,
        "step2_valid": f2 is not None,
    }


def paired(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    blocks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        blocks[row["anchor_id"]][row["arm"]] = row
    differences, rr_values, pr_values = [], [], []
    for arms in blocks.values():
        if set(arms) != set(ARMS):
            continue
        rr, pr = arms["RR"].get(metric), arms["PR"].get(metric)
        if rr is None or pr is None:
            continue
        rr_values.append(float(rr))
        pr_values.append(float(pr))
        differences.append(float(pr) - float(rr))
    return {
        "paired_anchors": len(differences),
        "RR_mean": statistics.fmean(rr_values) if rr_values else None,
        "PR_mean": statistics.fmean(pr_values) if pr_values else None,
        "PR_minus_RR_mean": statistics.fmean(differences) if differences else None,
        "PR_minus_RR_median": statistics.median(differences) if differences else None,
        "PR_better": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "RR_better": sum(value < 0 for value in differences),
        "differences": differences,
    }


def bootstrap_paired(
    rows: list[dict[str, Any]],
    metric: str,
    seed: int,
    samples: int = 10_000,
) -> list[float] | None:
    values = paired(rows, metric)["differences"]
    if not values:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def analyze(out: Path) -> dict[str, Any]:
    expected = read_jsonl(out / "schedule.jsonl")
    raw = load_results(out)
    if len(raw) != len({row["trial_id"] for row in raw}):
        raise ValueError("duplicate trials")
    behavior = {
        row["trial_id"]: row
        for row in read_jsonl(out / "first_step_behavior.jsonl")
    }
    rows = []
    for raw_row in raw:
        row = trajectory_metrics(raw_row)
        sensor = behavior.get(raw_row["trial_id"])
        if sensor:
            row.update(
                {
                    key: value
                    for key, value in sensor.items()
                    if key not in {"trial_id", "task", "anchor_id"}
                }
            )
        rows.append(row)
    metrics = (
        "relative_q2",
        "relative_q1",
        "relative_continuation_gain",
        "step1_valid",
        "step2_valid",
    )
    result: dict[str, Any] = {
        "protocol_id": PROTOCOL,
        "complete": len(rows) == len(expected),
        "expected_trajectories": len(expected),
        "completed_trajectories": len(rows),
        "evaluator_calls": sum(
            row["step1"].get("evaluation") is not None for row in rows
        )
        + sum(
            (row.get("step2") or {}).get("evaluation") is not None for row in rows
        ),
        "generation_calls": len(rows)
        + sum(row.get("step2") is not None for row in rows),
        "summary_calls": sum("summary" in row["step1"] for row in rows),
        "comparisons": {},
    }
    for state_index, state in enumerate(STATES):
        selected = [row for row in rows if row["state"] == state]
        result["comparisons"][state] = {}
        for metric_index, metric in enumerate(metrics):
            comparison = paired(selected, metric)
            comparison["bootstrap_95ci"] = bootstrap_paired(
                selected, metric, 2026090712 + state_index * 10 + metric_index
            )
            result["comparisons"][state][metric] = comparison
    for task in TASKS:
        result["comparisons"][task] = {}
        for state in STATES:
            selected = [
                row
                for row in rows
                if row["task"] == task and row["state"] == state
            ]
            result["comparisons"][task][state] = {
                metric: paired(selected, metric) for metric in metrics
            }
    pivot = [
        row
        for row in rows
        if row["arm"] == "PR" and row.get("behavior_movement") is not None
    ]
    valid_continuations = [
        row for row in pivot if row["relative_continuation_gain"] is not None
    ]
    correlation = (
        float(
            np.corrcoef(
                [row["behavior_movement"] for row in valid_continuations],
                [
                    row["relative_continuation_gain"]
                    for row in valid_continuations
                ],
            )[0, 1]
        )
        if len(valid_continuations) >= 3
        else None
    )
    option_correlation = (
        float(
            np.corrcoef(
                [row["behavior_movement"] for row in pivot],
                [row["relative_q2"] for row in pivot],
            )[0, 1]
        )
        if len(pivot) >= 3
        else None
    )
    large = [row for row in valid_continuations if row["large_move"]]
    small = [row for row in valid_continuations if not row["large_move"]]
    result["pivot_behavior_sensor"] = {
        "profiled": len(pivot),
        "valid_continuations": len(valid_continuations),
        "pearson_movement_continuation": correlation,
        "pearson_movement_q2": option_correlation,
        "continuations_adding_to_q1": sum(
            row["relative_q2"] > row["relative_q1"] for row in pivot
        ),
        "large_move": {
            "n": len(large),
            "mean_continuation": (
                statistics.fmean(row["relative_continuation_gain"] for row in large)
                if large
                else None
            ),
            "mean_q2": (
                statistics.fmean(row["relative_q2"] for row in large)
                if large
                else None
            ),
        },
        "small_move": {
            "n": len(small),
            "mean_continuation": (
                statistics.fmean(row["relative_continuation_gain"] for row in small)
                if small
                else None
            ),
            "mean_q2": (
                statistics.fmean(row["relative_q2"] for row in small)
                if small
                else None
            ),
        },
    }
    primary = result["comparisons"]["experienced"]["relative_q2"]
    directions = sum(
        result["comparisons"][task]["experienced"]["relative_q2"][
            "PR_minus_RR_mean"
        ]
        is not None
        and result["comparisons"][task]["experienced"]["relative_q2"][
            "PR_minus_RR_mean"
        ]
        > 0
        for task in TASKS
    )
    ci = primary["bootstrap_95ci"]
    result["decision"] = {
        "positive_gate": bool(
            primary["PR_minus_RR_mean"] is not None
            and primary["PR_minus_RR_mean"] > 0
            and ci
            and ci[0] > 0
            and directions >= 2
        ),
        "positive_task_directions": directions,
        "interpretation": (
            "Pilot support requires positive experienced-anchor relative Q2, "
            "an anchor-bootstrap lower bound above zero, and at least two positive "
            "task directions. Otherwise report the frozen result without retuning."
        ),
    }
    dump(out / "analysis.json", result)
    with (out / "analysis_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, allow_nan=False, sort_keys=True)
                + "\n"
            )
    return result


def status(out: Path) -> dict[str, Any]:
    schedule = read_jsonl(out / "schedule.jsonl")
    rows = load_results(out)
    return {
        "expected": len(schedule),
        "completed": len(rows),
        "step1_ok": sum(row["step1"]["status"] == "ok" for row in rows),
        "step2_ok": sum(
            (row.get("step2") or {}).get("status") == "ok" for row in rows
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--out", type=Path, default=DEFAULT)
    command.add_argument("--per-state", type=int, default=3)
    command.add_argument("--seed", type=int, default=2026090711)
    command = sub.add_parser("run-shard")
    command.add_argument("--out", type=Path, default=DEFAULT)
    command.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    command.add_argument("--shard-index", type=int, required=True)
    command.add_argument("--num-shards", type=int, required=True)
    command = sub.add_parser("profile")
    command.add_argument("--out", type=Path, default=DEFAULT)
    command.add_argument("--workers", type=int, default=12)
    command = sub.add_parser("analyze")
    command.add_argument("--out", type=Path, default=DEFAULT)
    command = sub.add_parser("status")
    command.add_argument("--out", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out, args.per_state, args.seed)
    elif args.command == "run-shard":
        run_shard(args.out, args.backend, args.shard_index, args.num_shards)
    elif args.command == "profile":
        profile_first_steps(args.out, args.workers)
    elif args.command == "analyze":
        print(json.dumps(analyze(args.out), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(status(args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

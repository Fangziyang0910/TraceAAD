"""Paired one-step probe for TraceAAD V10.1/V10.2 prompt drift.

The Fuse experiment fixes parent, trajectory, operator, donor, model sampling
seed, backend, and evaluator, then changes only cumulative prompt components.
An additional two-arm Init experiment isolates the changed root prompt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    REPO_ROOT,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    build_llm_client,
    build_task,
    resolve_backend,
)
from core import SecureEvaluator, TextFunctionProgramConverter
from traceaad.v10_1 import prompts as prompts_v101
from traceaad.v10_1.schema import Node
from traceaad.v10_1.traceaad import TraceAADV101
from traceaad.v10_2 import prompts as prompts_v102
from traceaad.v10_2.traceaad import TraceAADV102

PROTOCOL_ID = "traceaad-v10-paired-prompt-kernel-probe-v1"
DESIGN_SEED = 1010203
OUTPUT_TOKENS = 16384
TOTAL_CONTEXT_TOKENS = 32768
CHARS_PER_TOKEN = 3.5
MAX_TRAJECTORY_GENS = 8
TRANSPORT_RETRIES = 3
SOURCE_VERSIONS = ("v101", "v102")
FUSE_CONDITIONS = (
    "v101",
    "implementation_principle",
    "comment_free",
    "operator_instruction",
    "v102",
)
INIT_CONDITIONS = ("v101", "v102")
STRATA = ("low", "middle", "high")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _node(row: dict[str, Any]) -> Node:
    return Node(
        id=int(row["id"]),
        code=str(row["code"]),
        idea=str(row["idea"]),
        fitness=float(row["fitness"]),
        evaluation_id=row.get("evaluation_id"),
        parent_id=row.get("parent_id"),
        operator=str(row.get("operator") or row.get("origin_operator") or "Init"),
        donor_id=row.get("donor_id"),
    )


def _render_current(node: Node, *, strip_comments: bool) -> str:
    code = (
        prompts_v102.strip_comments_for_prompt(node.code)
        if strip_comments
        else node.code
    )
    return (
        "# Current Algorithm\n"
        f"Idea: {node.idea}\n"
        f"Fitness: {node.fitness}\n\n"
        f"```python\n{code}\n```"
    )


def _render_reference(node: Node, *, strip_comments: bool) -> str:
    code = (
        prompts_v102.strip_comments_for_prompt(node.code)
        if strip_comments
        else node.code
    )
    return (
        "# Reference Algorithm\n"
        f"Idea: {node.idea}\n"
        f"Fitness: {node.fitness}\n\n"
        f"```python\n{code}\n```"
    )


def _render_v101_runtime_trajectory(ancestors: list[Node], display: int) -> str:
    """Render the Generation labels used by the completed V10.1 runs."""
    if display <= 0 or not ancestors:
        return ""
    lines = ["# Historical Design Trajectory"]
    for index, node in enumerate(reversed(ancestors[:display])):
        generation = -(display - index)
        entry = (
            f"\nGeneration {generation}\n"
            f"Idea: {node.idea}\n"
            f"Fitness: {node.fitness}"
        )
        parent_position = display - index
        if parent_position < len(ancestors):
            entry += (
                " ("
                f"{prompts_v101._trend(node.fitness, ancestors[parent_position].fitness)}"  # noqa: SLF001
                ")"
            )
        lines.append(entry)
    return "\n".join(lines)


def build_prompt_variant(
    *,
    condition: str,
    task_contract: str,
    current: Node | None,
    ancestors: list[Node],
    trajectory_display: int,
    operator: str,
    donor: Node | None,
) -> str:
    """Build one cumulative treatment, with a fixed trajectory display count."""
    if condition == "v102":
        return prompts_v102._assemble(  # noqa: SLF001
            task_contract,
            current,
            ancestors,
            trajectory_display,
            operator,
            donor,
        )
    if condition not in FUSE_CONDITIONS:
        raise ValueError(f"unknown prompt condition: {condition}")
    if operator == "Init":
        if condition != "v101" or current is not None:
            raise ValueError("only the V10.1 endpoint is valid for this Init branch")
        parts = [task_contract]
        parts.append(prompts_v101._render_operator(operator, prompts_v101.INIT_INSTRUCTION))  # noqa: SLF001
        parts.append(f"# Output\n{prompts_v101.OUTPUT_CONTRACT}")
        return "\n\n\n".join(parts)
    if current is None:
        raise ValueError("intermediate prompt treatments apply only to expansion")

    add_principle = condition != "v101"
    strip_comments = condition in {"comment_free", "operator_instruction"}
    use_new_operator = condition == "operator_instruction"
    parts = [task_contract]
    if add_principle:
        parts.append(prompts_v102.IMPLEMENTATION_PRINCIPLE)
    parts.append(_render_current(current, strip_comments=strip_comments))
    trajectory = _render_v101_runtime_trajectory(ancestors, trajectory_display)
    if trajectory:
        parts.append(trajectory)
    if donor is not None:
        parts.append(_render_reference(donor, strip_comments=strip_comments))
    instructions = (
        prompts_v102.OPERATOR_INSTRUCTIONS
        if use_new_operator
        else prompts_v101.OPERATOR_INSTRUCTIONS
    )
    parts.append(prompts_v101._render_operator(operator, instructions[operator]))  # noqa: SLF001
    parts.append(f"# Output\n{prompts_v101.OUTPUT_CONTRACT}")
    return "\n\n\n".join(parts)


def _common_trajectory_display(
    *,
    task_contract: str,
    current: Node,
    ancestors: list[Node],
    donor: Node,
    max_prompt_chars: int,
) -> int:
    for display in range(min(MAX_TRAJECTORY_GENS, len(ancestors)), -1, -1):
        prompts = [
            build_prompt_variant(
                condition=condition,
                task_contract=task_contract,
                current=current,
                ancestors=ancestors,
                trajectory_display=display,
                operator="Fuse",
                donor=donor,
            )
            for condition in FUSE_CONDITIONS
        ]
        if all(len(prompt) <= max_prompt_chars for prompt in prompts):
            return display
    raise ValueError("fixed parent and donor do not fit the prompt context")


def _completed_run_dirs(task: str, source_version: str) -> list[Path]:
    root = REPO_ROOT / "experiments" / f"traceaad_v10_{source_version[-1]}" / "results" / task
    runs = []
    for path in sorted(root.glob(f"20260902_*_{source_version}_rep*")):
        summary_path = path / "logs" / "run_summary.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        if summary.get("status") == "finished" and summary.get("budget_used") == 1000:
            runs.append(path)
    return runs


def _ancestors(nodes: dict[int, Node], parent: Node) -> list[Node]:
    result = []
    parent_id = parent.parent_id
    while parent_id is not None:
        result.append(nodes[parent_id])
        parent_id = nodes[parent_id].parent_id
    return result[: MAX_TRAJECTORY_GENS + 1]


def _extract_fuse_pool(task: str, source_version: str) -> list[dict[str, Any]]:
    evaluation, _ = build_task(task, eval_workers=None)
    task_contract = prompts_v101.build_task_contract(
        evaluation.task_description, evaluation.template_program
    )
    if task_contract != prompts_v102.build_task_contract(
        evaluation.task_description, evaluation.template_program
    ):
        raise AssertionError("V10.1 and V10.2 task contracts differ")
    max_chars = int((TOTAL_CONTEXT_TOKENS - OUTPUT_TOKENS) * CHARS_PER_TOKEN)
    pool = []
    for run_dir in _completed_run_dirs(task, source_version):
        config = _read_json(run_dir / "run_config.json")
        state = _read_json(run_dir / "tree_state.json")
        nodes = {int(row["id"]): _node(row) for row in state["nodes"]}
        for event in _read_jsonl(run_dir / "events.jsonl"):
            if event.get("operator") != "Fuse":
                continue
            parent_id = event.get("parent_id")
            donor_id = event.get("donor_id")
            if parent_id is None or donor_id is None:
                continue
            parent = nodes[int(parent_id)]
            donor = nodes[int(donor_id)]
            lineage = _ancestors(nodes, parent)
            display = _common_trajectory_display(
                task_contract=task_contract,
                current=parent,
                ancestors=lineage,
                donor=donor,
                max_prompt_chars=max_chars,
            )
            baseline_prompt = build_prompt_variant(
                condition="v101",
                task_contract=task_contract,
                current=parent,
                ancestors=lineage,
                trajectory_display=display,
                operator="Fuse",
                donor=donor,
            )
            source_order = int(event.get("batch", event.get("step", 0)))
            pool.append(
                {
                    "anchor_id": (
                        f"{task}:{source_version}:{run_dir.name}:"
                        f"fuse_{source_order}"
                    ),
                    "task": task,
                    "source_version": source_version,
                    "source_run": str(run_dir.relative_to(REPO_ROOT)),
                    "source_repeat": int(config["repeat"]),
                    "source_order": source_order,
                    "source_outcome_status": event.get("status"),
                    "source_child_fitness": event.get("fitness"),
                    "parent": vars(parent),
                    "donor": vars(donor),
                    "ancestors": [vars(node) for node in lineage],
                    "trajectory_display": display,
                    "parent_fitness": parent.fitness,
                    "donor_fitness": donor.fitness,
                    "baseline_prompt_hash": _hash(baseline_prompt),
                    "recorded_baseline_prompt_match": (
                        event.get("prompt") == baseline_prompt
                        if "prompt" in event
                        else None
                    ),
                }
            )
    unique = {}
    for anchor in pool:
        unique.setdefault(anchor["baseline_prompt_hash"], anchor)
    return list(unique.values())


def _add_rank_strata(rows: list[dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda row: (row["parent_fitness"], row["anchor_id"]))
    for index, row in enumerate(ranked):
        row["stratum"] = STRATA[min(2, index * 3 // len(ranked))]


def _balanced_sample(
    rows: list[dict[str, Any]], *, count: int, rng: random.Random
) -> list[dict[str, Any]]:
    by_repeat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_repeat[int(row["source_repeat"])].append(row)
    for group in by_repeat.values():
        rng.shuffle(group)
    repeats = sorted(by_repeat)
    rng.shuffle(repeats)
    selected = []
    while len(selected) < count:
        progressed = False
        for repeat in repeats:
            if by_repeat[repeat] and len(selected) < count:
                selected.append(by_repeat[repeat].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"only {len(selected)} anchors available for {count}")
        rng.shuffle(repeats)
    return selected


def select_anchors(
    *, anchors_per_source_task: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if anchors_per_source_task % len(STRATA):
        raise ValueError("anchors per source/task must be divisible by three strata")
    rng = random.Random(seed)
    selected = []
    audit: dict[str, Any] = {}
    per_stratum = anchors_per_source_task // len(STRATA)
    for task in ALL_TASKS:
        for source_version in SOURCE_VERSIONS:
            pool = _extract_fuse_pool(task, source_version)
            _add_rank_strata(pool)
            key = f"{task}:{source_version}"
            audit[key] = {
                "completed_source_runs": [
                    str(path.relative_to(REPO_ROOT))
                    for path in _completed_run_dirs(task, source_version)
                ],
                "unique_prompt_pool": len(pool),
                "selected_by_stratum": {},
            }
            for stratum in STRATA:
                candidates = [row for row in pool if row["stratum"] == stratum]
                chosen = _balanced_sample(candidates, count=per_stratum, rng=rng)
                selected.extend(chosen)
                audit[key]["selected_by_stratum"][stratum] = len(chosen)
    return selected, audit


def build_schedule(
    anchors: list[dict[str, Any]], *, init_pairs_per_task: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 1)
    blocks = []
    for anchor in anchors:
        blocks.append(
            {
                "block_id": anchor["anchor_id"],
                "kind": "fuse",
                "anchor_id": anchor["anchor_id"],
                "task": anchor["task"],
                "conditions": rng.sample(list(FUSE_CONDITIONS), len(FUSE_CONDITIONS)),
            }
        )
    for task in ALL_TASKS:
        for replicate in range(1, init_pairs_per_task + 1):
            blocks.append(
                {
                    "block_id": f"{task}:init:{replicate}",
                    "kind": "init",
                    "anchor_id": None,
                    "task": task,
                    "conditions": rng.sample(list(INIT_CONDITIONS), len(INIT_CONDITIONS)),
                }
            )
    rng.shuffle(blocks)
    schedule = []
    for block_order, block in enumerate(blocks):
        sampling_seed = seed * 1000 + block_order + 1
        for within_block_order, condition in enumerate(block.pop("conditions")):
            schedule.append(
                {
                    **block,
                    "block_order": block_order,
                    "within_block_order": within_block_order,
                    "condition": condition,
                    "sampling_seed": sampling_seed,
                    "trial_id": f"{block['block_id']}:{condition}",
                }
            )
    return schedule


def _prompt_for_trial(
    trial: dict[str, Any], anchor: dict[str, Any] | None, evaluation: Any
) -> str:
    task_contract = prompts_v101.build_task_contract(
        evaluation.task_description, evaluation.template_program
    )
    if trial["kind"] == "init":
        return build_prompt_variant(
            condition=trial["condition"],
            task_contract=task_contract,
            current=None,
            ancestors=[],
            trajectory_display=0,
            operator="Init",
            donor=None,
        )
    if anchor is None:
        raise ValueError("Fuse trial is missing its anchor")
    return build_prompt_variant(
        condition=trial["condition"],
        task_contract=task_contract,
        current=_node(anchor["parent"]),
        ancestors=[_node(row) for row in anchor["ancestors"]],
        trajectory_display=int(anchor["trajectory_display"]),
        operator="Fuse",
        donor=_node(anchor["donor"]),
    )


def prepare(
    run_dir: Path,
    *,
    anchors_per_source_task: int,
    init_pairs_per_task: int,
    seed: int,
) -> None:
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    anchors, source_audit = select_anchors(
        anchors_per_source_task=anchors_per_source_task, seed=seed
    )
    schedule = build_schedule(
        anchors, init_pairs_per_task=init_pairs_per_task, seed=seed
    )
    anchor_map = {row["anchor_id"]: row for row in anchors}
    evaluations = {task: build_task(task, eval_workers=None)[0] for task in ALL_TASKS}
    prompt_audit = []
    for trial in schedule:
        prompt = _prompt_for_trial(
            trial,
            anchor_map.get(trial["anchor_id"]),
            evaluations[trial["task"]],
        )
        prompt_audit.append(
            {
                "trial_id": trial["trial_id"],
                "prompt_hash": _hash(prompt),
                "prompt_chars": len(prompt),
            }
        )
    run_dir.mkdir(parents=True)
    _write_jsonl(run_dir / "anchors.jsonl", anchors)
    _write_jsonl(run_dir / "schedule.jsonl", schedule)
    _write_jsonl(run_dir / "prompt_audit.jsonl", prompt_audit)
    _write_json(
        run_dir / "probe_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "design_seed": seed,
            "source_versions": list(SOURCE_VERSIONS),
            "fuse_conditions": list(FUSE_CONDITIONS),
            "init_conditions": list(INIT_CONDITIONS),
            "anchors_per_source_task": anchors_per_source_task,
            "init_pairs_per_task": init_pairs_per_task,
            "fuse_anchor_count": len(anchors),
            "trial_count": len(schedule),
            "output_tokens": OUTPUT_TOKENS,
            "total_context_tokens": TOTAL_CONTEXT_TOKENS,
            "temperature": SAMPLING_TEMPERATURE,
            "top_p": SAMPLING_TOP_P,
            "top_k": SAMPLING_TOP_K,
            "independent_unit": "fixed proposal-state block",
            "blocking": (
                "all conditions for a block share task, state, sampling seed, "
                "backend, and evaluator"
            ),
            "run_order": "seeded random condition order within every block",
            "generation_evaluation_phases_separated": True,
            "evaluation_execution": "one local sequential evaluator process",
            "primary_fuse_responses": [
                "delta_fitness",
                "unconditional_improvement_rate",
                "invalid_rate",
            ],
            "init_responses": ["fitness", "invalid_rate"],
            "source_audit": source_audit,
        },
    )


def _argument_names(args_source: str) -> list[str] | None:
    try:
        tree = ast.parse(f"def probe({args_source}):\n    pass")
    except SyntaxError:
        return None
    args = tree.body[0].args
    return [arg.arg for arg in args.posonlyargs + args.args]


def _parser(evaluation: Any, condition: str) -> TraceAADV101 | TraceAADV102:
    parser_type = TraceAADV102 if condition == "v102" else TraceAADV101
    parser = parser_type.__new__(parser_type)
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None or len(template.functions) != 1:
        raise ValueError("evaluation template must define exactly one function")
    parser._template = template
    parser._template_func = template.functions[0]
    parser._template_arg_names = _argument_names(template.functions[0].args)
    return parser


def generate_shard(
    run_dir: Path, *, backend: str, shard_index: int, num_shards: int
) -> None:
    config = _read_json(run_dir / "probe_config.json")
    if config["protocol_id"] != PROTOCOL_ID:
        raise ValueError("probe protocol mismatch")
    anchors = {row["anchor_id"]: row for row in _read_jsonl(run_dir / "anchors.jsonl")}
    schedule = [
        row
        for row in _read_jsonl(run_dir / "schedule.jsonl")
        if int(row["block_order"]) % num_shards == shard_index
    ]
    result_path = run_dir / "generations" / f"shard_{shard_index:02d}.jsonl"
    completed = {row["trial_id"] for row in _read_jsonl(result_path)}
    profile = resolve_backend(backend, None, None, None)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=OUTPUT_TOKENS,
    )
    evaluations: dict[str, Any] = {}
    shard_dir = run_dir / "shards" / f"shard_{shard_index:02d}"
    _write_json(
        shard_dir / "shard_config.json",
        {
            "protocol_id": PROTOCOL_ID,
            "phase": "generation",
            "backend": backend,
            "base_url": profile.base_url,
            "model": profile.model,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "assigned_trials": len(schedule),
        },
    )
    try:
        for trial in schedule:
            if trial["trial_id"] in completed:
                continue
            task = trial["task"]
            if task not in evaluations:
                evaluations[task] = build_task(task, eval_workers=None)[0]
            evaluation = evaluations[task]
            anchor = anchors.get(trial["anchor_id"])
            prompt = _prompt_for_trial(trial, anchor, evaluation)
            prompt_tokens = int(llm.count_prompt_tokens(prompt))
            if prompt_tokens + OUTPUT_TOKENS > TOTAL_CONTEXT_TOKENS:
                raise ValueError(
                    f"{trial['trial_id']} exceeds context: "
                    f"{prompt_tokens}+{OUTPUT_TOKENS}>{TOTAL_CONTEXT_TOKENS}"
                )
            start = time.time()
            for attempt in range(1, TRANSPORT_RETRIES + 1):
                try:
                    response = llm.draw_sample(
                        prompt,
                        max_tokens=OUTPUT_TOKENS,
                        seed=int(trial["sampling_seed"]),
                    )
                    break
                except Exception:
                    if attempt == TRANSPORT_RETRIES:
                        raise
            elapsed = time.time() - start
            parsed = _parser(evaluation, trial["condition"]).parse_response(response)
            base = {
                **trial,
                "protocol_id": PROTOCOL_ID,
                "source_version": anchor.get("source_version") if anchor else None,
                "source_run": anchor.get("source_run") if anchor else None,
                "stratum": anchor.get("stratum") if anchor else None,
                "parent_fitness": anchor.get("parent_fitness") if anchor else None,
                "prompt_hash": _hash(prompt),
                "prompt_tokens": prompt_tokens,
                "response_hash": _hash(response),
                "response": response,
                "response_tokens": int(llm.count_prompt_tokens(response)),
                "sample_seconds": elapsed,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            if parsed is None:
                result = {
                    **base,
                    "generation_status": "parse_failed",
                    "failure_kind": "parse",
                }
            else:
                idea, code, program = parsed
                result = {
                    **base,
                    "generation_status": "generated",
                    "failure_kind": None,
                    "idea": idea,
                    "candidate_code": code,
                    "candidate_code_hash": _hash(code),
                    "candidate_program": str(program),
                }
            _append_jsonl(result_path, result)
            completed.add(trial["trial_id"])
            _write_json(
                shard_dir / "summary.json",
                {
                    "status": "finished" if len(completed) == len(schedule) else "running",
                    "completed_trials": len(completed),
                    "assigned_trials": len(schedule),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            print(
                f"shard={shard_index} trial={trial['trial_id']} "
                f"status={result['generation_status']} "
                f"completed={len(completed)}/{len(schedule)}",
                flush=True,
            )
    finally:
        llm.close()


def evaluate_generated(run_dir: Path) -> None:
    schedule = _read_jsonl(run_dir / "schedule.jsonl")
    generated_rows = []
    for path in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        generated_rows.extend(_read_jsonl(path))
    generated = {row["trial_id"]: row for row in generated_rows}
    if len(generated_rows) != len(generated) or len(generated) != len(schedule):
        raise ValueError("generation must be complete and duplicate-free before evaluation")
    result_path = run_dir / "results" / "results.jsonl"
    completed = {row["trial_id"] for row in _read_jsonl(result_path)}
    evaluators: dict[str, SecureEvaluator] = {}
    for trial in schedule:
        if trial["trial_id"] in completed:
            continue
        generation = generated[trial["trial_id"]]
        if generation["generation_status"] == "parse_failed":
            result = {
                **generation,
                "status": "invalid",
                "valid": False,
                "evaluator_called": False,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        else:
            task = trial["task"]
            if task not in evaluators:
                evaluation, _ = build_task(task, eval_workers=None)
                evaluators[task] = SecureEvaluator(evaluation)
            outcome, seconds = evaluators[task].evaluate_program_record_time_with_details(
                generation["candidate_program"]
            )
            if outcome.failure_kind == "prepare_error":
                raise RuntimeError(
                    f"evaluator infrastructure failure for {trial['trial_id']}: "
                    f"{outcome.error}"
                )
            score = getattr(outcome.result, "fitness", outcome.result)
            try:
                fitness = float(score)
            except (TypeError, ValueError, OverflowError):
                fitness = math.nan
            if math.isfinite(fitness):
                parent_fitness = generation.get("parent_fitness")
                delta = (
                    fitness - float(parent_fitness)
                    if parent_fitness is not None
                    else None
                )
                result = {
                    **generation,
                    "status": "valid",
                    "valid": True,
                    "failure_kind": None,
                    "fitness": fitness,
                    "delta_fitness": delta,
                    "improved": delta > 0.0 if delta is not None else None,
                    "evaluator_called": True,
                    "evaluate_seconds": seconds,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
            else:
                result = {
                    **generation,
                    "status": "invalid",
                    "valid": False,
                    "failure_kind": outcome.failure_kind or "invalid_result",
                    "failure_error": outcome.error,
                    "evaluator_called": True,
                    "evaluate_seconds": seconds,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
        _append_jsonl(result_path, result)
        completed.add(trial["trial_id"])
        _write_json(
            run_dir / "evaluation_summary.json",
            {
                "status": "finished" if len(completed) == len(schedule) else "running",
                "completed_trials": len(completed),
                "total_trials": len(schedule),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(
            f"evaluation trial={trial['trial_id']} status={result['status']} "
            f"completed={len(completed)}/{len(schedule)}",
            flush=True,
        )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_ci(values: list[float], seed: int, samples: int = 10_000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    estimates = [
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(samples)
    ]
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _metric(row: dict[str, Any], metric: str) -> float | None:
    if metric == "invalid_rate":
        return float(row.get("valid") is not True)
    if metric == "unconditional_improvement_rate":
        return float(row.get("valid") is True and row.get("improved") is True)
    if metric == "delta_fitness":
        return float(row["delta_fitness"]) if row.get("valid") is True else None
    if metric == "fitness":
        return float(row["fitness"]) if row.get("valid") is True else None
    raise ValueError(metric)


def _contrast(
    blocks: dict[str, dict[str, dict[str, Any]]],
    *,
    left: str,
    right: str,
    metrics: tuple[str, ...],
    seed: int,
) -> dict[str, Any]:
    result = {"left": left, "right": right, "block_count": len(blocks), "metrics": {}}
    for index, metric in enumerate(metrics):
        left_values = []
        right_values = []
        differences = []
        for conditions in blocks.values():
            if left not in conditions or right not in conditions:
                continue
            a = _metric(conditions[left], metric)
            b = _metric(conditions[right], metric)
            if a is None or b is None:
                continue
            left_values.append(a)
            right_values.append(b)
            differences.append(b - a)
        result["metrics"][metric] = {
            "paired_block_count": len(differences),
            "left_mean": _mean(left_values),
            "right_mean": _mean(right_values),
            "paired_mean_difference_right_minus_left": _mean(differences),
            "paired_median_difference_right_minus_left": (
                statistics.median(differences) if differences else None
            ),
            "paired_mean_difference_bootstrap_95ci": _bootstrap_ci(
                differences, seed + index
            ),
            "blocks_better_right": sum(value > 0 for value in differences),
            "blocks_tied": sum(value == 0 for value in differences),
            "blocks_better_left": sum(value < 0 for value in differences),
        }
    return result


def analyze(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    schedule = _read_jsonl(run_dir / "schedule.jsonl")
    rows = _read_jsonl(run_dir / "results" / "results.jsonl")
    expected = {row["trial_id"] for row in schedule}
    by_trial = {row["trial_id"]: row for row in rows}
    blocks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in by_trial.values():
        blocks[row["block_id"]][row["condition"]] = row

    comparisons = [
        ("v101", "implementation_principle"),
        ("implementation_principle", "comment_free"),
        ("comment_free", "operator_instruction"),
        ("operator_instruction", "v102"),
        ("v101", "v102"),
    ]
    fuse: dict[str, Any] = {}
    for task_index, task in enumerate(ALL_TASKS):
        for source_index, source_version in enumerate((None, *SOURCE_VERSIONS)):
            name = f"task:{task}"
            if source_version is not None:
                name += f":source:{source_version}"
            selected = {
                block_id: conditions
                for block_id, conditions in blocks.items()
                if next(iter(conditions.values()))["kind"] == "fuse"
                and next(iter(conditions.values()))["task"] == task
                and (
                    source_version is None
                    or next(iter(conditions.values()))["source_version"] == source_version
                )
            }
            fuse[name] = [
                _contrast(
                    selected,
                    left=left,
                    right=right,
                    metrics=(
                        "delta_fitness",
                        "unconditional_improvement_rate",
                        "invalid_rate",
                    ),
                    seed=DESIGN_SEED + 1000 * task_index + 100 * source_index + i * 10,
                )
                for i, (left, right) in enumerate(comparisons)
            ]

    init: dict[str, Any] = {}
    for task_index, task in enumerate(ALL_TASKS):
        selected = {
            block_id: conditions
            for block_id, conditions in blocks.items()
            if next(iter(conditions.values()))["kind"] == "init"
            and next(iter(conditions.values()))["task"] == task
        }
        init[f"task:{task}"] = _contrast(
            selected,
            left="v101",
            right="v102",
            metrics=("fitness", "invalid_rate"),
            seed=DESIGN_SEED + 10_000 + task_index * 100,
        )

    summary = {
        "protocol_id": config["protocol_id"],
        "complete": set(by_trial) == expected and len(rows) == len(by_trial),
        "expected_trials": len(expected),
        "completed_unique_trials": len(by_trial),
        "missing_trial_ids": sorted(expected - set(by_trial)),
        "duplicate_result_rows": len(rows) - len(by_trial),
        "fuse_paired_contrasts": fuse,
        "init_paired_contrasts": init,
        "interpretation_boundary": (
            "Fuse contrasts identify one-step prompt effects conditional on sampled "
            "V10.1/V10.2 states. Init contrasts identify root-prompt effects. Neither "
            "estimates allocation, full-search, or held-out performance. Raw fitness "
            "differences are reported per task and are not pooled across tasks."
        ),
    }
    _write_json(run_dir / "analysis" / "summary.json", summary)
    return summary


def status(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "probe_config.json")
    generated = []
    for path in sorted((run_dir / "generations").glob("shard_*.jsonl")):
        generated.extend(_read_jsonl(path))
    results = _read_jsonl(run_dir / "results" / "results.jsonl")
    return {
        "protocol_id": config["protocol_id"],
        "total_trials": config["trial_count"],
        "generated_unique_trials": len({row["trial_id"] for row in generated}),
        "parse_failed": sum(row.get("generation_status") == "parse_failed" for row in generated),
        "evaluated_unique_trials": len({row["trial_id"] for row in results}),
        "valid_trials": sum(row.get("valid") is True for row in results),
        "invalid_trials": sum(row.get("valid") is False for row in results),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--anchors-per-source-task", type=int, default=6)
    prepare_parser.add_argument("--init-pairs-per-task", type=int, default=4)
    prepare_parser.add_argument("--seed", type=int, default=DESIGN_SEED)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--run-dir", type=Path, required=True)
    generate_parser.add_argument("--backend", choices=tuple(BACKENDS), required=True)
    generate_parser.add_argument("--shard-index", type=int, required=True)
    generate_parser.add_argument("--num-shards", type=int, required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--run-dir", type=Path, required=True)
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--run-dir", type=Path, required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(
            args.run_dir,
            anchors_per_source_task=args.anchors_per_source_task,
            init_pairs_per_task=args.init_pairs_per_task,
            seed=args.seed,
        )
    elif args.command == "generate":
        if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard index must be in [0, num_shards)")
        generate_shard(
            args.run_dir,
            backend=args.backend,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    elif args.command == "evaluate":
        evaluate_generated(args.run_dir)
    elif args.command == "analyze":
        summary = analyze(args.run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps(status(args.run_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

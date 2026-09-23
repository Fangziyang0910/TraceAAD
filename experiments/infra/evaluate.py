"""Evaluate the best program from search runs on held-out test sets.

Unified entry for the five tasks. Task and method are derived from each
run directory's `run_config.json`.

    uv run python -m experiments.infra.evaluate <run_dir> [...] --output-dir DIR

Per-task options:
    tsp_construct       --units 50,100,200  --timeout 1000  --workers 16
    cvrp_aco / op_aco   --units test_50,test_100,test_200  --workers 16
    online_bin_packing  --units 1k_100,5k_100,...,10k_500  --max-sample-order N
    vrptw_construct     --units 50,100,200  (eval seed from generated_data_config)

Batch mode evaluates every run dir on every unit and writes `results.json`
under --output-dir. Single-run print mode (tsp_construct only) prints the
train-sanity and per-unit scores without writing files.

Searches must be finished by default. `--allow-incomplete` is an explicit
diagnostic mode for interrupted or in-progress searches; the output records
each search status and consumed slot count.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.evaluate import SecureEvaluator  # noqa: E402
from benchmarks.cvrp_aco import (  # noqa: E402
    CVRPACOEvaluation,
    load_split_instances as load_cvrp_instances,
)
from benchmarks.generated_data_config import (  # noqa: E402
    get_generated_task_kwargs,
)
from benchmarks.online_bin_packing import OBPEvaluation  # noqa: E402
from benchmarks.op_aco import (  # noqa: E402
    OPACOEvaluation,
    load_split_instances as load_op_instances,
)
from benchmarks.tsp_construct import TSPEvaluation  # noqa: E402
from benchmarks.vrptw_construct import VRPTWEvaluation  # noqa: E402

from experiments.infra.artifacts import (  # noqa: E402
    load_run_summary,
    pick_best_sample,
)


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.fmean(finite) if finite else None,
        "sample_std": statistics.stdev(finite) if len(finite) >= 2 else None,
    }


def _read_run_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_task(run_dir: Path) -> str:
    task = _read_run_config(run_dir).get("task")
    if isinstance(task, str) and task.strip():
        return task.strip()
    raise ValueError(f"cannot derive task from run_config.json under {run_dir}")


def _resolve_method(run_dir: Path) -> str:
    method = _read_run_config(run_dir).get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()
    return run_dir.parent.name


def _relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# tsp_construct
# ---------------------------------------------------------------------------

TSP_TRAIN_SEED = 2024
TSP_EVAL_SEED = 2025
TSP_DEFAULT_UNITS = (50, 100, 200)
TSP_DEFAULT_TIMEOUT = 1000
TSP_DEFAULT_WORKERS = 16


def _tsp_eval_one_instance(args: tuple[str, int, Any]) -> float | None:
    program, size, dataset = args
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    task = TSPEvaluation(timeout_seconds=None, n_instance=1, problem_size=size, seed=0)
    task._datasets = [dataset]
    return task.evaluate(namespace["select_next_node"])


def tsp_eval_program(
    program: str,
    size: int,
    seed: int,
    n_instance: int = 16,
    timeout: int | None = 120,
    workers: int = 1,
) -> tuple[float | None, float]:
    if workers <= 1:
        task = TSPEvaluation(
            timeout_seconds=timeout, n_instance=n_instance, problem_size=size, seed=seed
        )
        evaluator = SecureEvaluator(task)
        return evaluator.evaluate_program_record_time(program)

    task = TSPEvaluation(
        timeout_seconds=None, n_instance=n_instance, problem_size=size, seed=seed
    )
    worker_args = [(program, size, dataset) for dataset in task._datasets]
    started_at = time.time()
    pool = multiprocessing.get_context("spawn").Pool(processes=min(workers, n_instance))
    try:
        if timeout is None:
            scores = pool.map(_tsp_eval_one_instance, worker_args)
        else:
            scores = pool.map_async(_tsp_eval_one_instance, worker_args).get(
                timeout=timeout
            )
    except multiprocessing.TimeoutError:
        pool.terminate()
        pool.join()
        return None, time.time() - started_at
    else:
        pool.close()
        pool.join()
        if any(score is None for score in scores):
            return None, time.time() - started_at
        return float(sum(scores) / len(scores)), time.time() - started_at


def _tsp_parse_units(text: str) -> list[int]:
    units = [int(x) for x in text.split(",") if x.strip()]
    if not units:
        raise ValueError("--units must contain at least one size")
    return units


def _make_tsp_spec() -> dict[str, Any]:
    def eval_unit(
        program: str, size: int, timeout: int, workers: int
    ) -> tuple[float, float]:
        score, seconds = tsp_eval_program(
            program, size, TSP_EVAL_SEED, timeout=timeout, workers=workers
        )
        if score is None:
            raise RuntimeError(f"evaluation timed out for tsp{size}")
        return score, seconds

    def train_sanity(program: str, timeout: int, workers: int) -> tuple[float, float]:
        score, seconds = tsp_eval_program(
            program, 50, TSP_TRAIN_SEED, timeout=timeout, workers=workers
        )
        if score is None:
            raise RuntimeError("train sanity eval timed out")
        return score, seconds

    return {
        "default_units": TSP_DEFAULT_UNITS,
        "parse_units": _tsp_parse_units,
        "unit_key": lambda size: f"tsp{size}",
        "unit_label": lambda size: f"TSP{size}",
        "eval_unit": eval_unit,
        "train_sanity": train_sanity,
        "train_sanity_label": "[sanity train  | size=50 seed=2024]",
        "eval_seed_label": "[test tsp{size:<3}| seed=2025]",
        "objective": lambda score: -score,
        "row_extra": {},
        "container_key": "eval_results_by_size",
        "score_semantics": "score is negative mean tour length; higher score is better and lower objective is better",
        "print_mode": True,
    }


# ---------------------------------------------------------------------------
# cvrp_aco / op_aco (identical harness, different evaluator and semantics)
# ---------------------------------------------------------------------------

ACO_DEFAULT_UNITS = ("test_50", "test_100", "test_200")
ACO_DEFAULT_WORKERS = 16

_ACO_WORKER: dict[str, Any] = {}


def _aco_init_worker(
    program: str,
    split: str,
    eval_cls: type,
    n_ants: int,
    n_iterations: int,
    aco_seed: int,
) -> None:
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    _ACO_WORKER["heuristic"] = namespace["heuristics"]
    _ACO_WORKER["evaluator"] = eval_cls(
        split=split,
        n_ants=n_ants,
        n_iterations=n_iterations,
        aco_seed=aco_seed,
        timeout_seconds=None,
    )


def _aco_solve_instance(args: tuple[int, Any]) -> tuple[int, float]:
    index, instance = args
    return index, float(
        _ACO_WORKER["evaluator"]._solve_instance(
            instance, _ACO_WORKER["heuristic"], index
        )
    )


def _make_aco_eval(
    eval_cls: type,
    load_instances: Callable[[str], tuple[list[Any], Any]],
    n_ants: int,
    n_iterations: int,
    aco_seed: int,
    score_transform: Callable[[float], float] = lambda value: value,
) -> Callable[[str, str, int], tuple[float, list[float], float]]:
    def eval_program(
        program: str, split: str, workers: int
    ) -> tuple[float, list[float], float]:
        instances, _ = load_instances(split)
        started_at = time.time()
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=min(workers, len(instances)),
            initializer=_aco_init_worker,
            initargs=(program, split, eval_cls, n_ants, n_iterations, aco_seed),
        ) as pool:
            indexed = pool.map(_aco_solve_instance, list(enumerate(instances)))
        values = [value for _, value in sorted(indexed)]
        return (
            score_transform(statistics.fmean(values)),
            values,
            time.time() - started_at,
        )

    return eval_program


def _make_aco_spec(
    task: str, eval_cls: type, n_ants: int, n_iterations: int
) -> dict[str, Any]:
    is_op = task == "op_aco"
    load_instances = load_op_instances if is_op else load_cvrp_instances
    aco_seed = 1234
    # 保持与原脚本一致的符号约定:score 越大越好(cvrp 为负均值成本,op 为均值收益),
    # objective 为原始方向(cvrp 为正值成本、越低越好,op 为正值收益、越高越好)。
    score_transform = (lambda value: value) if is_op else (lambda value: -value)
    eval_program = _make_aco_eval(
        eval_cls, load_instances, n_ants, n_iterations, aco_seed, score_transform
    )

    def eval_unit(
        program: str, split: str, timeout: int, workers: int
    ) -> tuple[float, float, list[float]]:
        score, values, seconds = eval_program(program, split, workers)
        return score, seconds, values

    def objective(score: float) -> float:
        return score if is_op else -score

    return {
        "default_units": ACO_DEFAULT_UNITS,
        "parse_units": lambda text: [x.strip() for x in text.split(",") if x.strip()],
        "unit_key": lambda split: split,
        "unit_label": lambda split: split.upper(),
        "eval_unit": eval_unit,
        "train_sanity": None,
        "container_key": "results_by_split",
        "instance_field": "instance_prizes" if is_op else "instance_costs",
        "score_semantics": (
            "score is mean collected prize across instances; higher is better"
            if is_op
            else "score is negative mean best route length; higher score is better and lower objective is better"
        ),
        "aco_config": {
            "n_ants": n_ants,
            "n_iterations": n_iterations,
            "aco_seed": aco_seed,
        },
        "objective": objective,
        "print_mode": False,
    }


# ---------------------------------------------------------------------------
# online_bin_packing
# ---------------------------------------------------------------------------

OBP_DEFAULT_UNITS = (
    (1000, 100),
    (5000, 100),
    (10000, 100),
    (1000, 500),
    (5000, 500),
    (10000, 500),
)


def _obp_scale_key(n_items: int, capacity: int) -> str:
    if n_items % 1000 == 0 and n_items >= 1000:
        return f"{n_items // 1000}k_{capacity}"
    return f"{n_items}_{capacity}"


def _obp_parse_units(text: str) -> list[tuple[int, int]]:
    scales: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "_" not in part:
            raise ValueError(
                f"invalid scale {part!r}; expected like 5k_100 or 5000_100"
            )
        left, right = part.rsplit("_", 1)
        capacity = int(right)
        if left.endswith("k") or left.endswith("K"):
            n_items = int(left[:-1]) * 1000
        else:
            n_items = int(left)
        scales.append((n_items, capacity))
    if not scales:
        raise ValueError("--units must contain at least one scale")
    return scales


def _obp_task_kwargs_for_scale(
    base_kwargs: dict[str, Any], n_items: int, capacity: int
) -> dict[str, Any]:
    kwargs = dict(base_kwargs)
    dataset_specs = kwargs.get("dataset_specs")
    if dataset_specs is None:
        kwargs.update(n_items=n_items, capacity=capacity)
        return kwargs
    for spec in dataset_specs:
        if int(spec["n_items"]) == n_items and capacity in spec["capacities"]:
            kwargs["dataset_specs"] = [
                {
                    "n_instances": int(spec["n_instances"]),
                    "n_items": n_items,
                    "capacities": [capacity],
                }
            ]
            return kwargs
    raise ValueError(
        f"scale {n_items}_{capacity} is not part of the fixed OBP test protocol"
    )


def _obp_eval_program(program: str, task_kwargs: dict[str, Any]) -> tuple[float, float]:
    namespace: dict[str, Any] = {}
    exec(program, namespace)
    if "priority" not in namespace or not callable(namespace["priority"]):
        raise RuntimeError("program does not define callable priority(...)")
    evaluator = OBPEvaluation(**task_kwargs)
    started_at = time.time()
    score = evaluator.evaluate(namespace["priority"])
    if (
        score is None
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise RuntimeError(f"evaluation returned invalid score: {score!r}")
    return float(score), time.time() - started_at


def _make_obp_spec() -> dict[str, Any]:
    def eval_unit(
        program: str, scale: tuple[int, int], timeout: int, workers: int
    ) -> tuple[float, float]:
        n_items, capacity = scale
        kwargs = _obp_task_kwargs_for_scale(
            {
                **get_generated_task_kwargs("online_bin_packing", "eval"),
                "timeout_seconds": timeout,
            },
            n_items,
            capacity,
        )
        return _obp_eval_program(program, kwargs)

    def train_sanity(program: str, timeout: int, workers: int) -> tuple[float, float]:
        kwargs = {
            **get_generated_task_kwargs("online_bin_packing", "train"),
            "timeout_seconds": timeout,
        }
        return _obp_eval_program(program, kwargs)

    return {
        "default_units": OBP_DEFAULT_UNITS,
        "parse_units": _obp_parse_units,
        "unit_key": lambda scale: _obp_scale_key(*scale),
        "unit_label": lambda scale: _obp_scale_key(*scale),
        "eval_unit": eval_unit,
        "train_sanity": train_sanity,
        "container_key": "eval_results_by_scale",
        "score_semantics": (
            "OBPEvaluation returns negative mean bins used across instances; "
            "higher score is better. bins_used_mean = -score."
        ),
        "objective": lambda score: -score,
        "print_mode": False,
    }


# ---------------------------------------------------------------------------
# generated-instance template task (vrptw)
# ---------------------------------------------------------------------------

_GENERATED_TASK_EVAL_CLASSES: dict[str, type] = {
    "vrptw_construct": VRPTWEvaluation,
}

_GENERATED_SCORE_SEMANTICS = {
    "vrptw_construct": (
        "score is negative mean total distance across held-out instances; "
        "higher is better and lower objective is better"
    ),
}

VRPTW_DEFAULT_UNITS = (50, 100, 200)


def _parse_vrptw_units(text: str) -> list[int]:
    units = [int(value) for value in text.split(",") if value.strip()]
    if not units or any(value <= 0 for value in units):
        raise ValueError("VRPTW --units must contain positive problem sizes")
    return units


def _vrptw_eval_kwargs(problem_size: int, timeout: int) -> dict[str, Any]:
    kwargs = get_generated_task_kwargs("vrptw_construct", "eval")
    kwargs["problem_size"] = problem_size
    kwargs["timeout_seconds"] = timeout
    return kwargs


def _make_generated_spec(task: str) -> dict[str, Any]:
    eval_cls = _GENERATED_TASK_EVAL_CLASSES[task]

    def eval_unit(
        program: str, unit: int, timeout: int, workers: int
    ) -> tuple[float, float]:
        kwargs = _vrptw_eval_kwargs(unit, timeout)
        evaluator = SecureEvaluator(eval_cls(**kwargs))
        score, seconds = evaluator.evaluate_program_record_time(program)
        if score is None:
            raise RuntimeError("generated-task eval returned no score")
        return float(score), seconds

    def train_sanity(program: str, timeout: int, workers: int) -> tuple[float, float]:
        evaluator = SecureEvaluator(
            eval_cls(**get_generated_task_kwargs(task, "train"))
        )
        score, seconds = evaluator.evaluate_program_record_time(program)
        if score is None:
            raise RuntimeError("generated-task train sanity eval returned no score")
        return float(score), seconds

    return {
        "default_units": VRPTW_DEFAULT_UNITS,
        "parse_units": _parse_vrptw_units,
        "unit_key": lambda unit: f"vrptw{unit}",
        "unit_label": lambda unit: f"VRPTW{unit}",
        "eval_unit": eval_unit,
        "eval_config": _vrptw_eval_kwargs,
        "train_sanity": train_sanity,
        "train_sanity_label": "[sanity train  | seed 2024]",
        "container_key": "eval_results_by_split",
        "score_semantics": _GENERATED_SCORE_SEMANTICS[task],
        "objective": lambda score: -score,
        "row_extra": {},
        "print_mode": False,
    }


TASK_SPECS: dict[str, dict[str, Any]] = {
    "tsp_construct": _make_tsp_spec(),
    "cvrp_aco": _make_aco_spec(
        "cvrp_aco", CVRPACOEvaluation, n_ants=30, n_iterations=100
    ),
    "op_aco": _make_aco_spec("op_aco", OPACOEvaluation, n_ants=20, n_iterations=50),
    "online_bin_packing": _make_obp_spec(),
}
for _generated_task in _GENERATED_TASK_EVAL_CLASSES:
    TASK_SPECS[_generated_task] = _make_generated_spec(_generated_task)


def _run_batch(
    task: str,
    spec: dict[str, Any],
    run_dirs: list[Path],
    output_dir: Path,
    units: list[Any],
    timeout: int,
    workers: int,
    max_sample_order: int | None,
    allow_incomplete: bool,
) -> None:
    methods = {_resolve_method(run_dir) for run_dir in run_dirs}
    if len(methods) != 1:
        raise ValueError(
            f"all run directories must belong to one method: {sorted(methods)}"
        )
    method = next(iter(methods))
    output_dir.mkdir(parents=True, exist_ok=True)

    model = "unknown"
    run_records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        best, all_samples = pick_best_sample(
            run_dir,
            max_sample_order=max_sample_order,
            allow_incomplete=allow_incomplete,
        )
        program = str(best["program"])
        sample_order = int(best["sample_order"])
        program_path = output_dir / f"{run_dir.name}_sample_{sample_order}_program.py"
        program_path.write_text(program.rstrip() + "\n", encoding="utf-8")
        config = _read_run_config(run_dir)
        llm_config = config.get("llm")
        if isinstance(llm_config, dict):
            configured_model = llm_config.get("model")
            if isinstance(configured_model, str) and configured_model.strip():
                model = configured_model
        if model == "unknown":
            environment = config.get("generator_environment")
            if isinstance(environment, dict):
                logical_model = environment.get("logical_model_name")
                if isinstance(logical_model, str) and logical_model.strip():
                    model = logical_model
        record: dict[str, Any] = {
            "run_dir": _relative_to_root(run_dir),
            "run_name": run_dir.name,
            "num_valid_samples": len(all_samples),
            "best_sample_order": sample_order,
            "best_operator": best.get("operator"),
            "train_artifact_score": float(best["score"]),
            "program_path": _relative_to_root(program_path),
            "program": program,
        }
        search_summary = load_run_summary(
            run_dir,
            require_finished=not allow_incomplete,
        )
        record["search_status"] = search_summary.get("status")
        record["search_budget_slots"] = search_summary.get("budget_slots")
        record["search_aborted"] = search_summary.get("search_aborted")
        if max_sample_order is not None:
            record["max_sample_order"] = max_sample_order
        if spec.get("train_sanity") is not None:
            train_score, train_seconds = spec["train_sanity"](program, timeout, workers)
            record["train_recomputed_score"] = float(train_score)
            record["train_eval_seconds"] = train_seconds
        run_records.append(record)

    container: dict[str, Any] = {}
    for unit in units:
        key = spec["unit_key"](unit)
        rows: list[dict[str, Any]] = []
        for row in run_records:
            try:
                result = spec["eval_unit"](row["program"], unit, timeout, workers)
            except Exception as exc:
                rows.append(
                    {
                        "run_name": row["run_name"],
                        "best_sample_order": row["best_sample_order"],
                        "best_operator": row["best_operator"],
                        "eval_failed": str(exc),
                        "program_path": row["program_path"],
                    }
                )
                continue
            if isinstance(result, tuple) and len(result) == 3:
                score, seconds, values = result
            else:
                score, seconds = result
                values = None
            entry: dict[str, Any] = {
                "run_name": row["run_name"],
                "best_sample_order": row["best_sample_order"],
                "best_operator": row["best_operator"],
                "eval_score": score,
                "eval_objective": spec["objective"](score),
                "eval_seconds": seconds,
                "program_path": row["program_path"],
            }
            if values is not None:
                entry[spec["instance_field"]] = values
            if task == "online_bin_packing":
                entry["bins_used_mean"] = -score
            rows.append(entry)

        ok_rows = [entry for entry in rows if "eval_objective" in entry]
        objectives = [entry["eval_objective"] for entry in ok_rows]
        scores = [entry["eval_score"] for entry in ok_rows]
        container[key] = {
            "results": rows,
            "summary": {
                "num_runs": len(rows),
                "num_successful_eval_runs": len(ok_rows),
                "mean_eval_score": _mean_std(scores)["mean"],
                "sample_std_eval_score": _mean_std(scores)["sample_std"],
                "mean_eval_objective": _mean_std(objectives)["mean"],
                "sample_std_eval_objective": _mean_std(objectives)["sample_std"],
            },
        }
        if task == "online_bin_packing":
            n_items, capacity = unit
            container[key]["n_items"] = n_items
            container[key]["capacity"] = capacity
            container[key]["eval_config"] = _obp_task_kwargs_for_scale(
                {
                    **get_generated_task_kwargs("online_bin_packing", "eval"),
                    "timeout_seconds": timeout,
                },
                n_items,
                capacity,
            )
        elif task == "tsp_construct":
            container[key]["problem_size"] = unit
            container[key]["eval_config"] = {
                "n_instance": 16,
                "problem_size": unit,
                "seed": TSP_EVAL_SEED,
                "timeout_seconds": timeout,
                "workers": workers,
                "evaluation_mode": "complete_run",
            }
        elif task in _GENERATED_TASK_EVAL_CLASSES:
            container[key]["problem_size"] = unit
            container[key]["eval_config"] = spec["eval_config"](unit, timeout)
        else:
            container[key]["split"] = unit
            container[key]["metadata"] = (
                load_op_instances(unit)[1]
                if task == "op_aco"
                else load_cvrp_instances(unit)[1]
            )
            container[key]["config"] = {**spec["aco_config"], "workers": workers}

    for unit, key in [(unit, spec["unit_key"](unit)) for unit in units]:
        if task == "online_bin_packing":
            n_items, capacity = unit
            kwargs = _obp_task_kwargs_for_scale(
                {
                    **get_generated_task_kwargs("online_bin_packing", "eval"),
                    "timeout_seconds": timeout,
                },
                n_items,
                capacity,
            )
            container[key]["eval_config"] = kwargs

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task": task,
        "method": method,
        "model": model,
        "source": (
            (
                f"{len(run_records)} {method} {task} repeat(s); "
                "incomplete search runs explicitly allowed"
            )
            if allow_incomplete
            else f"{len(run_records)} completed {method} {task} repeat(s)"
        ),
        "allow_incomplete_search_runs": allow_incomplete,
        "score_semantics": spec["score_semantics"],
        "run_records": [
            {k: v for k, v in row.items() if k != "program"} for row in run_records
        ],
    }
    if task == "tsp_construct":
        payload["problem_sizes"] = units
        payload["eval_timeout_seconds"] = timeout
        payload["eval_workers"] = workers
        payload["eval_results_by_size"] = container
    elif task == "online_bin_packing":
        payload["test_scales"] = [spec["unit_key"](unit) for unit in units]
        payload["eval_timeout_seconds"] = timeout
        payload["max_sample_order"] = max_sample_order
        payload["split_configs"] = {
            "train": {
                **get_generated_task_kwargs("online_bin_packing", "train"),
                "timeout_seconds": timeout,
            },
            "eval_base": {
                **get_generated_task_kwargs("online_bin_packing", "eval"),
                "timeout_seconds": timeout,
            },
        }
        payload["eval_results_by_scale"] = container
        train_scores = [float(row["train_recomputed_score"]) for row in run_records]
        payload["summary"] = {
            "num_runs": len(run_records),
            "mean_train_recomputed_score": _mean_std(train_scores)["mean"],
            "sample_std_train_recomputed_score": _mean_std(train_scores)["sample_std"],
        }
    elif task == "vrptw_construct":
        payload["problem_sizes"] = units
        payload["results_by_size"] = container
    else:
        payload["results_by_split"] = container

    output_path = output_dir / "results.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _print_single_run(
    task: str,
    spec: dict[str, Any],
    run_dir: Path,
    units: list[Any],
    timeout: int,
    workers: int,
    sample_order: int | None,
    allow_incomplete: bool,
) -> None:
    picked, samples = pick_best_sample(
        run_dir,
        sample_order=sample_order,
        allow_incomplete=allow_incomplete,
    )
    program = picked["program"]
    print(f"run_dir     : {run_dir}")
    print(
        f"valid samples: {len(samples)}; using sample_order={picked['sample_order']} "
        f"(logged score={picked['score']:.6f}, op={picked.get('operator')})"
    )
    train_score, train_seconds = spec["train_sanity"](program, timeout, workers)
    print(
        f"  {spec['train_sanity_label']} score={train_score:.6f}  "
        f"(logged {picked['score']:.6f}, diff {abs(train_score - picked['score']):.2e})"
    )
    for unit in units:
        score, seconds = spec["eval_unit"](program, unit, timeout, workers)[:2]
        label = spec["eval_seed_label"].format(size=unit)
        print(f"  {label} score={score:.6f}  (eval_time {seconds:.2f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", help="finished run directories")
    ap.add_argument(
        "--task",
        choices=sorted(TASK_SPECS),
        default=None,
        help="default: derive from run_config.json",
    )
    ap.add_argument(
        "--units", default=None, help="per-task unit list; default per task"
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=TSP_DEFAULT_TIMEOUT,
        help="timeout per unit in seconds",
    )
    ap.add_argument("--workers", type=int, default=ACO_DEFAULT_WORKERS)
    ap.add_argument(
        "--sample-order", type=int, default=None, help="single-run print mode only"
    )
    ap.add_argument(
        "--max-sample-order",
        type=int,
        default=None,
        help="only consider candidates up to this search evaluation (obp)",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "evaluate the best artifact currently present in interrupted or "
            "in-progress searches and record that status in results.json"
        ),
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.max_sample_order is not None and args.max_sample_order <= 0:
        raise ValueError("--max-sample-order must be positive")

    run_dirs = [Path(p).resolve() for p in args.run_dirs]
    tasks = {_resolve_task(run_dir) for run_dir in run_dirs}
    if args.task is not None:
        tasks.add(args.task)
    if len(tasks) != 1:
        raise ValueError(
            f"cannot derive one task from run dirs {sorted(tasks)}; pass --task explicitly"
        )
    task = next(iter(tasks))
    spec = TASK_SPECS[task]
    units = (
        spec["parse_units"](args.units)
        if args.units is not None
        else list(spec["default_units"])
    )

    if args.output_dir is not None:
        if args.sample_order is not None:
            raise ValueError(
                "--sample-order is only supported in single-run print mode"
            )
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
        _run_batch(
            task,
            spec,
            run_dirs,
            output_dir,
            units,
            args.timeout,
            args.workers,
            args.max_sample_order,
            args.allow_incomplete,
        )
        return

    if not spec["print_mode"]:
        raise ValueError(
            "print mode is only supported for tsp_construct; use --output-dir for batch"
        )
    if len(run_dirs) != 1:
        raise ValueError(
            "print mode accepts exactly one run_dir; use --output-dir for batch"
        )
    _print_single_run(
        task,
        spec,
        run_dirs[0],
        units,
        args.timeout,
        args.workers,
        args.sample_order,
        args.allow_incomplete,
    )


if __name__ == "__main__":
    main()

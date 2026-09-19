"""E1 profile core reused from d234021251800e1ef9979993c799717ddbb0e577 BehaveSim v3, with training-only panels.

The measured object is a problem-solving trajectory (PSTraj): a sequence of
intermediate solutions produced by the same decision process used by the task
evaluator. Pairwise trajectory distance follows the paper definition:
normalized solution edit distance, DTW alignment, then division by the shorter
trajectory length. Raw artifacts are local-only under this experiment's ``raw/`` directory.
"""

from __future__ import annotations

import copy
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numba
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.task.optimization.cvrp_aco.evaluation import (  # noqa: E402
    ACO as CVRPACO,
    CVRPACOEvaluation,
)
from llm4ad.task.optimization.generated_data_config import (  # noqa: E402
    get_generated_task_kwargs,
)
from llm4ad.task.optimization.online_bin_packing.generate_weibull_instances import (  # noqa: E402
    generate_weibull_multiscale_dataset,
)
from llm4ad.task.optimization.op_aco.evaluation import (  # noqa: E402
    ACO as OPACO,
    OPACOEvaluation,
)
from llm4ad.task.optimization.tsp_construct.evaluation import (  # noqa: E402
    TSPEvaluation,
)
from llm4ad.task.optimization.vrptw_construct.evaluation import (  # noqa: E402
    VRPTWEvaluation,
)

SCHEMA_VERSION = 3
TASKS = (
    "tsp_construct",
    "op_aco",
    "online_bin_packing",
    "cvrp_aco",
    "vrptw_construct",
)
PREFIX_TASKS = {"tsp_construct", "online_bin_packing", "vrptw_construct"}
DEFAULT_SAMPLE_SIZE = {
    "tsp_construct": 128,
    "op_aco": 32,
    "online_bin_packing": 128,
    "cvrp_aco": 32,
    "vrptw_construct": 128,
}
DEFAULT_TIMEOUT_SECONDS = {
    "tsp_construct": 30.0,
    "op_aco": 180.0,
    "online_bin_packing": 30.0,
    "cvrp_aco": 240.0,
    "vrptw_construct": 60.0,
}
DEFAULT_TRAJECTORY_POINTS = {
    "tsp_construct": 12,
    "op_aco": 5,
    "online_bin_packing": 12,
    "cvrp_aco": 5,
    "vrptw_construct": 12,
}
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 0.96, 0.05))
PROGRAM_RANDOM_SEED = 730_241
# Historical E1 callers keep the frozen protocol through the function defaults.
# New population-geometry analyses should request these calibrated settings.
CALIBRATED_OBP_SCALE = "matched"
CALIBRATED_ACO_SEED_OFFSETS = (0, 10_000, 20_000, 30_000)

_GLOBAL_TASK: str | None = None
_GLOBAL_DATA: dict[str, Any] | None = None
_GLOBAL_MAX_POINTS = 0
_GLOBAL_TIMEOUT_SECONDS = 0.0


class ProfileError(RuntimeError):
    """A candidate cannot produce a valid evaluator-consistent PSTraj."""


def _uniform_trajectory_indices(length: int, max_points: int) -> list[int]:
    if length <= 0:
        raise ProfileError("empty PSTraj")
    if length <= max_points:
        return list(range(length))
    indices = np.rint(np.linspace(0, length - 1, max_points)).astype(int)
    unique_indices = list(dict.fromkeys(int(index) for index in indices))
    if len(unique_indices) != max_points:
        raise AssertionError("trajectory sampling produced duplicate points")
    return unique_indices


def _uniform_trajectory_sample(states: Sequence[Sequence[int]], max_points: int) -> list[list[int]]:
    indices = _uniform_trajectory_indices(len(states), max_points)
    return [list(states[index]) for index in indices]


def _build_probe_data(
    task: str,
    panel: str,
    *,
    obp_scale: str = "compact",
    aco_seed_offsets: Sequence[int] = (0,),
) -> dict[str, Any]:
    if panel not in {"A", "B"}:
        raise ValueError(f"unknown probe panel: {panel}")
    seed = 42 if panel == "A" else 43

    if task == "tsp_construct":
        evaluator = TSPEvaluation(n_instance=4, problem_size=50, seed=seed)
        return {
            "instances": evaluator._datasets,
            "evaluator": evaluator,
            "probe_metadata": {"panel": panel, "seed": seed, "instances": 4, "size": 50},
        }

    if task == "vrptw_construct":
        evaluator = VRPTWEvaluation(n_instance=4, problem_size=50, seed=seed)
        return {
            "instances": evaluator._datasets,
            "evaluator": evaluator,
            "probe_metadata": {"panel": panel, "seed": seed, "instances": 4, "size": 50},
        }

    if task == "op_aco":
        split = "train"
        evaluator = OPACOEvaluation(split=split, n_workers=1)
        if panel == "B":
            evaluator._datasets = np.random.RandomState(20260907).rand(4, 50, 2)
        instance_indices = list(range(4))
        return {
            "instances": evaluator._datasets[instance_indices],
            "instance_indices": instance_indices,
            "evaluator": evaluator,
            "probe_metadata": {
                "panel": panel,
                "split": split,
                "role": "training-probe",
                "instances": 4,
                "size": 50,
                "instance_indices": instance_indices,
                "ants": evaluator.n_ants,
                "iterations": evaluator.n_iterations,
                "aco_seed": evaluator.aco_seed,
                "aco_seed_offsets": list(aco_seed_offsets),
                "trajectory_order": "instance-major, seed-offset-minor",
                "new_op_probe_seed": 20260907 if task == "op_aco" and panel == "B" else None,
            },
        }

    if task == "cvrp_aco":
        split = "train"
        evaluator = CVRPACOEvaluation(split=split, n_workers=1)
        instance_indices = list(range(4)) if panel == "A" else list(range(4, 8))
        return {
            "instances": evaluator._datasets[instance_indices],
            "instance_indices": instance_indices,
            "evaluator": evaluator,
            "probe_metadata": {
                "panel": panel,
                "split": split,
                "role": "training-probe",
                "instances": 4,
                "size": 50,
                "instance_indices": instance_indices,
                "ants": evaluator.n_ants,
                "iterations": evaluator.n_iterations,
                "aco_seed": evaluator.aco_seed,
                "aco_seed_offsets": list(aco_seed_offsets),
                "trajectory_order": "instance-major, seed-offset-minor",
                "new_op_probe_seed": 20260907 if task == "op_aco" and panel == "B" else None,
            },
        }

    if task == "online_bin_packing":
        if obp_scale == "compact":
            dataset_specs = [
                {"n_instances": 2, "n_items": 256, "capacities": [100, 500]}
            ]
        elif obp_scale == "matched":
            dataset_specs = get_generated_task_kwargs(task, "train")["dataset_specs"]
        else:
            raise ValueError(f"unknown OBP probe scale: {obp_scale}")
        dataset = generate_weibull_multiscale_dataset(
            dataset_specs,
            seed=seed,
        )
        return {
            "instances": list(dataset.values()),
            "probe_metadata": {
                "panel": panel,
                "seed": seed,
                "instances": 4,
                "scale": obp_scale,
                "item_counts": sorted({spec["n_items"] for spec in dataset_specs}),
                "capacities": [100, 500],
            },
        }

    raise ValueError(f"unsupported task: {task}")


def _init_worker(
    task: str,
    panel: str,
    max_points: int,
    timeout_seconds: float,
    *,
    obp_scale: str = "compact",
    aco_seed_offsets: Sequence[int] = (0,),
) -> None:
    global _GLOBAL_TASK, _GLOBAL_DATA, _GLOBAL_MAX_POINTS, _GLOBAL_TIMEOUT_SECONDS
    _GLOBAL_TASK = task
    _GLOBAL_DATA = _build_probe_data(
        task,
        panel,
        obp_scale=obp_scale,
        aco_seed_offsets=aco_seed_offsets,
    )
    _GLOBAL_MAX_POINTS = max_points
    _GLOBAL_TIMEOUT_SECONDS = timeout_seconds


def _timeout_handler(signum, frame) -> None:  # noqa: ARG001
    raise TimeoutError("candidate profiling timed out")


def _reset_program_randomness(probe_index: int) -> None:
    seed = PROGRAM_RANDOM_SEED + probe_index
    random.seed(seed)
    np.random.seed(seed)


def _coerce_node(value: Any) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    converted = int(value)
    if isinstance(value, np.ndarray) and value.size != 1:
        raise ProfileError("candidate returned a non-scalar node")
    return converted


def _profile_tsp(function: Any) -> tuple[list[list[list[int]]], float]:
    assert _GLOBAL_DATA is not None
    trajectories = []
    costs = []
    evaluator: TSPEvaluation = _GLOBAL_DATA["evaluator"]
    for probe_index, (coordinates, original_distances) in enumerate(_GLOBAL_DATA["instances"]):
        _reset_program_randomness(probe_index)
        distances = np.asarray(original_distances).copy()
        neighbor_matrix = evaluator.generate_neighborhood_matrix(coordinates)
        route = [0]
        states = [route.copy()]
        current_node = 0
        for _ in range(1, evaluator.problem_size - 1):
            near_nodes = neighbor_matrix[current_node][1:]
            unvisited = near_nodes[~np.isin(near_nodes, route)]
            next_node = _coerce_node(function(current_node, 0, unvisited, distances))
            if next_node in route or next_node < 0 or next_node >= evaluator.problem_size:
                raise ProfileError(f"invalid TSP next node: {next_node}")
            current_node = next_node
            route.append(next_node)
            states.append(route.copy())
        remaining = [node for node in range(evaluator.problem_size) if node not in route]
        if len(remaining) != 1:
            raise ProfileError("TSP did not leave exactly one final node")
        route.append(remaining[0])
        states.append(route.copy())
        trajectories.append(_uniform_trajectory_sample(states, _GLOBAL_MAX_POINTS))
        costs.append(float(evaluator.tour_cost(coordinates, route, evaluator.problem_size)))
    return trajectories, -float(np.mean(costs))


def _profile_obp(function: Any) -> tuple[list[list[list[int]]], float]:
    assert _GLOBAL_DATA is not None
    trajectories = []
    used_bins = []
    for probe_index, instance in enumerate(_GLOBAL_DATA["instances"]):
        _reset_program_randomness(probe_index)
        capacity = int(instance["capacity"])
        items = np.asarray(instance["items"])
        bins = np.full(int(instance["num_items"]), capacity, dtype=np.int64)
        choices: list[int] = []
        states: list[list[int]] = []
        sample_indices = set(_uniform_trajectory_indices(len(items), _GLOBAL_MAX_POINTS))
        for item_index, item in enumerate(items):
            valid_indices = np.nonzero((bins - item) >= 0)[0]
            priorities = np.asarray(function(item, bins[valid_indices].copy()))
            if priorities.ndim != 1 or len(priorities) != len(valid_indices):
                raise ProfileError("OBP priority returned the wrong shape")
            if not np.all(np.isfinite(priorities)):
                raise ProfileError("OBP priority returned a non-finite value")
            best_bin = int(valid_indices[int(np.argmax(priorities))])
            bins[best_bin] -= int(item)
            choices.append(best_bin)
            if item_index in sample_indices:
                states.append(choices.copy())
        trajectories.append(states)
        used_bins.append(int(np.count_nonzero(bins != capacity)))
    return trajectories, -float(np.mean(used_bins))


def _profile_vrptw(function: Any) -> tuple[list[list[list[int]]], float]:
    assert _GLOBAL_DATA is not None
    trajectories = []
    costs = []
    evaluator: VRPTWEvaluation = _GLOBAL_DATA["evaluator"]
    for probe_index, raw_instance in enumerate(_GLOBAL_DATA["instances"]):
        _reset_program_randomness(probe_index)
        coordinates, distance_matrix, demands, capacity, service_time, time_windows = raw_instance
        route = [0]
        states = [route.copy()]
        current_load = 0.0
        current_node = 0
        current_time = 0.0
        unvisited = set(range(1, evaluator.problem_size + 1))

        def feasible_customers() -> np.ndarray:
            return np.array(
                [
                    node
                    for node in sorted(unvisited)
                    if current_load + demands[node] <= capacity
                    and max(
                        current_time + distance_matrix[current_node, node],
                        time_windows[node, 0],
                    )
                    <= time_windows[node, 1]
                    and max(
                        current_time + distance_matrix[current_node, node],
                        time_windows[node, 0],
                    )
                    + service_time[node]
                    + distance_matrix[node, 0]
                    <= time_windows[0, 1]
                ],
                dtype=int,
            )

        feasible = feasible_customers()
        while unvisited:
            if len(feasible) == 0:
                if current_node == 0:
                    raise ProfileError("VRPTW has no feasible customer at depot")
                route.append(0)
                states.append(route.copy())
                current_load = 0.0
                current_time = 0.0
                current_node = 0
                feasible = feasible_customers()
                continue
            next_node = _coerce_node(
                function(
                    current_node,
                    0,
                    feasible.copy(),
                    capacity - current_load,
                    current_time,
                    copy.deepcopy(demands),
                    copy.deepcopy(distance_matrix),
                    copy.deepcopy(time_windows),
                )
            )
            if next_node == 0:
                if current_node == 0:
                    raise ProfileError("VRPTW returned depot while already at depot")
                route.append(0)
                current_load = 0.0
                current_time = 0.0
                current_node = 0
            else:
                if next_node not in feasible:
                    raise ProfileError(f"invalid VRPTW next node: {next_node}")
                current_time += float(distance_matrix[current_node, next_node])
                current_time = max(current_time, float(time_windows[next_node, 0]))
                current_time += float(service_time[next_node])
                route.append(next_node)
                current_load += float(demands[next_node])
                unvisited.remove(next_node)
                current_node = next_node
            states.append(route.copy())
            feasible = feasible_customers()
        if route[-1] != 0:
            route.append(0)
            states.append(route.copy())
        if len(set(route)) != evaluator.problem_size + 1:
            raise ProfileError("VRPTW route did not visit every customer")
        trajectories.append(_uniform_trajectory_sample(states, _GLOBAL_MAX_POINTS))
        costs.append(
            float(evaluator.tour_cost(distance_matrix, route, service_time, time_windows))
        )
    return trajectories, -float(np.mean(costs))


def _trim_op_route(route: np.ndarray, dummy_node: int) -> list[int]:
    result = []
    for value in route.tolist():
        node = int(value)
        if node == dummy_node:
            break
        result.append(node)
    return result


def _trim_cvrp_route(route: np.ndarray) -> list[int]:
    result = [int(value) for value in route.tolist()]
    while len(result) > 1 and result[-1] == 0 and result[-2] == 0:
        result.pop()
    return result


def _profile_op_aco(function: Any) -> tuple[list[list[list[int]]], float]:
    assert _GLOBAL_DATA is not None
    evaluator: OPACOEvaluation = _GLOBAL_DATA["evaluator"]
    trajectories = []
    final_scores = []
    offsets = _GLOBAL_DATA["probe_metadata"]["aco_seed_offsets"]
    n_instances = len(_GLOBAL_DATA["instances"])
    for instance_position, (instance_index, coordinates) in enumerate(
        zip(_GLOBAL_DATA["instance_indices"], _GLOBAL_DATA["instances"])
    ):
        for seed_position, seed_offset in enumerate(offsets):
            _reset_program_randomness(instance_position + seed_position * n_instances)
            prizes, distances, prior = evaluator._build_prior(coordinates, function)
            aco = OPACO(
                prizes,
                distances,
                evaluator.max_len,
                prior,
                n_ants=evaluator.n_ants,
                rng=np.random.default_rng(
                    evaluator.aco_seed + instance_index + seed_offset
                ),
            )
            best_score = -float("inf")
            best_route: list[int] | None = None
            states = []
            for _ in range(evaluator.n_iterations):
                solutions = aco._gen_sol()
                objectives = aco._gen_sol_obj(solutions)
                iteration_best = int(np.argmax(objectives))
                iteration_score = float(objectives[iteration_best])
                if iteration_score > best_score:
                    best_score = iteration_score
                    best_route = _trim_op_route(solutions[:, iteration_best], aco.n)
                if best_route is None:
                    raise ProfileError("OP-ACO produced no incumbent route")
                states.append(best_route.copy())
                aco.alltime_best_obj = max(aco.alltime_best_obj, iteration_score)
                aco._update_pheromone(solutions.T, objectives)
            trajectories.append(_uniform_trajectory_sample(states, _GLOBAL_MAX_POINTS))
            final_scores.append(best_score)
    return trajectories, float(np.mean(final_scores))


def _profile_cvrp_aco(function: Any) -> tuple[list[list[list[int]]], float]:
    assert _GLOBAL_DATA is not None
    evaluator: CVRPACOEvaluation = _GLOBAL_DATA["evaluator"]
    trajectories = []
    final_costs = []
    offsets = _GLOBAL_DATA["probe_metadata"]["aco_seed_offsets"]
    n_instances = len(_GLOBAL_DATA["instances"])
    for instance_position, (instance_index, instance) in enumerate(
        zip(_GLOBAL_DATA["instance_indices"], _GLOBAL_DATA["instances"])
    ):
        for seed_position, seed_offset in enumerate(offsets):
            _reset_program_randomness(instance_position + seed_position * n_instances)
            distances, demands, prior = evaluator._build_prior(instance, function)
            aco = CVRPACO(
                distances,
                demands,
                prior,
                evaluator.capacity,
                n_ants=evaluator.n_ants,
                rng=np.random.default_rng(
                    evaluator.aco_seed + instance_index + seed_offset
                ),
            )
            best_cost = float("inf")
            best_route: list[int] | None = None
            states = []
            for _ in range(evaluator.n_iterations):
                paths = aco._generate_paths()
                costs = aco._path_costs(paths)
                iteration_best = int(np.argmin(costs))
                iteration_cost = float(costs[iteration_best])
                if iteration_cost < best_cost:
                    best_cost = iteration_cost
                    best_route = _trim_cvrp_route(paths[:, iteration_best])
                if best_route is None:
                    raise ProfileError("CVRP-ACO produced no incumbent route")
                states.append(best_route.copy())
                aco.lowest_cost = min(aco.lowest_cost, iteration_cost)
                aco._update_pheromone(paths, costs)
            trajectories.append(_uniform_trajectory_sample(states, _GLOBAL_MAX_POINTS))
            final_costs.append(best_cost)
    return trajectories, -float(np.mean(final_costs))


def _profile_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    assert _GLOBAL_TASK is not None
    started_at = time.perf_counter()
    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, _GLOBAL_TIMEOUT_SECONDS)
        _reset_program_randomness(-1)
        namespace: dict[str, Any] = {"np": np}
        exec(candidate["code"], namespace)
        function_name = "priority" if _GLOBAL_TASK == "online_bin_packing" else (
            "heuristics" if _GLOBAL_TASK in {"op_aco", "cvrp_aco"} else "select_next_node"
        )
        function = namespace.get(function_name)
        if not callable(function):
            raise ProfileError(f"missing callable {function_name}")
        if _GLOBAL_TASK == "tsp_construct":
            trajectories, probe_score = _profile_tsp(function)
        elif _GLOBAL_TASK == "online_bin_packing":
            trajectories, probe_score = _profile_obp(function)
        elif _GLOBAL_TASK == "vrptw_construct":
            trajectories, probe_score = _profile_vrptw(function)
        elif _GLOBAL_TASK == "op_aco":
            trajectories, probe_score = _profile_op_aco(function)
        elif _GLOBAL_TASK == "cvrp_aco":
            trajectories, probe_score = _profile_cvrp_aco(function)
        else:
            raise ValueError(f"unsupported task: {_GLOBAL_TASK}")
        return {
            "ok": True,
            "candidate": {key: value for key, value in candidate.items() if key != "code"},
            "probe_score": probe_score,
            "trajectories": trajectories,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    except Exception as exc:
        return {
            "ok": False,
            "candidate_key": candidate["key"],
            "candidate_id": candidate["id"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "elapsed_seconds": time.perf_counter() - started_at,
        }
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def normalized_edit_distance(left: Sequence[int], right: Sequence[int]) -> float:
    """Normalized Levenshtein distance between two intermediate solutions."""
    n, m = len(left), len(right)
    if n == 0 and m == 0:
        return 0.0
    previous = list(range(m + 1))
    for i, left_value in enumerate(left, 1):
        current = [i] + [0] * m
        for j, right_value in enumerate(right, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (left_value != right_value),
            )
        previous = current
    return previous[m] / max(n, m)


def pstraj_dtw_distance(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> float:
    """Paper DTW distance normalized by the shorter PSTraj length."""
    if not left or not right:
        raise ValueError("PSTrajs must be non-empty")
    n, m = len(left), len(right)
    previous = [float("inf")] * (m + 1)
    previous[0] = 0.0
    for i in range(1, n + 1):
        current = [float("inf")] * (m + 1)
        for j in range(1, m + 1):
            local = normalized_edit_distance(left[i - 1], right[j - 1])
            current[j] = local + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return previous[m] / min(n, m)


def profile_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_trajectories = left["trajectories"]
    right_trajectories = right["trajectories"]
    if len(left_trajectories) != len(right_trajectories):
        raise ValueError("profiles use different probe counts")
    return float(
        np.mean(
            [
                pstraj_dtw_distance(left_traj, right_traj)
                for left_traj, right_traj in zip(left_trajectories, right_trajectories)
            ]
        )
    )


@numba.njit(cache=True)
def _edit_distance_with_workspace(a, len_a, b, len_b, previous, current):
    for index in range(len_b + 1):
        previous[index] = index
    for i in range(1, len_a + 1):
        current[0] = i
        left_value = a[i - 1]
        for j in range(1, len_b + 1):
            cost = 0 if left_value == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        for index in range(len_b + 1):
            previous[index] = current[index]
    return previous[len_b]


@numba.njit(cache=True)
def _prefix_probe_distance(left_states, left_lengths, right_states, right_lengths):
    left_points = 0
    right_points = 0
    for index in range(len(left_lengths)):
        if left_lengths[index] > 0:
            left_points += 1
        if right_lengths[index] > 0:
            right_points += 1
    left_final_len = left_lengths[left_points - 1]
    right_final_len = right_lengths[right_points - 1]
    edit_dp = np.empty((left_final_len + 1, right_final_len + 1), dtype=np.int32)
    for i in range(left_final_len + 1):
        edit_dp[i, 0] = i
    for j in range(right_final_len + 1):
        edit_dp[0, j] = j
    left_final = left_states[left_points - 1]
    right_final = right_states[right_points - 1]
    for i in range(1, left_final_len + 1):
        for j in range(1, right_final_len + 1):
            cost = 0 if left_final[i - 1] == right_final[j - 1] else 1
            edit_dp[i, j] = min(
                edit_dp[i - 1, j] + 1,
                edit_dp[i, j - 1] + 1,
                edit_dp[i - 1, j - 1] + cost,
            )
    previous = np.full(right_points + 1, np.inf, dtype=np.float64)
    current = np.full(right_points + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    for i in range(1, left_points + 1):
        current[0] = np.inf
        len_a = left_lengths[i - 1]
        for j in range(1, right_points + 1):
            len_b = right_lengths[j - 1]
            denominator = max(len_a, len_b)
            local = 0.0 if denominator == 0 else edit_dp[len_a, len_b] / denominator
            current[j] = local + min(previous[j], current[j - 1], previous[j - 1])
        for j in range(right_points + 1):
            previous[j] = current[j]
    return previous[right_points] / min(left_points, right_points)


@numba.njit(cache=True)
def _generic_probe_distance(left_states, left_lengths, right_states, right_lengths):
    left_points = 0
    right_points = 0
    for index in range(len(left_lengths)):
        if left_lengths[index] > 0:
            left_points += 1
        if right_lengths[index] > 0:
            right_points += 1
    max_state = max(left_states.shape[1], right_states.shape[1])
    edit_previous = np.empty(max_state + 1, dtype=np.int32)
    edit_current = np.empty(max_state + 1, dtype=np.int32)
    dtw_previous = np.full(right_points + 1, np.inf, dtype=np.float64)
    dtw_current = np.full(right_points + 1, np.inf, dtype=np.float64)
    dtw_previous[0] = 0.0
    for i in range(1, left_points + 1):
        dtw_current[0] = np.inf
        len_a = left_lengths[i - 1]
        for j in range(1, right_points + 1):
            len_b = right_lengths[j - 1]
            raw = _edit_distance_with_workspace(
                left_states[i - 1],
                len_a,
                right_states[j - 1],
                len_b,
                edit_previous,
                edit_current,
            )
            denominator = max(len_a, len_b)
            local = 0.0 if denominator == 0 else raw / denominator
            dtw_current[j] = local + min(
                dtw_previous[j], dtw_current[j - 1], dtw_previous[j - 1]
            )
        for j in range(right_points + 1):
            dtw_previous[j] = dtw_current[j]
    return dtw_previous[right_points] / min(left_points, right_points)


@numba.njit(parallel=True, cache=True)
def _pairwise_pstraj_distance(states, lengths, prefix_mode):
    n_candidates, n_probes, _, _ = states.shape
    matrix = np.zeros((n_candidates, n_candidates), dtype=np.float32)
    for left_index in numba.prange(n_candidates):
        for right_index in range(left_index + 1, n_candidates):
            total = 0.0
            for probe_index in range(n_probes):
                if prefix_mode:
                    distance = _prefix_probe_distance(
                        states[left_index, probe_index],
                        lengths[left_index, probe_index],
                        states[right_index, probe_index],
                        lengths[right_index, probe_index],
                    )
                else:
                    distance = _generic_probe_distance(
                        states[left_index, probe_index],
                        lengths[left_index, probe_index],
                        states[right_index, probe_index],
                        lengths[right_index, probe_index],
                    )
                total += distance
            value = total / n_probes
            matrix[left_index, right_index] = value
            matrix[right_index, left_index] = value
    return matrix


def _pack_profiles(profiles: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    n_candidates = len(profiles)
    n_probes = len(profiles[0]["trajectories"])
    max_points = max(
        len(trajectory)
        for profile in profiles
        for trajectory in profile["trajectories"]
    )
    max_state = max(
        len(state)
        for profile in profiles
        for trajectory in profile["trajectories"]
        for state in trajectory
    )
    states = np.full(
        (n_candidates, n_probes, max_points, max_state), -1, dtype=np.int32
    )
    lengths = np.zeros((n_candidates, n_probes, max_points), dtype=np.int32)
    for candidate_index, profile in enumerate(profiles):
        if len(profile["trajectories"]) != n_probes:
            raise ValueError("profile probe counts differ")
        for probe_index, trajectory in enumerate(profile["trajectories"]):
            for point_index, state in enumerate(trajectory):
                lengths[candidate_index, probe_index, point_index] = len(state)
                states[candidate_index, probe_index, point_index, : len(state)] = state
    return states, lengths


def compute_distance_matrix(
    profiles: Sequence[dict[str, Any]], *, prefix_mode: bool
) -> np.ndarray:
    if len(profiles) < 2:
        return np.zeros((len(profiles), len(profiles)), dtype=np.float32)
    states, lengths = _pack_profiles(profiles)
    return _pairwise_pstraj_distance(states, lengths, prefix_mode)

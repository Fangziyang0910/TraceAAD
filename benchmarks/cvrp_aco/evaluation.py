from __future__ import annotations

import multiprocessing
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation, set_kill_with_parent
from benchmarks.cvrp_aco.dataset import (
    CAPACITY,
    DEFAULT_SPLIT,
    load_split_instances,
)
from benchmarks.cvrp_aco.template import (
    task_description,
    template_program,
)

__all__ = ["CVRPACOEvaluation"]

# (distances, demands, prior, capacity, n_ants, n_iterations, aco_seed, instance_index)
_AcoJob = tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, int, int]


def _run_aco_job(job: _AcoJob) -> float:
    """Top-level worker for spawn ProcessPool (must be picklable)."""
    distances, demands, prior, capacity, n_ants, n_iterations, aco_seed, instance_index = job
    rng = np.random.default_rng(aco_seed + instance_index)
    return ACO(
        distances,
        demands,
        prior,
        capacity,
        n_ants=n_ants,
        rng=rng,
    ).run(n_iterations)


class ACO:
    """NumPy port of the CVRP ACO used by ReEvo, MCTS-AHD, and PathWise."""

    def __init__(
        self,
        distances: np.ndarray,
        demands: np.ndarray,
        heuristic: np.ndarray,
        capacity: int,
        *,
        n_ants: int = 30,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        rng: np.random.Generator,
    ):
        self.distances = np.asarray(distances, dtype=np.float64)
        self.demands = np.asarray(demands, dtype=np.float64)
        self.heuristic = np.asarray(heuristic, dtype=np.float64)
        self.capacity = int(capacity)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = rng
        self.problem_size = len(self.distances)
        self.pheromone = np.ones_like(self.distances)
        self.lowest_cost = float("inf")

    def run(self, n_iterations: int) -> float:
        for _ in range(int(n_iterations)):
            paths = self._generate_paths()
            costs = self._path_costs(paths)
            self.lowest_cost = min(self.lowest_cost, float(costs.min()))
            self._update_pheromone(paths, costs)
        return self.lowest_cost

    def _generate_paths(self) -> np.ndarray:
        actions = np.zeros(self.n_ants, dtype=np.int64)
        visit_mask = np.ones((self.n_ants, self.problem_size), dtype=bool)
        self._update_visit_mask(visit_mask, actions)
        used_capacity = np.zeros(self.n_ants, dtype=np.float64)
        capacity_mask = self._update_capacity_mask(actions, used_capacity)
        paths = [actions.copy()]

        while not ((~visit_mask[:, 1:]).all() and (actions == 0).all()):
            weights = (
                self.pheromone[actions] ** self.alpha
                * self.heuristic[actions] ** self.beta
                * visit_mask
                * capacity_mask
            )
            totals = weights.sum(axis=1)
            if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
                raise ValueError("ACO transition weights must contain a valid move.")
            actions = np.array(
                [self.rng.choice(self.problem_size, p=row / total)
                 for row, total in zip(weights, totals)],
                dtype=np.int64,
            )
            paths.append(actions.copy())
            self._update_visit_mask(visit_mask, actions)
            capacity_mask = self._update_capacity_mask(actions, used_capacity)

        return np.stack(paths)

    def _update_visit_mask(self, mask: np.ndarray, actions: np.ndarray) -> None:
        mask[np.arange(self.n_ants), actions] = False
        mask[:, 0] = True
        has_customers = mask[:, 1:].any(axis=1)
        mask[(actions == 0) & has_customers, 0] = False

    def _update_capacity_mask(
        self, actions: np.ndarray, used_capacity: np.ndarray
    ) -> np.ndarray:
        used_capacity[actions == 0] = 0.0
        used_capacity += self.demands[actions]
        remaining = self.capacity - used_capacity
        return self.demands[np.newaxis, :] <= remaining[:, np.newaxis]

    def _path_costs(self, paths: np.ndarray) -> np.ndarray:
        routes = paths.T
        next_nodes = np.roll(routes, shift=-1, axis=1)
        return np.sum(self.distances[routes[:, :-1], next_nodes[:, :-1]], axis=1)

    def _update_pheromone(self, paths: np.ndarray, costs: np.ndarray) -> None:
        self.pheromone *= self.decay
        for ant_index in range(self.n_ants):
            path = paths[:, ant_index]
            next_nodes = np.roll(path, shift=-1)
            self.pheromone[path[:-1], next_nodes[:-1]] += 1.0 / costs[ant_index]
        self.pheromone[self.pheromone < 1e-10] = 1e-10


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        coordinates[:, np.newaxis] - coordinates[np.newaxis, :], axis=2
    )
    distances[np.diag_indices_from(distances)] = 1.0
    return distances


class CVRPACOEvaluation(Evaluation):
    """Evaluate an edge heuristic with the published CVRP-ACO framework.

    Scores are negative mean route lengths because LLM4AD methods maximize
    fitness while CVRP minimizes distance.
    """

    def __init__(
        self,
        timeout_seconds: int | float | None = 120,
        split: str = DEFAULT_SPLIT,
        n_ants: int = 30,
        n_iterations: int = 100,
        aco_seed: int = 1234,
        n_workers: int = 1,
        **kwargs,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        if (
            not isinstance(n_ants, int)
            or isinstance(n_ants, bool)
            or not isinstance(n_iterations, int)
            or isinstance(n_iterations, bool)
            or n_ants < 1
            or n_iterations < 1
        ):
            raise ValueError("n_ants and n_iterations must be positive integers.")
        if not isinstance(n_workers, int) or isinstance(n_workers, bool) or n_workers < 1:
            raise ValueError("n_workers must be a positive integer.")
        self._datasets, self.dataset_metadata = load_split_instances(split)
        self.split = split
        self.n_instance = len(self._datasets)
        self.problem_size = int(self.dataset_metadata["problem_size"])
        self.capacity = int(self.dataset_metadata.get("capacity", CAPACITY))
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.aco_seed = int(aco_seed)
        self.n_workers = n_workers

    def _build_prior(
        self, instance: np.ndarray, heuristic: Callable
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        demands = instance[:, 0].copy()
        coordinates = instance[:, 1:].copy()
        distances = _distance_matrix(coordinates)
        prior = np.asarray(
            heuristic(
                distances.copy(), coordinates.copy(), demands.copy(), self.capacity
            ),
            dtype=np.float64,
        )
        if prior.shape != distances.shape or not np.all(np.isfinite(prior)):
            raise ValueError(
                f"heuristics must return a finite {distances.shape} matrix."
            )
        prior = np.maximum(prior + 1e-9, 1e-9)
        return distances, demands, prior

    def _solve_instance(
        self, instance: np.ndarray, heuristic: Callable, instance_index: int
    ) -> float:
        distances, demands, prior = self._build_prior(instance, heuristic)
        return _run_aco_job(
            (
                distances,
                demands,
                prior,
                self.capacity,
                self.n_ants,
                self.n_iterations,
                self.aco_seed,
                instance_index,
            )
        )

    def evaluate(self, heuristic: Callable) -> float | None:
        try:
            return self._evaluate_or_raise(heuristic)
        except Exception:
            return None

    def _evaluate_or_raise(self, heuristic: Callable) -> float:
        jobs: list[_AcoJob] = []
        for index, instance in enumerate(self._datasets):
            distances, demands, prior = self._build_prior(instance, heuristic)
            jobs.append(
                (
                    distances,
                    demands,
                    prior,
                    self.capacity,
                    self.n_ants,
                    self.n_iterations,
                    self.aco_seed,
                    index,
                )
            )
        if self.n_workers <= 1 or len(jobs) <= 1:
            costs = [_run_aco_job(job) for job in jobs]
        else:
            workers = min(self.n_workers, len(jobs))
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                processes=workers, initializer=set_kill_with_parent
            ) as pool:
                costs = pool.map(_run_aco_job, jobs)
        return -float(np.mean(costs))

    def evaluate_program(
        self, program_str: str, callable_func: Callable, **kwargs
    ) -> Any | None:
        return self._evaluate_or_raise(callable_func)

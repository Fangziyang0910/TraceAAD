from __future__ import annotations

import multiprocessing
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation, set_kill_with_parent
from benchmarks.op_aco.dataset import (
    DEFAULT_SPLIT,
    gen_distance_matrix,
    gen_prizes,
    load_split_instances,
)
from benchmarks.op_aco.template import design_notes, task_description, template_program

__all__ = ["OPACOEvaluation"]

# (prizes, distances, heuristic, maxlen, n_ants, n_iterations, aco_seed, instance_index)
_AcoJob = tuple[np.ndarray, np.ndarray, np.ndarray, float, int, int, int, int]


def _run_aco_job(job: _AcoJob) -> float:
    """Top-level worker for spawn ProcessPool (must be picklable)."""
    prizes, distances, heuristic, maxlen, n_ants, n_iterations, aco_seed, instance_index = job
    rng = np.random.default_rng(aco_seed + instance_index)
    return ACO(
        prizes,
        distances,
        maxlen,
        heuristic,
        n_ants=n_ants,
        rng=rng,
    ).run(n_iterations)


class ACO:
    """NumPy port of the OP ACO used by ReEvo, HSEvo, PathWise, and CALM."""

    def __init__(
        self,
        prizes: np.ndarray,
        distances: np.ndarray,
        max_len: float,
        heuristic: np.ndarray,
        *,
        n_ants: int = 20,
        decay: float = 0.9,
        alpha: float = 1.0,
        beta: float = 1.0,
        rng: np.random.Generator,
    ):
        self.n = len(prizes)
        self.prizes = np.asarray(prizes, dtype=np.float64).copy()
        self.distances = np.asarray(distances, dtype=np.float64).copy()
        self.heuristic = np.asarray(heuristic, dtype=np.float64).copy()
        self.max_len = float(max_len)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = rng
        self.Q = 1.0 / float(self.prizes.sum())
        self.alltime_best_obj = 0.0
        self._add_dummy_node()
        self.pheromone = np.ones_like(self.distances)

    def _add_dummy_node(self) -> None:
        """Append a dummy sink so ants can terminate when no feasible city remains."""
        n = self.n
        self.prizes = np.concatenate([self.prizes, np.array([0.0])])
        distances = np.concatenate(
            [self.distances, 1e10 * np.ones((1, n), dtype=np.float64)],
            axis=0,
        )
        self.distances = np.concatenate(
            [distances, 1e-10 + np.zeros((n + 1, 1), dtype=np.float64)],
            axis=1,
        )
        heuristic = np.concatenate(
            [self.heuristic, np.zeros((1, n), dtype=np.float64)],
            axis=0,
        )
        self.heuristic = np.concatenate(
            [heuristic, np.ones((n + 1, 1), dtype=np.float64)],
            axis=1,
        )
        self.distances[self.distances == 1e-10] = 0.0

    def run(self, n_iterations: int) -> float:
        for _ in range(int(n_iterations)):
            sols = self._gen_sol()
            objs = self._gen_sol_obj(sols)
            sols_t = sols.T
            best_obj = float(objs.max())
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
            self._update_pheromone(sols_t, objs)
        return self.alltime_best_obj

    def _update_pheromone(self, sols: np.ndarray, objs: np.ndarray) -> None:
        self.pheromone *= self.decay
        for ant_index in range(self.n_ants):
            sol = sols[ant_index]
            obj = float(objs[ant_index])
            self.pheromone[sol[:-1], np.roll(sol, shift=-1)[:-1]] += self.Q * obj

    def _gen_sol_obj(self, solutions: np.ndarray) -> np.ndarray:
        return self.prizes[solutions.T].sum(axis=1)

    def _gen_sol(self) -> np.ndarray:
        solutions = [np.zeros(self.n_ants, dtype=np.int64)]
        mask = np.ones((self.n_ants, self.n + 1), dtype=np.float64)
        travel_dis = np.zeros(self.n_ants, dtype=np.float64)
        cur_node = np.zeros(self.n_ants, dtype=np.int64)

        mask = self._update_mask(travel_dis, cur_node, mask)
        while not self._check_done(mask):
            nxt_node = self._pick_node(mask, cur_node)
            solutions.append(nxt_node.copy())
            travel_dis += self.distances[cur_node, nxt_node]
            cur_node = nxt_node
            mask = self._update_mask(travel_dis, cur_node, mask)
        return np.stack(solutions)

    def _pick_node(self, mask: np.ndarray, cur_node: np.ndarray) -> np.ndarray:
        pheromone = self.pheromone[cur_node]
        heuristic = self.heuristic[cur_node]
        weights = (pheromone ** self.alpha) * (heuristic ** self.beta) * mask
        totals = weights.sum(axis=1)
        if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
            raise ValueError("OP ACO transition weights must contain a valid move.")
        return np.array(
            [self.rng.choice(self.n + 1, p=row / total) for row, total in zip(weights, totals)],
            dtype=np.int64,
        )

    def _update_mask(
        self, travel_dis: np.ndarray, cur_node: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        mask = mask.copy()
        mask[np.arange(self.n_ants), cur_node] = 0.0

        for ant_id in range(self.n_ants):
            if cur_node[ant_id] == self.n:
                continue
            candidates = np.nonzero(mask[ant_id] > 0)[0]
            if candidates.size == 0:
                continue
            trails = (
                travel_dis[ant_id]
                + self.distances[cur_node[ant_id], candidates]
                + self.distances[candidates, 0]
            )
            fail_idx = candidates[trails > self.max_len]
            mask[ant_id, fail_idx] = 0.0

        mask[:, -1] = 0.0
        go2dummy = (mask[:, :-1] == 0).all(axis=1)
        mask[go2dummy, -1] = 1.0
        return mask

    def _check_done(self, mask: np.ndarray) -> bool:
        return bool((mask[:, :-1] == 0).all())


class OPACOEvaluation(Evaluation):
    """Evaluate an edge heuristic with the published OP-ACO framework.

    Scores are mean collected prize (higher is better).
    """

    def __init__(
        self,
        timeout_seconds: int | float | None = 60,
        split: str = DEFAULT_SPLIT,
        n_ants: int = 20,
        n_iterations: int = 50,
        aco_seed: int = 1234,
        n_workers: int = 1,
        **kwargs,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            timeout_seconds=timeout_seconds,
        )
        self.design_notes = design_notes
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
        self.max_len = float(self.dataset_metadata["max_len"])
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.aco_seed = int(aco_seed)
        self.n_workers = n_workers

    def _build_prior(
        self, coordinates: np.ndarray, heuristic: Callable
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prizes = gen_prizes(coordinates)
        distances = gen_distance_matrix(coordinates)
        prior = np.asarray(
            heuristic(prizes.copy(), distances.copy(), float(self.max_len)),
            dtype=np.float64,
        )
        if prior.shape != distances.shape or not np.all(np.isfinite(prior)):
            raise ValueError(
                f"heuristics must return a finite {distances.shape} matrix."
            )
        prior = np.maximum(prior + 1e-9, 1e-9)
        return prizes, distances, prior

    def _solve_instance(
        self, coordinates: np.ndarray, heuristic: Callable, instance_index: int
    ) -> float:
        prizes, distances, prior = self._build_prior(coordinates, heuristic)
        return _run_aco_job(
            (
                prizes,
                distances,
                prior,
                self.max_len,
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
        for index, coordinates in enumerate(self._datasets):
            prizes, distances, prior = self._build_prior(coordinates, heuristic)
            jobs.append(
                (
                    prizes,
                    distances,
                    prior,
                    self.max_len,
                    self.n_ants,
                    self.n_iterations,
                    self.aco_seed,
                    index,
                )
            )
        if self.n_workers <= 1 or len(jobs) <= 1:
            objs = [_run_aco_job(job) for job in jobs]
        else:
            workers = min(self.n_workers, len(jobs))
            context = multiprocessing.get_context("spawn")
            with context.Pool(
                processes=workers, initializer=set_kill_with_parent
            ) as pool:
                objs = pool.map(_run_aco_job, jobs)
        return float(np.mean(objs))

    def evaluate_program(
        self, program_str: str, callable_func: Callable, **kwargs
    ) -> Any | None:
        return self._evaluate_or_raise(callable_func)

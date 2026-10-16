# Module Name: TSPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the constructive heuristic for Traveling Salseman Problem (TSP).
#              Given a set of locations,
#              the goal is to find optimal route to travel all locations and back to start point
#              while minimizing the total travel distance.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - problem_size: Number of customers to serve: int (default: 50).
#
#
# References:
#   - Fei Liu, Xialiang Tong, Mingxuan Yuan, and Qingfu Zhang.
#     "Algorithm Evolution using Large Language Model." arXiv preprint arXiv:2311.15249 (2023).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any
import numpy as np
from core import Evaluation
from benchmarks.tsp_construct.get_instance import GetData
from benchmarks.tsp_construct.template import template_program, task_description

__all__ = ['TSPEvaluation']


class TSPEvaluation(Evaluation):
    """Evaluator for traveling salesman problem."""

    def __init__(self,
                 timeout_seconds=30,
                 n_instance=16,
                 problem_size=50,
                 seed=2024,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.problem_size = problem_size
        self.seed = seed
        getData = GetData(self.n_instance, self.problem_size, self.seed)
        self._datasets = getData.generate_instances()
        self._neighborhood_cache = {
            id(instance): self.generate_neighborhood_matrix(instance)
            for instance, _ in self._datasets
        }
        self.design_notes = (
            'The heuristic is called once per tour-construction step on each instance; '
            'the final remaining node is appended automatically without calling it. '
            '`unvisited_nodes` lists the IDs of nodes not yet visited, ordered by '
            'increasing distance from the current node, and the returned node must be '
            'one of them; `distance_matrix` is the full-instance matrix and the returned '
            'value is a full-graph node ID. Re-selecting an already visited node aborts '
            'the evaluation.'
        )

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def tour_cost(self, instance, solution, problem_size):
        cost = 0
        for j in range(problem_size - 1):
            cost += np.linalg.norm(instance[int(solution[j])] - instance[int(solution[j + 1])])
        cost += np.linalg.norm(instance[int(solution[-1])] - instance[int(solution[0])])
        return cost

    def generate_neighborhood_matrix(self, instance):
        coordinates = np.asarray(instance)
        distances = np.linalg.norm(coordinates[:, None] - coordinates, axis=2)
        return np.argsort(distances, axis=1)

    def evaluate(self, eva: callable) -> float:
        costs = []
        for instance, distance_matrix in self._datasets[:self.n_instance]:
            neighbors = self._neighborhood_cache.get(id(instance))
            if neighbors is None:
                neighbors = self.generate_neighborhood_matrix(instance)
                self._neighborhood_cache[id(instance)] = neighbors
            route = np.zeros(self.problem_size, dtype=int)
            visited = np.zeros(self.problem_size, dtype=bool)
            visited[0] = True
            current = 0
            for index in range(1, self.problem_size - 1):
                order = neighbors[current]
                unvisited = order[~visited[order]]
                next_node = eva(current, 0, unvisited, distance_matrix)
                if (
                    not isinstance(next_node, (int, np.integer))
                    or isinstance(next_node, (bool, np.bool_))
                    or not 0 <= next_node < self.problem_size
                    or visited[next_node]
                ):
                    return None
                route[index] = current = next_node
                visited[next_node] = True
            route[-1] = np.flatnonzero(~visited)[0]
            costs.append(self.tour_cost(instance, route, self.problem_size))
        return float(np.mean(costs))

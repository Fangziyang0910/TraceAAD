# Module Name: OBPEvaluation
# Last Revision: 2025/2/16
# Description: Evaluates the Online Bin Packing Problem (OBP).
#              Given a sequence of items arriving one by one, the goal is to pack them into bins
#              of fixed capacity in real-time, minimizing the number of bins used.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instances: Number of problem instances to generate: int (default: 5).
#    - n_items: Number of items to pack: int (default: 5000).
#    - capacity: Maximum capacity of each bin: int (default: 100).
#
# References:
#   - Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
#       Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
#       with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
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
from benchmarks.online_bin_packing.template import template_program, task_description
from benchmarks.online_bin_packing.generate_weibull_instances import (
    generate_weibull_dataset,
    generate_weibull_multiscale_dataset,
)

__all__ = ['OBPEvaluation']


class OBPEvaluation(Evaluation):
    """Evaluator for online bin packing problem."""

    def __init__(self, timeout_seconds=30,
                 n_instances=5,
                 n_items=5000,
                 capacity=100,
                 seed=2024,
                 dataset_specs=None,
                 **kwargs):
        """
        Args:
            n_instances: Number of generated Weibull instances.
            n_items: Number of items per generated instance.
            capacity: Bin capacity.
            seed: Seed used to generate reproducible instances.
            dataset_specs: Optional fixed multi-scale dataset description. Each
                entry defines ``n_instances``, ``n_items``, and ``capacities``.
        """

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            timeout_seconds=timeout_seconds
        )

        self.n_instances = n_instances
        self.n_items = n_items
        self.capacity = capacity
        self.seed = seed

        if dataset_specs is None:
            self._datasets = generate_weibull_dataset(
                self.n_instances,
                self.n_items,
                self.capacity,
                self.seed,
            )
        else:
            self._datasets = generate_weibull_multiscale_dataset(
                dataset_specs,
                self.seed,
            )

    @property
    def dataset_metadata(self) -> tuple[tuple[str, int, int], ...]:
        """Return the fixed instance names, sizes, and capacities."""
        return tuple(
            (
                name,
                int(instance["num_items"]),
                int(instance["capacity"]),
            )
            for name, instance in self._datasets.items()
        )

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        # Execute the program afresh for every instance: module-level state
        # (e.g. running item statistics) must not carry over between
        # instances, or the score would depend on instance order.
        name = callable_func.__name__

        def fresh() -> callable:
            namespace = {}
            exec(program_str, namespace)
            return namespace[name]

        return self.evaluate(callable_func, fresh=fresh)

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        """Returns indices of bins in which item can fit."""
        return np.nonzero((bins - item) >= 0)[0]

    def online_binpack(self,
                       items: tuple[float, ...], bins: np.ndarray, priority: callable
                       ) -> np.ndarray:
        """Pack items and return the remaining capacity of each bin."""
        # Add items to bins.
        for item in items:
            # Extract bins that have sufficient space to fit item.
            valid_bin_indices = self.get_valid_bin_indices(item, bins)
            # Score each bin based on heuristic.
            priorities = priority(item, bins[valid_bin_indices])
            # Add item to bin with highest priority.
            best_bin = valid_bin_indices[np.argmax(priorities)]
            bins[best_bin] -= item
        return bins

    def evaluate(self, priority: callable, fresh: callable | None = None) -> float:
        """Evaluate heuristic function on a set of online binpacking instances.

        ``fresh`` returns a newly executed copy of the heuristic per instance.
        """
        # List storing number of bins used for each instance.
        num_bins = []
        # Perform online binpacking for each instance.
        for name in self._datasets:
            instance = self._datasets[name]
            capacity = instance['capacity']
            items = instance['items']
            # Create num_items bins so there will always be space for all items,
            # regardless of packing order. Array has shape (num_items,).
            bins = np.full(instance['num_items'], capacity)
            # Pack items into bins and return remaining capacity in bins_packed, which
            # has shape (num_items,).
            bins_packed = self.online_binpack(items, bins, fresh() if fresh else priority)

            # If remaining capacity in a bin is equal to initial capacity, then it is
            # unused. Count number of used bins.
            num_bins.append((bins_packed != capacity).sum())
        # Average number of bins used across instances (lower is better).
        return float(np.mean(num_bins))

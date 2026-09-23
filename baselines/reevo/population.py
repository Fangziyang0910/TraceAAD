from __future__ import annotations

import copy
import math
from threading import Lock
from typing import List

import numpy as np

from core import Function


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | "Population" | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = list(pop)
        else:
            self._population = list(pop._population)

        self._pop_size = pop_size
        self._lock = Lock()
        self._generation = generation

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def elite_function(self):
        return copy.deepcopy(max(self.valid_functions(), key=lambda f: f.score))

    @property
    def generation(self):
        return self._generation

    @staticmethod
    def is_valid_score(score) -> bool:
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    def valid_functions(self) -> List[Function]:
        return [func for func in self._population if self.is_valid_score(func.score)]

    def set_population(self, funcs: List[Function], *, increment_generation=True):
        with self._lock:
            self._population = [func for func in funcs if self.is_valid_score(func.score)]
            if increment_generation:
                self._generation += 1

    def extend(self, funcs: List[Function]):
        with self._lock:
            self._population.extend(func for func in funcs if self.is_valid_score(func.score))

    def register_function(self, func: Function, *, increment_generation=False) -> bool:
        if not self.is_valid_score(func.score):
            return False
        with self._lock:
            self._population.append(func)
            if increment_generation:
                self._generation += 1
        return True

    def advance_generation(self):
        with self._lock:
            self._generation += 1

    def selection(self) -> Function:
        funcs = self.valid_functions()
        if not funcs:
            raise ValueError("Cannot select from an empty ReEvo population.")
        return np.random.choice(funcs)

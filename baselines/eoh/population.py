from __future__ import annotations

import math
import random
from threading import Lock
from typing import List

from core import Function


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
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
    def generation(self):
        return self._generation

    @staticmethod
    def _is_valid_score(score) -> bool:
        if score is None:
            return False
        try:
            return math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    def survival(self, candidates: List[Function] | None = None, *, increment_generation=True):
        pop = self._population + self._next_gen_pop
        if candidates:
            pop += candidates

        valid_pop = [func for func in pop if self._is_valid_score(func.score)]
        unique = []
        seen_score = set()
        for func in sorted(valid_pop, key=lambda f: f.score):
            score_key = float(func.score)
            if score_key in seen_score:
                continue
            seen_score.add(score_key)
            unique.append(func)

        self._population = unique[:self._pop_size]
        self._next_gen_pop = []
        if increment_generation:
            self._generation += 1

    def advance_generation(self):
        self._generation += 1

    def register_function(self, func: Function, *, survive=True, increment_generation=True) -> bool:
        if not self._is_valid_score(func.score):
            return False
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                return False
            if survive:
                self._population.append(func)
                self.survival(increment_generation=increment_generation)
            else:
                self._next_gen_pop.append(func)
            return True
        except Exception:
            return False
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if func.score == f.score:
                return True
        return False

    def selection(self) -> Function:
        funcs = [f for f in self._population if self._is_valid_score(f.score)]
        if not funcs:
            raise ValueError("Cannot select from an empty EoH population.")
        func = sorted(funcs, key=lambda f: f.score)
        p = [1 / (r + 1 + len(func)) for r in range(len(func))]
        return random.choices(func, weights=p, k=1)[0]

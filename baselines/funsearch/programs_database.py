# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The FunSearch programs database: islands of score-signature clusters.

Ported from google-deepmind/funsearch ``implementation/programs_database.py``.
Scores follow the original convention, higher is better; the method passes
the negated minimized task fitness. Each task here returns one scalar, so a
signature is the one-element tuple of that score.
"""

from __future__ import annotations

import copy
import dataclasses
import time
from collections.abc import Callable, Sequence

import numpy as np
import scipy.special

from core import Function, Program, TextFunctionProgramConverter

from . import code_manipulation
from .config import ProgramsDatabaseConfig

Signature = tuple[float, ...]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result


@dataclasses.dataclass(frozen=True)
class Prompt:
    """A prompt produced by the ProgramsDatabase.

    code: the prompt, ending with the header of the function to be completed.
    version_generated: the function to be completed is `_v{version_generated}`.
    island_id: island that produced the implementations in the prompt; the
        generated program is registered back into it.
    """
    code: str
    version_generated: int
    island_id: int


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(self, config: ProgramsDatabaseConfig, template: Program, function_to_evolve: str,
                 *, clock: Callable[[], float] = time.time) -> None:
        self._config = config
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._clock = clock
        self._islands = [self._new_island() for _ in range(config.num_islands)]
        self._best_score_per_island: list[float] = [-float('inf')] * config.num_islands
        self._best_program_per_island: list[Function | None] = [None] * config.num_islands
        self._last_reset_time = clock()
        self.num_resets = 0

    def _new_island(self) -> Island:
        return Island(self._template, self._function_to_evolve, self._config.functions_per_prompt,
                      self._config.cluster_sampling_temperature_init,
                      self._config.cluster_sampling_temperature_period)

    @property
    def islands(self) -> list[Island]:
        return self._islands

    @property
    def best_score_per_island(self) -> list[float]:
        return list(self._best_score_per_island)

    def get_prompt(self) -> Prompt:
        """Returns a prompt containing implementations from one chosen island."""
        island_id = np.random.randint(len(self._islands))
        code, version_generated = self._islands[island_id].get_prompt()
        return Prompt(code, version_generated, island_id)

    def _register_program_in_island(self, program: Function, island_id: int, score: float) -> None:
        self._islands[island_id].register_program(program, score)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_score_per_island[island_id] = score

    def register_program(self, program: Function, island_id: int | None, score: float) -> None:
        """Registers `program` in the database; `island_id=None` seeds every island."""
        if island_id is None:
            for island in range(len(self._islands)):
                self._register_program_in_island(program, island, score)
        else:
            self._register_program_in_island(program, island_id, score)
        if self._clock() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = self._clock()
            self.reset_islands()

    def reset_islands(self) -> None:
        """Resets the weaker half of islands, each seeded with the best program of a kept island."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score = np.argsort(
            np.array(self._best_score_per_island) + np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = self._new_island()
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_score = self._best_score_per_island[founder_island_id]
            self._register_program_in_island(founder, island_id, founder_score)
        self.num_resets += 1


class Island:
    """A sub-population of the programs database."""

    def __init__(self, template: Program, function_to_evolve: str, functions_per_prompt: int,
                 cluster_sampling_temperature_init: float, cluster_sampling_temperature_period: int) -> None:
        self._template = template
        self._function_to_evolve = function_to_evolve
        self._functions_per_prompt = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = cluster_sampling_temperature_period
        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs = 0

    @property
    def clusters(self) -> dict[Signature, Cluster]:
        return self._clusters

    @property
    def num_programs(self) -> int:
        return self._num_programs

    def register_program(self, program: Function, score: float) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = (score,)
        if signature not in self._clusters:
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing functions from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array([self._clusters[signature].score for signature in signatures])

        # Convert scores to probabilities using softmax with temperature schedule.
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(len(signatures), size=functions_per_prompt, p=probabilities)
        chosen_signatures = [signatures[i] for i in idx]
        implementations, scores = [], []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        return self._generate_prompt(sorted_implementations), len(sorted_implementations)

    def _generate_prompt(self, implementations: Sequence[Function]) -> str:
        """Creates a prompt containing a sequence of function `implementations`."""
        implementations = copy.deepcopy(implementations)  # We will mutate these.

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[Function] = []
        for i, implementation in enumerate(implementations):
            new_function_name = f'{self._function_to_evolve}_v{i}'
            implementation.name = new_function_name
            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = f'Improved version of `{self._function_to_evolve}_v{i - 1}`.'
            # If the function is recursive, replace calls to itself with its new name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation), self._function_to_evolve, new_function_name)
            versioned_functions.append(TextFunctionProgramConverter.text_to_function(implementation))

        # Create the header of the function to be generated by the LLM.
        next_version = len(implementations)
        header = dataclasses.replace(
            implementations[-1],
            name=f'{self._function_to_evolve}_v{next_version}',
            body='',
            docstring=f'Improved version of `{self._function_to_evolve}_v{next_version - 1}`.',
        )
        versioned_functions.append(header)

        # Replace functions in the template with the list constructed here.
        return str(dataclasses.replace(self._template, functions=versioned_functions))


class Cluster:
    """A cluster of programs on the same island and with the same Signature."""

    def __init__(self, score: float, implementation: Function):
        self._score = score
        self._programs: list[Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    @property
    def programs(self) -> list[Function]:
        return self._programs

    def register_program(self, program: Function) -> None:
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> Function:
        """Samples a program, giving higher probability to shorter programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return self._programs[np.random.choice(len(self._programs), p=probabilities)]

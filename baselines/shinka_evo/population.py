from __future__ import annotations

import copy
import math
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core import Function


@dataclass
class ShinkaProgram:
    id: str
    function: Function
    program: str
    parent_id: str | None = None
    archive_inspiration_ids: list[str] = field(default_factory=list)
    top_k_inspiration_ids: list[str] = field(default_factory=list)
    island_idx: int | None = None
    generation: int = 0
    patch_type: str | None = None
    code_diff: str | None = None
    combined_score: float = 0.0
    public_metrics: dict[str, Any] = field(default_factory=dict)
    private_metrics: dict[str, Any] = field(default_factory=dict)
    text_feedback: str | list[str] = ""
    correct: bool = False
    children_count: int = 0
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    in_archive: bool = False
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
            cls,
            function: Function,
            program: str,
            *,
            parent_id: str | None = None,
            archive_inspiration_ids: Sequence[str] = (),
            top_k_inspiration_ids: Sequence[str] = (),
            island_idx: int | None = None,
            generation: int = 0,
            patch_type: str | None = None,
            code_diff: str | None = None,
            combined_score: float = 0.0,
            public_metrics: dict[str, Any] | None = None,
            private_metrics: dict[str, Any] | None = None,
            text_feedback: str | list[str] = "",
            correct: bool = False,
            embedding: Sequence[float] | None = None,
            metadata: dict[str, Any] | None = None,
            program_id: str | None = None,
    ) -> "ShinkaProgram":
        func = copy.deepcopy(function)
        func.score = combined_score if correct else None
        return cls(
            id=program_id or str(uuid.uuid4()),
            function=func,
            program=program,
            parent_id=parent_id,
            archive_inspiration_ids=list(archive_inspiration_ids),
            top_k_inspiration_ids=list(top_k_inspiration_ids),
            island_idx=island_idx,
            generation=generation,
            patch_type=patch_type,
            code_diff=code_diff,
            combined_score=float(combined_score or 0.0),
            public_metrics=dict(public_metrics or {}),
            private_metrics=dict(private_metrics or {}),
            text_feedback=text_feedback,
            correct=bool(correct),
            embedding=list(embedding or []),
            metadata=dict(metadata or {}),
        )

    def copy_for_island(self, island_idx: int) -> "ShinkaProgram":
        clone = copy.deepcopy(self)
        clone.id = str(uuid.uuid4())
        clone.island_idx = island_idx
        clone.parent_id = None
        clone.children_count = 0
        clone.in_archive = False
        clone.metadata = dict(clone.metadata)
        clone.metadata["_is_island_copy"] = True
        clone.metadata["_original_program_id"] = self.id
        clone.function = copy.deepcopy(self.function)
        return clone


def is_valid_score(score: Any) -> bool:
    if score is None:
        return False
    try:
        return math.isfinite(float(score))
    except (TypeError, ValueError):
        return False


def stable_sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    z = math.exp(x)
    return z / (1 + z)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class ShinkaArchive:
    def __init__(
            self,
            *,
            num_islands: int = 2,
            archive_size: int = 40,
            elite_selection_ratio: float = 0.3,
            num_archive_inspirations: int = 1,
            num_top_k_inspirations: int = 1,
            parent_selection_strategy: str = "weighted",
            exploitation_alpha: float = 1.0,
            parent_selection_lambda: float = 10.0,
            num_beams: int = 5,
            archive_selection_strategy: str = "fitness",
            archive_criteria: dict[str, float] | None = None,
            enforce_island_separation: bool = True,
            island_selection_strategy: str = "uniform",
            migration_interval: int = 10,
            migration_rate: float = 0.0,
            island_elitism: bool = True,
            rng: random.Random | None = None,
    ):
        self.num_islands = max(1, int(num_islands))
        self.archive_size = int(archive_size)
        self.elite_selection_ratio = float(elite_selection_ratio)
        self.num_archive_inspirations = int(num_archive_inspirations)
        self.num_top_k_inspirations = int(num_top_k_inspirations)
        self.parent_selection_strategy = parent_selection_strategy
        self.exploitation_alpha = float(exploitation_alpha)
        self.parent_selection_lambda = float(parent_selection_lambda)
        self.num_beams = int(num_beams)
        self.archive_selection_strategy = archive_selection_strategy
        self.archive_criteria = dict(archive_criteria or {"combined_score": 1.0})
        self.enforce_island_separation = bool(enforce_island_separation)
        self.island_selection_strategy = island_selection_strategy
        self.migration_interval = int(migration_interval)
        self.migration_rate = float(migration_rate)
        self.island_elitism = bool(island_elitism)
        self.rng = rng or random.Random()

        self.programs: dict[str, ShinkaProgram] = {}
        self.islands: dict[int, list[str]] = {i: [] for i in range(self.num_islands)}
        self.archive_ids: list[str] = []
        self.best_program_id: str | None = None
        self.initial_program_id: str | None = None
        self.beam_parent_id: str | None = None

    def __len__(self) -> int:
        return len(self.programs)

    @property
    def best_program(self) -> ShinkaProgram | None:
        if self.best_program_id is None:
            return None
        return self.programs.get(self.best_program_id)

    @property
    def initial_program(self) -> ShinkaProgram | None:
        if self.initial_program_id is None:
            return None
        return self.programs.get(self.initial_program_id)

    def all_programs(self) -> list[ShinkaProgram]:
        return list(self.programs.values())

    def correct_programs(self, island_idx: int | None = None) -> list[ShinkaProgram]:
        programs = self._programs_for_island(island_idx)
        return [program for program in programs if program.correct]

    def incorrect_programs(self, island_idx: int | None = None) -> list[ShinkaProgram]:
        programs = self._programs_for_island(island_idx)
        return [program for program in programs if not program.correct]

    def archived_programs(self, island_idx: int | None = None) -> list[ShinkaProgram]:
        programs = [self.programs[pid] for pid in self.archive_ids if pid in self.programs]
        if island_idx is not None:
            programs = [program for program in programs if program.island_idx == island_idx]
        return programs

    def add_program(self, program: ShinkaProgram, *, update_archive: bool = True) -> None:
        self.programs[program.id] = program
        if program.island_idx is not None:
            self.islands.setdefault(program.island_idx, []).append(program.id)
        if program.parent_id and program.parent_id in self.programs:
            self.programs[program.parent_id].children_count += 1
        if self.initial_program_id is None and program.generation == 0 and not program.metadata.get("_is_island_copy"):
            self.initial_program_id = program.id
        self._update_best(program)
        if update_archive:
            self.update_archive(program)

    def seed_islands(self, seed: ShinkaProgram) -> list[ShinkaProgram]:
        self.add_program(seed)
        copies = []
        for island_idx in range(1, self.num_islands):
            clone = seed.copy_for_island(island_idx)
            self.add_program(clone)
            copies.append(clone)
        return copies

    def _programs_for_island(self, island_idx: int | None) -> list[ShinkaProgram]:
        if island_idx is None:
            return self.all_programs()
        return [self.programs[pid] for pid in self.islands.get(island_idx, []) if pid in self.programs]

    def _update_best(self, program: ShinkaProgram) -> None:
        if not program.correct:
            return
        current = self.best_program
        if current is None or self._is_better(program, current):
            self.best_program_id = program.id

    def _criterion_value(self, program: ShinkaProgram, criterion: str) -> float:
        if criterion == "combined_score":
            return float(program.combined_score or 0.0)
        if criterion == "complexity":
            return float(program.metadata.get("complexity", 0.0))
        value = program.public_metrics.get(criterion, program.private_metrics.get(criterion, program.metadata.get(criterion, 0.0)))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _archive_rank_score(self, program: ShinkaProgram, context: list[ShinkaProgram]) -> float:
        all_programs = context + [program]
        total = 0.0
        for criterion, weight in self.archive_criteria.items():
            reverse = weight >= 0
            ranked = sorted(all_programs, key=lambda p: self._criterion_value(p, criterion), reverse=reverse)
            rank = ranked.index(program)
            normalized = 1.0 - rank / max(len(ranked) - 1, 1)
            total += abs(float(weight)) * normalized
        return total

    def _is_better(
            self,
            left: ShinkaProgram,
            right: ShinkaProgram,
            archive_context: list[ShinkaProgram] | None = None,
    ) -> bool:
        if archive_context is not None and len(self.archive_criteria) > 1:
            return self._archive_rank_score(left, archive_context) > self._archive_rank_score(right, archive_context)
        return float(left.combined_score or 0.0) > float(right.combined_score or 0.0)

    def update_archive(self, program: ShinkaProgram) -> str | None:
        if self.archive_size <= 0 or not program.correct:
            return None
        if program.id in self.archive_ids:
            program.in_archive = True
            return None
        if len(self.archive_ids) < self.archive_size:
            self.archive_ids.append(program.id)
            program.in_archive = True
            return None
        if self.archive_selection_strategy == "crowding":
            replaced = self._update_archive_crowding(program)
        else:
            replaced = self._update_archive_fitness(program)
        return replaced

    def _update_archive_fitness(self, program: ShinkaProgram) -> str | None:
        archive_programs = self.archived_programs()
        if not archive_programs:
            self.archive_ids.append(program.id)
            program.in_archive = True
            return None
        if len(self.archive_criteria) > 1:
            worst = min(archive_programs, key=lambda p: self._archive_rank_score(p, archive_programs))
        else:
            worst = min(archive_programs, key=lambda p: p.combined_score)
        if self._is_better(program, worst, archive_programs):
            self.archive_ids.remove(worst.id)
            worst.in_archive = False
            self.archive_ids.append(program.id)
            program.in_archive = True
            return worst.id
        return None

    def _update_archive_crowding(self, program: ShinkaProgram) -> str | None:
        if not program.embedding:
            return self._update_archive_fitness(program)
        archive_programs = [prog for prog in self.archived_programs() if prog.embedding]
        if not archive_programs:
            return self._update_archive_fitness(program)
        most_similar = max(archive_programs, key=lambda p: cosine_similarity(program.embedding, p.embedding))
        if self._is_better(program, most_similar, self.archived_programs()):
            self.archive_ids.remove(most_similar.id)
            most_similar.in_archive = False
            self.archive_ids.append(program.id)
            program.in_archive = True
            return most_similar.id
        return None

    def select_island(self) -> int | None:
        initialized = [idx for idx, ids in self.islands.items() if ids]
        if not initialized:
            return None
        if self.island_selection_strategy == "proportional":
            weights = [len(self.islands[idx]) for idx in initialized]
            return self.rng.choices(initialized, weights=weights, k=1)[0]
        if self.island_selection_strategy == "weighted":
            weights = []
            for idx in initialized:
                best = self._best_in(self.correct_programs(idx))
                weights.append(max(best.combined_score if best else 0.0, 0.0) + 1e-9)
            return self.rng.choices(initialized, weights=weights, k=1)[0]
        return self.rng.choice(initialized)

    def sample_parent_with_fix_mode(self, island_idx: int | None = None) -> tuple[ShinkaProgram, bool]:
        if self.correct_programs(island_idx):
            return self.sample_parent(island_idx), False
        incorrect = self.incorrect_programs(island_idx)
        if not incorrect:
            raise ValueError("Archive is empty; no program is available for parent or fix sampling.")
        return self.rng.choice(incorrect), True

    def sample_parent(self, island_idx: int | None = None) -> ShinkaProgram:
        if self.parent_selection_strategy == "weighted":
            return self._sample_parent_weighted(island_idx)
        if self.parent_selection_strategy == "power_law":
            return self._sample_parent_power_law(island_idx)
        if self.parent_selection_strategy == "beam_search":
            return self._sample_parent_beam_search(island_idx)
        raise ValueError(f"Unknown parent selection strategy: {self.parent_selection_strategy}")

    def _sample_parent_power_law(self, island_idx: int | None) -> ShinkaProgram:
        candidates = self.archived_programs(island_idx) or self.correct_programs(island_idx)
        if not candidates and island_idx is None:
            candidates = self.correct_programs(None)
        if not candidates:
            raise ValueError("No correct programs available for power-law parent sampling.")
        ranked = sorted(candidates, key=lambda p: p.combined_score, reverse=True)
        alpha = self.exploitation_alpha
        if alpha == 0:
            return self.rng.choice(ranked)
        weights = [((rank + 1) ** (-alpha)) for rank in range(len(ranked))]
        return self.rng.choices(ranked, weights=weights, k=1)[0]

    def weighted_parent_probabilities(self, island_idx: int | None = None) -> list[tuple[ShinkaProgram, float]]:
        candidates = [program for program in self.archived_programs(island_idx) if program.correct]
        if not candidates:
            return []
        scores = [float(program.combined_score or 0.0) for program in candidates]
        median = float(np.median(scores))
        mad = float(np.median([abs(score - median) for score in scores]))
        scale_factor = max(mad, 1e-6)
        weights = []
        for program in candidates:
            normalized_diff = (float(program.combined_score or 0.0) - median) / scale_factor
            performance = stable_sigmoid(self.parent_selection_lambda * normalized_diff)
            novelty = 1.0 / (1.0 + program.children_count)
            weights.append(performance * novelty)
        total = sum(weights)
        if total <= 0:
            probability = 1.0 / len(candidates)
            return [(program, probability) for program in candidates]
        return [(program, weight / total) for program, weight in zip(candidates, weights)]

    def _sample_parent_weighted(self, island_idx: int | None) -> ShinkaProgram:
        probabilities = self.weighted_parent_probabilities(island_idx)
        if not probabilities:
            best = self.best_program
            if best and best.correct and (island_idx is None or best.island_idx == island_idx):
                return best
            candidates = self.correct_programs(island_idx)
            if not candidates and island_idx is None:
                candidates = self.correct_programs(None)
            if candidates:
                return self.rng.choice(candidates)
            raise ValueError("No correct programs available for weighted parent sampling.")
        programs, weights = zip(*probabilities)
        return self.rng.choices(list(programs), weights=list(weights), k=1)[0]

    def _sample_parent_beam_search(self, island_idx: int | None) -> ShinkaProgram:
        current = self.programs.get(self.beam_parent_id) if self.beam_parent_id else None
        if current and current.correct and current.children_count < self.num_beams:
            return current
        candidates = self.correct_programs(island_idx) or self.correct_programs(None)
        if not candidates:
            raise ValueError("No correct programs available for beam-search parent sampling.")
        best = max(candidates, key=lambda p: p.combined_score)
        self.beam_parent_id = best.id
        return best

    def sample_inspirations(self, parent: ShinkaProgram) -> tuple[list[ShinkaProgram], list[ShinkaProgram]]:
        archive_inspirations = self._sample_archive_inspirations(parent, self.num_archive_inspirations)
        top_k = self._sample_top_k_inspirations(parent, archive_inspirations, self.num_top_k_inspirations)
        return archive_inspirations, top_k

    def _eligible_archive_for_parent(self, parent: ShinkaProgram) -> list[ShinkaProgram]:
        programs = self.archived_programs()
        if self.enforce_island_separation:
            programs = [program for program in programs if program.island_idx == parent.island_idx]
        return [program for program in programs if program.correct and program.id != parent.id]

    def _sample_archive_inspirations(self, parent: ShinkaProgram, n: int) -> list[ShinkaProgram]:
        if n <= 0:
            return []
        inspirations: list[ShinkaProgram] = []
        seen = {parent.id}
        best = self.best_program
        if best and best.correct and best.id not in seen:
            if not self.enforce_island_separation or best.island_idx == parent.island_idx:
                inspirations.append(best)
                seen.add(best.id)
        num_elites = max(0, int(n * self.elite_selection_ratio))
        if num_elites > 0 and len(inspirations) < n:
            elites = sorted(self._eligible_archive_for_parent(parent), key=lambda p: p.combined_score, reverse=True)
            for elite in elites:
                if len(inspirations) >= n:
                    break
                if elite.id not in seen:
                    inspirations.append(elite)
                    seen.add(elite.id)
        if len(inspirations) < n:
            candidates = [program for program in self._eligible_archive_for_parent(parent) if program.id not in seen]
            self.rng.shuffle(candidates)
            inspirations.extend(candidates[:n - len(inspirations)])
        return inspirations[:n]

    def _sample_top_k_inspirations(
            self,
            parent: ShinkaProgram,
            archive_inspirations: list[ShinkaProgram],
            k: int,
    ) -> list[ShinkaProgram]:
        if k <= 0:
            return []
        excluded = {parent.id}
        excluded.update(program.id for program in archive_inspirations)
        candidates = [program for program in self.archived_programs() if program.correct and program.id not in excluded]
        if self.enforce_island_separation:
            candidates = [program for program in candidates if program.island_idx == parent.island_idx]
        return sorted(candidates, key=lambda p: p.combined_score, reverse=True)[:k]

    def compute_similarities(self, embedding: Sequence[float], island_idx: int | None) -> list[float]:
        programs = self.correct_programs(island_idx)
        return [cosine_similarity(embedding, program.embedding) for program in programs if program.embedding]

    def most_similar_program(self, embedding: Sequence[float], island_idx: int | None) -> ShinkaProgram | None:
        programs = [program for program in self.correct_programs(island_idx) if program.embedding]
        if not programs:
            return None
        return max(programs, key=lambda program: cosine_similarity(embedding, program.embedding))

    def maybe_migrate(self, current_generation: int) -> list[dict[str, Any]]:
        if self.num_islands < 2 or self.migration_rate <= 0:
            return []
        if self.migration_interval <= 0 or current_generation % self.migration_interval != 0:
            return []
        events = []
        for source_idx in range(self.num_islands):
            source = self._programs_for_island(source_idx)
            if len(source) <= 1:
                continue
            num_migrants = max(1, int(len(source) * self.migration_rate))
            migrants = self._select_migrants(source_idx, num_migrants)
            for migrant in migrants:
                destinations = [idx for idx in range(self.num_islands) if idx != source_idx]
                destination = self.rng.choice(destinations)
                self.islands[source_idx].remove(migrant.id)
                self.islands.setdefault(destination, []).append(migrant.id)
                migrant.metadata.setdefault("migration_history", []).append({
                    "from": source_idx,
                    "to": destination,
                    "generation": current_generation,
                })
                migrant.island_idx = destination
                events.append({"program_id": migrant.id, "from": source_idx, "to": destination})
        return events

    def _select_migrants(self, source_idx: int, count: int) -> list[ShinkaProgram]:
        programs = self._programs_for_island(source_idx)
        if self.island_elitism:
            best = self._best_in([program for program in programs if program.correct])
            programs = [program for program in programs if best is None or program.id != best.id]
        if not programs:
            return []
        self.rng.shuffle(programs)
        return programs[:count]

    @staticmethod
    def _best_in(programs: list[ShinkaProgram]) -> ShinkaProgram | None:
        if not programs:
            return None
        return max(programs, key=lambda p: p.combined_score)

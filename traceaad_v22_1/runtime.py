"""Shared mechanism-1.1 lifecycle, independent of the generation protocol."""

import json
import math
from dataclasses import asdict
from pathlib import Path

from traceaad_base import Config
from traceaad_base.selection import identity

from .selection import Selector


class TrajectoryAllocation:
    """Cooperative search hooks for quality-first trajectory allocation and replay."""

    @property
    def _version(self) -> str:
        return self.method.removeprefix("traceaad_").upper().replace("_", ".")

    def __init__(self, task, model, *, config=None, **options):
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError(f"{self._version} needs at least one initial program")
        self.selector = Selector()
        if options.get("resume"):
            self._validate_resume(task, config, options.get("run_dir"))
        super().__init__(task, model, config=config, **options)
        if not options.get("resume") and self.store.path is not None:
            saved = self.store.read("run_config.json")
            self.store.write("run_config.json", {**saved, "selection_settings": self.selector.settings})

    def _validate_resume(self, task, config, directory):
        """Reject changed settings before Base repairs or replays any logs."""
        if directory is None or not (Path(directory) / "run_config.json").is_file():
            raise ValueError(f"{self._version} resume requires its original run_config.json")
        saved = json.loads((Path(directory) / "run_config.json").read_text())
        expected = {
            "method": self.method,
            "config": asdict(config),
            "selection_settings": self.selector.settings,
            "task": json.loads(json.dumps(task.description())),
        }
        differences = []
        for name, value in expected.items():
            if saved.get(name) != value:
                if isinstance(value, dict) and isinstance(saved.get(name), dict):
                    differences.extend(
                        f"{name}.{key}" for key in sorted(value.keys() | saved[name].keys())
                        if key not in value or key not in saved[name] or saved[name][key] != value[key]
                    )
                else:
                    differences.append(name)
        if differences:
            raise ValueError(f"Incompatible {self._version} resume settings: " + ", ".join(differences))

    def select(self, attempt_id: int):
        action, record = self.selector.choose(
            self.config.evaluation_budget - self.evaluations_used,
            self.rng,
            self.config,
        )
        return (
            self.archive[action.parent_id],
            action.operator,
            self.archive.get(action.reference_id),
            {"selection": record},
        )

    def _evaluate(self, attempt, candidate):
        # Check at the actual call site, including after a repair request.
        if self.evaluations_used >= self.config.evaluation_budget:
            raise RuntimeError("Evaluation budget exhausted")
        is_first_evaluation = len(attempt.evaluations) == 0
        super()._evaluate(attempt, candidate)
        if is_first_evaluation and attempt.operator != "Init":
            program = self.archive[attempt.program_id]
            ev = program.evaluation
            valid = (program.valid and ev is not None and ev.fitness is not None
                     and math.isfinite(ev.fitness))
            source = identity(program.code) if valid else None
            usable = bool(valid and source not in self.selector.sources)
            selection_record = attempt.record.get("selection")
            if selection_record is not None:
                selection_record["first_result"] = {
                    "status": ev.status if ev is not None else None,
                    "fitness": float(ev.fitness) if valid else None,
                    "source_identity": source,
                    "usable_at_attempt_start": usable,
                }

    def _observe(self, attempt):
        if attempt.id in self.selector.processed:
            return
        record = attempt.record.get("selection")
        if attempt.operator != "Init" and record is None:
            raise ValueError(f"A {self._version} evolution attempt needs its recorded selection")
        program = self.archive.get(attempt.program_id)
        first_result = record.get("first_result") if record is not None else None
        final_result = record.get("final_result") if record is not None else None
        if record is not None and first_result is None and len(attempt.evaluations) == 0:
            first_result = {
                "status": None,
                "fitness": None,
                "source_identity": None,
                "usable_at_attempt_start": False,
            }
        self.selector.observe(
            attempt.id,
            record,
            program,
            evaluations=len(attempt.evaluations),
            model_calls=len(attempt.calls),
            first_result=first_result,
            final_result=final_result,
        )
        if record is not None:
            record["evaluation_calls"] = len(attempt.evaluations)
        if program is not None and program.id in self.selector.nodes:
            attempt.record["node"] = asdict(self.selector.nodes[program.id])

    finish = _observe

    def restore(self, attempt):
        if attempt.operator != "Init":
            record = attempt.record.get("selection")
            if record is None or "final_result" not in record or "first_result" not in record:
                raise ValueError(f"Missing {self._version} selection result record at attempt {attempt.id}")
        self._observe(attempt)

    @property
    def best(self):
        return min(
            (p for p in self.archive.values() if p.valid and p.evaluation.fitness is not None
             and math.isfinite(p.evaluation.fitness)),
            key=lambda p: (p.evaluation.fitness, p.id),
            default=None,
        )

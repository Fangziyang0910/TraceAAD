"""Shared mechanism-1 lifecycle, independent of the generation protocol."""

from dataclasses import asdict
import json
import math
from pathlib import Path

from traceaad_base import Config

from .selection import RESIDUAL_CLIP, Selector


class TrajectoryAllocation:
    """Cooperative search hooks for trajectory allocation and replay."""

    @property
    def _version(self):
        return self.method.removeprefix("traceaad_").upper()

    def __init__(self, task, model, *, config=None, residual_clip=RESIDUAL_CLIP, **options):
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError(f"{self._version} needs at least one initial program")
        self.selector = Selector(residual_clip)
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
        expected = {"method": self.method, "config": asdict(config),
                    "selection_settings": self.selector.settings,
                    "task": json.loads(json.dumps(task.description()))}
        differences = []
        for name, value in expected.items():
            if saved.get(name) != value:
                if isinstance(value, dict) and isinstance(saved.get(name), dict):
                    differences.extend(f"{name}.{key}" for key in sorted(value.keys() | saved[name].keys())
                                       if key not in value or key not in saved[name]
                                       or saved[name][key] != value[key])
                else:
                    differences.append(name)
        if differences:
            raise ValueError(f"Incompatible {self._version} resume settings: " + ", ".join(differences))

    def select(self, attempt_id):
        action, record = self.selector.choose(self.config.evaluation_budget - self.evaluations_used, self.rng)
        return (self.archive[action.parent_id], action.operator, self.archive.get(action.reference_id),
                {"selection": record})

    def _evaluate(self, attempt, candidate):
        # Check at the actual call site, including after a repair request.
        if self.evaluations_used >= self.config.evaluation_budget:
            raise RuntimeError("Evaluation budget exhausted")
        super()._evaluate(attempt, candidate)

    def _observe(self, attempt):
        if attempt.id in self.selector.processed:
            return
        record = attempt.record.get("selection")
        if attempt.operator != "Init" and record is None:
            raise ValueError(f"A {self._version} evolution attempt needs its recorded selection")
        program = self.archive.get(attempt.program_id)
        self.selector.observe(attempt.id, record, program)
        if record is not None:
            record["evaluation_calls"] = len(attempt.evaluations)
        if program is not None and program.id in self.selector.nodes:
            attempt.record["node"] = asdict(self.selector.nodes[program.id])
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        if initialized >= self.config.initial_size:
            self.selector.initialize()
        scale = {"f0": self.selector.f0, "s0": self.selector.s0}
        if "selection_scale" in attempt.record and attempt.record["selection_scale"] != scale:
            raise ValueError(f"Incompatible {self._version} restored scale at attempt {attempt.id}")
        attempt.record["selection_scale"] = scale

    finish = _observe

    def restore(self, attempt):
        if "selection_scale" not in attempt.record:
            raise ValueError(f"Missing {self._version} scale record at attempt {attempt.id}")
        self._observe(attempt)

    @property
    def best(self):
        return min((p for p in self.archive.values() if p.valid and p.evaluation.fitness is not None
                    and math.isfinite(p.evaluation.fitness)),
                   key=lambda p: (p.evaluation.fitness, p.id), default=None)

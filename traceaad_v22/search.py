"""V22 = V21 plus trajectory response prediction and joint budget allocation."""

from dataclasses import asdict
import math

from traceaad_base import Config
from traceaad_v21.search import TraceAAD as V21

from .selection import Selector


class TraceAAD(V21):
    method = "traceaad_v22"

    def __init__(self, task, model, *, config=None, **options):
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError("V22 needs at least one initial program")
        self.selector = Selector()
        super().__init__(task, model, config=config, **options)

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
        record = attempt.record.get("selection")
        if attempt.operator != "Init" and record is None:
            raise ValueError("A V22 evolution attempt needs its recorded selection")
        program = self.archive.get(attempt.program_id)
        self.selector.observe(attempt.id, record, program)
        if record is not None:
            record["evaluation_calls"] = len(attempt.evaluations)
        if program is not None and program.id in self.selector.nodes:
            attempt.record["node"] = asdict(self.selector.nodes[program.id])
        initialized = sum(p.valid and p.operator == "Init" for p in self.archive.values())
        if initialized >= self.config.initial_size:
            self.selector.initialize()

    finish = _observe
    restore = _observe

    @property
    def best(self):
        return min((p for p in self.archive.values() if p.valid and p.evaluation.fitness is not None
                    and math.isfinite(p.evaluation.fitness)),
                   key=lambda p: (p.evaluation.fitness, p.id), default=None)

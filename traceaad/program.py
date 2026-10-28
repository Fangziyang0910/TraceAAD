"""One parsed model-generated program and everything learned about it."""

from dataclasses import asdict, dataclass, field
from datetime import datetime

from .delivery import validate_source


@dataclass
class Program:
    id: int
    code: str
    idea: str
    analysis: str | None
    action: str
    generation: int
    parent_id: int | None = None
    reference_id: int | None = None
    repair_of: int | None = None
    depth: int = 0
    # What the program changed from its parent, as stated in the reply; empty when not asked.
    change: str = ""
    status: str = "pending"
    fitness: float | None = None
    failure: dict | None = None
    eval_seconds: float | None = None
    eval_wall_seconds: float | None = None
    calls: int | None = None
    function_seconds: float | None = None
    request_id: int | None = None
    call_ids: list[int] = field(default_factory=list)
    request: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds")
    )

    @property
    def valid(self):
        return self.status == "valid"

    def metadata(self):
        return {
            **{k: v for k, v in asdict(self).items() if k != "code"},
            "score": self.fitness,
            "valid": self.valid,
            "repaired": self.repair_of is not None,
        }

    def evaluate(self, evaluator, template):
        """Update this individual only after validation or evaluation has completed."""
        try:
            validate_source(self.code, template)
        except (SyntaxError, ValueError) as exc:
            self.failure = {
                "kind": "invalid_source",
                "error": str(exc),
                "line": None,
                "seconds": None,
                "calls": None,
                "function_seconds": None,
                "call_running": False,
            }
        evaluations = []
        if self.failure is None:
            result = evaluator.evaluate(self.code, self.id)
            self.fitness, self.failure = result["fitness"], result["failure"]
            self.eval_seconds = result.get("instance_seconds", result["seconds"])
            self.eval_wall_seconds = result["seconds"]
            self.calls, self.function_seconds = (result["calls"], result["function_seconds"])
            evaluations = result["evaluations"]
        self.status = self.failure["kind"] if self.failure else "valid"
        return evaluations

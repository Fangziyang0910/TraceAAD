"""Program records and the separate accounting for generation attempts."""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Literal

Operator = Literal["Init", "Refine", "Explore", "Crossover"]
Status = Literal["success", "error", "timeout"]


@dataclass(frozen=True)
class Evaluation:
    status: Status
    fitness: float | None = None
    failure: str | None = None

    def __post_init__(self):
        if self.status not in {"success", "error", "timeout"}:
            raise ValueError(f"Unknown evaluation status: {self.status}")
        if self.status == "success":
            if self.fitness is None or not math.isfinite(self.fitness) or self.failure is not None:
                raise ValueError("A successful evaluation needs a finite fitness and no failure")
        elif self.fitness is not None or not self.failure:
            raise ValueError("A failed evaluation needs feedback and no fitness")


@dataclass
class Program:
    id: int
    code: str
    design: str | None
    operator: Operator | None
    parent_id: int | None
    reference_id: int | None
    evaluation: Evaluation | None
    analysis: str | None = None

    @property
    def valid(self) -> bool:
        return self.id != 0 and self.evaluation is not None and self.evaluation.status == "success"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        data = dict(data)
        if data["evaluation"] is not None:
            data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)


class FormatError(ValueError):
    """The model reply cannot supply the required analysis/design/code object."""


@dataclass(frozen=True)
class Candidate:
    analysis: str
    design: str
    code: str

    @classmethod
    def parse(cls, response: str) -> "Candidate":
        def invalid_constant(value):
            raise ValueError(f"Invalid JSON constant: {value}")

        try:
            value = json.loads(response, parse_constant=invalid_constant)
        except (TypeError, ValueError) as exc:
            raise FormatError(str(exc)) from exc
        if not isinstance(value, dict):
            raise FormatError(f"Expected a JSON object, got {type(value).__name__}")
        for name in ("analysis", "design", "code"):
            if name not in value:
                raise FormatError(f"Missing required field: {name}")
            if not isinstance(value[name], str) or not value[name].strip():
                raise FormatError(f"Field {name!r} must be a nonempty string")
        return cls(analysis=value["analysis"], design=value["design"], code=value["code"])


@dataclass(frozen=True)
class ModelReply:
    content: str
    model: str | None = None
    finish_reason: str | None = None
    usage: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Attempt:
    id: int
    parent_id: int
    operator: Operator
    reference_id: int | None = None
    selection: dict = field(default_factory=dict)
    program_id: int | None = None
    calls: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    repair_used: bool = False
    completed: bool = False
    interruption: str | None = None
    # Raw final outcome; failures have no invented numerical fitness.
    outcome_fitness: float | None = None
    duplicate_of: int | None = None

    @property
    def model_calls(self) -> int:
        return len(self.calls)


@dataclass(frozen=True)
class SearchResult:
    status: Literal["finished", "failed"]
    stop_reason: str
    best: Program | None
    evaluations: int
    model_calls: int
    attempts: int
    valid_programs: int

    def to_dict(self) -> dict:
        return asdict(self)


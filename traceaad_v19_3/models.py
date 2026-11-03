"""Evaluated code versions, generation attempts and separate reflection replies."""

from dataclasses import asdict, dataclass, field
import json
import math


@dataclass(frozen=True)
class Evaluation:
    status: str
    fitness: float | None = None
    failure: str | None = None

    def __post_init__(self):
        if self.status == "success":
            if self.fitness is None or not math.isfinite(self.fitness) or self.failure is not None:
                raise ValueError("A successful evaluation needs finite fitness and no failure")
        elif self.status not in {"error", "timeout"} or self.fitness is not None or not self.failure:
            raise ValueError("A failed evaluation needs an error/timeout, feedback and no fitness")


@dataclass(frozen=True)
class Program:
    id: int
    code: str
    design: str | None
    operator: str | None
    parent_id: int | None
    evaluation: Evaluation | None
    analysis: str | None = None
    attempt_id: int | None = None
    edited_from: int | None = None
    phase: str = "generation"
    duplicate_of: int | None = None

    @property
    def valid(self):
        return self.id != 0 and self.evaluation is not None and self.evaluation.status == "success"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        if data["evaluation"] is not None:
            data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)


class FormatError(ValueError):
    pass


def json_object(text):
    def reject(value):
        raise ValueError(f"Invalid JSON constant: {value}")
    try:
        value = json.loads(text, parse_constant=reject)
    except (TypeError, ValueError) as exc:
        raise FormatError(str(exc)) from exc
    if not isinstance(value, dict):
        raise FormatError("Expected a JSON object")
    return value


@dataclass(frozen=True)
class Candidate:
    code: str
    analysis: str
    design: str

    @classmethod
    def parse(cls, text):
        value = json_object(text)
        if set(value) != {"code", "analysis", "design"}:
            raise FormatError("Expected exactly code, analysis and design")
        for name in value:
            if not isinstance(value[name], str) or not value[name].strip():
                raise FormatError(f"{name} must be a nonempty string")
        return cls(**value)


def parse_lessons(text, limit=3):
    data = json_object(text)
    if set(data) != {"positive", "negative"} or not all(isinstance(v, list) for v in data.values()):
        raise FormatError("Expected positive and negative arrays")
    accepted, discarded = [], []
    for polarity, entries in data.items():
        for content in entries:
            valid = (isinstance(content, dict)
                     and set(content) == {"summary", "recommendations", "applicable_when"}
                     and all(isinstance(content.get(k), str) and content[k].strip()
                             for k in ("summary", "applicable_when"))
                     and isinstance(content.get("recommendations"), list)
                     and bool(content["recommendations"])
                     and all(isinstance(s, str) and s.strip() for s in content["recommendations"]))
            item = {"polarity": polarity, "content": content}
            if valid and len(accepted) < limit:
                accepted.append(item)
            else:
                discarded.append({**item, "reason": "limit" if valid else "invalid_fields"})
    return accepted, discarded


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
    operator: str
    selection: dict = field(default_factory=dict)
    program_id: int | None = None
    version_ids: list[int] = field(default_factory=list)
    calls: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    provided_experiences: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)
    phase: str = "generate"
    repair_used: bool = False
    pending_candidate: dict | None = None
    candidate_phase: str = "generation"
    utility_applied: bool = False
    reflection_applied: bool = False
    reflection_status: str = "pending"
    reflection_context: dict = field(default_factory=dict)
    reflection_raw_count: int = 0
    reflection_accepted_count: int = 0
    experience_ids: list[int] = field(default_factory=list)
    discarded_lessons: list[dict] = field(default_factory=list)
    completed: bool = False
    interruption: str | None = None
    failure: str | None = None
    outcome_fitness: float | None = None
    duplicate_of: int | None = None

    @property
    def model_calls(self):
        return len(self.calls)


@dataclass(frozen=True)
class SearchResult:
    status: str
    stop_reason: str
    best: Program | None
    evaluations: int
    model_calls: int
    attempts: int
    valid_programs: int

    def to_dict(self):
        return asdict(self)

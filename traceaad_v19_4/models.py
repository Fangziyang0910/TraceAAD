"""Evaluated code versions, generation attempts and separate reflection replies."""

from dataclasses import asdict, dataclass, field
import json
import math
import re


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


@dataclass(frozen=True)
class ResponseEvent:
    evaluation_id: int
    attempt_id: int
    selected_parent_id: int
    version_id: int
    valid: bool
    first_valid_source: bool
    improved_parent: bool
    response: int
    gain: float = 0.

    def __post_init__(self):
        for name in ("evaluation_id", "attempt_id", "selected_parent_id", "version_id"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must identify a completed development evaluation")
        for name in ("valid", "first_valid_source", "improved_parent"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if (type(self.response) is not int or self.response not in (0, 1)
                or self.response != int(self.valid and self.first_valid_source and self.improved_parent)):
            raise ValueError("Response must match the immutable evaluation outcome")
        if not self.valid and (self.first_valid_source or self.improved_parent):
            raise ValueError("An invalid version cannot be novel-valid or improved")
        if type(self.gain) not in (int, float) or (self.gain > 0) != bool(self.response):
            raise ValueError("Gain must be positive exactly for an improving response")


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


def parse_reflection(text):
    data = json_object(text)
    if set(data) != {"reflection"} or not isinstance(data["reflection"], str) or not data["reflection"].strip():
        raise FormatError("Expected one nonempty reflection string")
    return data["reflection"].strip()


def parse_selection(text, ids):
    data = json_object(text)
    if set(data) != {"assessment", "choice"} or not isinstance(data["assessment"], str):
        raise FormatError("Expected an assessment string and a choice")
    if type(data["choice"]) is not int or data["choice"] not in ids:
        raise FormatError("The choice must be the number of a candidate program")
    return data["assessment"].strip(), data["choice"]


def parse_experience(text):
    """Complete updated pools; malformed items are kept aside rather than installed."""
    data = json_object(text)
    if set(data) != {"positive", "negative"} or not all(isinstance(v, list) for v in data.values()):
        raise FormatError("Expected positive and negative arrays")
    accepted, discarded = [], []
    for polarity, entries in data.items():
        for item in entries:
            valid = (isinstance(item, dict) and set(item) == {"summary", "applicable_when"}
                     and all(isinstance(v, str) and v.strip() for v in item.values()))
            (accepted if valid else discarded).append({"polarity": polarity, "content": item})
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
    outcome_applied: bool = False
    reflection_applied: bool = False
    reflection_status: str = "pending"
    reflection_context: dict = field(default_factory=dict)
    reflection: str | None = None
    long_term_status: str = "not_due"
    long_term_context: dict = field(default_factory=dict)
    discarded_experiences: list[dict] = field(default_factory=list)
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

"""树状种群、候选解析（analysis + code + design）与搜索记录模型。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Literal

ROOT_ID = 0
Operator = Literal["Init", "Refine", "Crossover", "Explore"]
Status = Literal["success", "error", "timeout"]


def json_schema(name: str, *fields: str) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {f: {"type": "string"} for f in fields},
                "required": list(fields),
                "additionalProperties": False,
            },
        },
    }


CANDIDATE_RESPONSE_FORMAT = json_schema("evotrace_candidate", "analysis", "code", "design")

_CTRL_REPAIR = str.maketrans({"\x08": r"\b", "\x0c": r"\f", "\r": " ", "\t": " "})


def clean_text(text: str | None) -> str:
    """修复 JSON 误转义的 LaTeX 控制字符（\\b、\\f）并折叠空白。"""
    if not text:
        return ""
    repaired = text.translate(_CTRL_REPAIR)
    repaired = "".join(ch for ch in repaired if ch == "\n" or ord(ch) >= 32)
    return " ".join(repaired.split())


def fitness_text(fitness: float) -> str:
    """保留可还原为实测值的精度，让相邻分数在提示中仍可区分。"""
    return str(fitness).removesuffix(".0")


@dataclass(frozen=True)
class Config:
    evaluation_budget: int = 1000
    initial_size: int = 5
    temperature: float = 1.0
    max_tokens: int = 16384
    model_timeout: float = 600.0
    seed: int = 0
    failure_count: int = 2
    max_unevaluated: int = 10

    def __post_init__(self):
        if self.evaluation_budget < 1 or self.initial_size < 1 or self.max_unevaluated < 1:
            raise ValueError(
                "Evaluation budget, initial size and unevaluated limit must be positive"
            )


@dataclass
class TreePopulation:
    """现役树状种群：parents 存父子拓扑，node_notes 存节点经验，edge_notes 存边经验。"""

    parents: dict[int, int] = field(default_factory=dict)
    node_notes: dict[int, str] = field(default_factory=dict)
    edge_notes: dict[int, str] = field(default_factory=dict)

    def path_to(self, node_id: int) -> tuple[int, ...]:
        """从虚拟根直接子节点到 node_id 的现役谱系路径。"""
        path: list[int] = []
        cur = node_id
        seen: set[int] = set()
        while cur != ROOT_ID and cur not in seen:
            seen.add(cur)
            path.append(cur)
            if (cur := self.parents.get(cur)) is None:
                break
        path.reverse()
        return tuple(path)

    def snapshot(self) -> dict:
        return {
            "parents": self.parents.copy(),
            "node_notes": self.node_notes.copy(),
            "edge_notes": self.edge_notes.copy(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreePopulation:
        return cls(
            {int(i): v for i, v in data["parents"].items()},
            {int(i): t for i, v in data.get("node_notes", {}).items() if (t := clean_text(v))},
            {int(i): t for i, v in data.get("edge_notes", {}).items() if (t := clean_text(v))},
        )


@dataclass(frozen=True)
class Evaluation:
    status: Status
    fitness: float | None = None
    failure: str | None = None


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
    duplicate_of: int | None = None

    @property
    def valid(self) -> bool:
        return self.id != 0 and self.evaluation is not None and self.evaluation.status == "success"

    @property
    def fitness(self) -> float | None:
        return self.evaluation.fitness if self.valid else None

    @classmethod
    def from_dict(cls, data: dict) -> Program:
        data = dict(data)
        if data.get("evaluation") is not None:
            data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)


class FormatError(ValueError):
    """The model reply does not contain the analysis/code/design JSON object."""


def parse_json_object(response: str) -> dict:
    """解析 JSON 对象，兼容模型偶发包裹的 Markdown 代码块围栏。"""
    if not isinstance(response, str):
        raise TypeError(f"Expected a JSON string, got {type(response).__name__}")
    text = response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class Candidate:
    analysis: str
    code: str
    design: str

    @classmethod
    def parse(cls, response: str) -> Candidate:
        try:
            value = parse_json_object(response)
        except (TypeError, ValueError) as exc:
            raise FormatError(str(exc)) from exc
        for name in ("analysis", "code", "design"):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise FormatError(f"Field {name!r} must be a nonempty string")
        return cls(
            analysis=value["analysis"], code=value["code"], design=clean_text(value["design"])
        )


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
    program_id: int | None = None
    calls: list[dict] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    repair_used: bool = False
    finished_at: float | None = None
    record: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    status: Literal["finished", "failed"]
    stop_reason: str
    best: Program | None
    evaluations: int
    model_calls: int
    attempts: int
    valid_programs: int

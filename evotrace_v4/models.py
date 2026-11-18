"""树状种群、候选解析（design + code）与搜索记录模型。"""

from dataclasses import dataclass, field
import json
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


CANDIDATE_RESPONSE_FORMAT = json_schema("evotrace_candidate", "design", "code")

_CTRL_REPAIR = str.maketrans({"\x08": r"\b", "\x0c": r"\f", "\r": " ", "\t": " "})


def clean_text(text: str | None) -> str:
    """修复 JSON 误转义的 LaTeX 控制字符（\\b、\\f）并折叠空白。"""
    if not text:
        return ""
    repaired = text.translate(_CTRL_REPAIR)
    repaired = "".join(ch for ch in repaired if ch == "\n" or ord(ch) >= 32)
    return " ".join(repaired.split())


def outcome_text(parent_fit: float | None, child_fit: float | None) -> str:
    if child_fit is None:
        return "failed"
    if parent_fit is None:
        return f"fitness = {child_fit:.6g}"
    word = "improved" if child_fit < parent_fit else "tied" if child_fit == parent_fit else "regressed"
    return f"{parent_fit:.6g} -> {child_fit:.6g} ({word})"


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


@dataclass
class Node:
    """受评算法个体，以及围绕该位置剪除旁支所凝练的节点经验。"""

    id: int
    code: str
    design: str
    fitness: float | None
    failure: str | None = None
    experience: str = ""
    direct_evaluations: int = 0

    def __post_init__(self) -> None:
        self.design = clean_text(self.design)
        self.experience = clean_text(self.experience)


@dataclass
class Edge:
    """算法之间的新生派生边或路径收缩边，以及收缩凝练的边经验。"""

    parent_id: int
    child_id: int
    operator: Operator | None
    reference_id: int | None = None
    experience: str = ""
    contracted: bool = False

    def __post_init__(self) -> None:
        self.experience = clean_text(self.experience)


@dataclass
class TreePopulation:
    """虚拟根编号固定为 ROOT_ID=0；算法节点与入边均按节点 id 索引。"""

    root_experience: str = ""
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: dict[int, Edge] = field(default_factory=dict)

    def add(self, node: Node, edge: Edge, *, evaluations: int = 0) -> None:
        if edge.parent_id != ROOT_ID:
            self.nodes[edge.parent_id].direct_evaluations += evaluations
        self.nodes[node.id] = node
        self.edges[node.id] = edge

    @classmethod
    def from_dict(cls, data: dict) -> "TreePopulation":
        return cls(
            root_experience=clean_text(data.get("root_experience", "")),
            nodes={int(i): Node(**n) for i, n in data["nodes"].items()},
            edges={int(i): Edge(**e) for i, e in data["edges"].items()},
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

    @property
    def valid(self) -> bool:
        return self.id != 0 and self.evaluation is not None and self.evaluation.status == "success"

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        data = dict(data)
        data.pop("analysis", None)
        if data["evaluation"] is not None:
            data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)


class FormatError(ValueError):
    """The model reply does not contain the design/code JSON object."""


@dataclass(frozen=True)
class Candidate:
    design: str
    code: str

    @classmethod
    def parse(cls, response: str) -> "Candidate":
        try:
            value = json.loads(response)
        except (TypeError, ValueError) as exc:
            raise FormatError(str(exc)) from exc
        if not isinstance(value, dict):
            raise FormatError(f"Expected a JSON object, got {type(value).__name__}")
        for name in ("design", "code"):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise FormatError(f"Field {name!r} must be a nonempty string")
        return cls(design=clean_text(value["design"]), code=value["code"])


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

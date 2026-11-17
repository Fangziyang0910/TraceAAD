"""树状种群、候选解析与搜索记录模型。"""

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


CANDIDATE_RESPONSE_FORMAT = json_schema("traceaad_candidate", "analysis", "code", "design")

_CTRL_REPAIR = str.maketrans({"\x08": r"\b", "\x0c": r"\f", "\r": " ", "\t": " "})


def clean_text(text: str | None) -> str:
    """修复 JSON 误转义的 LaTeX 控制字符（\\b、\\f）并折叠空白。"""
    if not text:
        return ""
    repaired = text.translate(_CTRL_REPAIR)
    repaired = "".join(ch for ch in repaired if ch == "\n" or ord(ch) >= 32)
    return " ".join(repaired.split())


def outcome_text(parent_fit: float | None, child_fit: float | None, *, include_scores: bool = True) -> str:
    if child_fit is None:
        return "failed"
    if parent_fit is None:
        return f"fitness={child_fit:.6g}" if include_scores else "evaluated (no algorithm parent)"
    word = "improved" if child_fit < parent_fit else "tied" if child_fit == parent_fit else "regressed"
    return f"{parent_fit:.6g} -> {child_fit:.6g} ({word})" if include_scores else word


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
    """已评价的算法版本，以及从它出发的选择经验。"""

    id: int
    code: str
    design: str
    fitness: float | None
    failure: str | None = None
    selection_experience: str = ""
    children_count: int = 0
    direction_id: int | None = None
    last_used: int = 0

    def __post_init__(self) -> None:
        self.design = clean_text(self.design)
        self.selection_experience = clean_text(self.selection_experience)


@dataclass
class Edge:
    """算法之间的发展关系，以及这段发展的变异经验。"""

    parent_id: int
    child_id: int
    operator: Operator | None
    reference_id: int | None = None
    variation_experience: str = ""
    contracted: bool = False

    def __post_init__(self) -> None:
        self.variation_experience = clean_text(self.variation_experience)


@dataclass
class TreePopulation:
    """虚拟根编号固定为 ROOT_ID=0；算法节点与入边均按节点 id 索引。"""

    root_selection_experience: str = ""
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: dict[int, Edge] = field(default_factory=dict)
    used_crossover_pairs: list[tuple[int, int]] = field(default_factory=list)
    evaluations_used: int = 0

    def direction_of(self, node_id: int) -> int:
        return int(self.nodes[node_id].direction_id)  # type: ignore[arg-type]

    def add(self, node: Node, edge: Edge, *, evaluations: int = 1) -> None:
        clock = self.evaluations_used + evaluations
        if edge.parent_id == ROOT_ID:
            direction = node.id
        else:
            parent = self.nodes[edge.parent_id]
            if edge.operator == "Explore":
                direction = node.id
            else:
                direction = self.direction_of(parent.id)
                parent.children_count += 1
                parent.last_used = clock
            if edge.operator == "Crossover" and edge.reference_id is not None:
                pair = (parent.id, edge.reference_id)
                if pair not in self.used_crossover_pairs:
                    self.used_crossover_pairs.append(pair)
        node.direction_id = direction
        node.last_used = clock
        self.evaluations_used = clock
        self.nodes[node.id] = node
        self.edges[node.id] = edge

    @classmethod
    def from_dict(cls, data: dict) -> "TreePopulation":
        return cls(
            root_selection_experience=clean_text(data.get("root_selection_experience", "")),
            nodes={int(i): Node(**n) for i, n in data["nodes"].items()},
            edges={int(i): Edge(**e) for i, e in data["edges"].items()},
            used_crossover_pairs=[(p, r) for p, r in data.get("used_crossover_pairs", ())],
            evaluations_used=int(data.get("evaluations_used", 0)),
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

    @property
    def valid(self) -> bool:
        return self.id != 0 and self.evaluation is not None and self.evaluation.status == "success"

    @classmethod
    def from_dict(cls, data: dict) -> "Program":
        data = dict(data)
        if data["evaluation"] is not None:
            data["evaluation"] = Evaluation(**data["evaluation"])
        return cls(**data)


class FormatError(ValueError):
    """The model reply does not contain the analysis/design/code object."""


@dataclass(frozen=True)
class Candidate:
    analysis: str
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
        for name in ("analysis", "design", "code"):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise FormatError(f"Field {name!r} must be a nonempty string")
        return cls(analysis=value["analysis"], design=clean_text(value["design"]), code=value["code"])


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

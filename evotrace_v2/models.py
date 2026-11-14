"""树状种群的节点、边与容器。"""

from dataclasses import dataclass, field
import json
import math
from typing import Literal

CANDIDATE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "traceaad_candidate",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analysis": {"type": "string"},
                "code": {"type": "string"},
                "design": {"type": "string"},
            },
            "required": ["analysis", "code", "design"],
            "additionalProperties": False,
        },
    },
}

Operator = Literal["Init", "Refine", "Crossover", "Explore"]
Status = Literal["success", "error", "timeout"]


@dataclass
class Node:
    """已评价的算法版本，以及从它出发的选择经验。"""

    id: int
    code: str
    design: str
    fitness: float | None  # None 表示不可行个体。
    failure: str | None = None
    selection_experience: str = ""
    children_count: int = 0  # 同路线开发尝试数（Refine / Crossover 主父代受评尝试）。
    direction_id: int | None = None  # 入树时确定；原始方向起点被压缩后仍保持。
    last_used: int = 0  # 出生或作为同路线主父代开发时的评价时钟。


@dataclass
class Edge:
    """算法之间的发展关系，以及这段发展的变异经验。"""

    parent_id: int
    child_id: int
    operator: Operator | None  # 虚拟根连接初始算法时为 None。
    reference_id: int | None = None
    variation_experience: str = ""
    contracted: bool = False


@dataclass
class TreePopulation:
    """虚拟根单独表示；算法节点按 id 索引，入边按 child_id 索引。"""

    root_id: int = 0
    root_selection_experience: str = ""
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: dict[int, Edge] = field(default_factory=dict)
    latest_node_id: int = 0
    direction_evaluations: dict[int, int] = field(default_factory=dict)
    used_crossover_pairs: list[tuple[int, int]] = field(default_factory=list)
    evaluations_used: int = 0

    def direction_of(self, node_id: int) -> int:
        """Return the original Init/Explore direction ID, including removed origins."""
        direction = self.nodes[node_id].direction_id
        if direction is None:
            raise ValueError("Use TreePopulation.add to assign a node's direction")
        return direction

    def add(self, node: Node, edge: Edge, *, evaluations: int) -> None:
        """Attach one evaluated version, counting its actual evaluations including repairs.

        Node IDs increase across consolidations. Each new Explore starts its own
        evaluation count; Refine and Crossover contribute to their parent's direction.
        """
        if node.id == self.root_id or node.id <= self.latest_node_id or node.id != edge.child_id:
            raise ValueError("New node IDs must increase and match edge.child_id")
        if type(evaluations) is not int or evaluations < 1:
            raise ValueError("evaluations must count at least one actual evaluation")
        if node.fitness is not None and not math.isfinite(node.fitness):
            raise ValueError("fitness must be finite or None")
        reference = None
        if edge.operator == "Crossover" and edge.reference_id is not None:
            reference = self.nodes.get(edge.reference_id)
            if reference is None or reference.fitness is None:
                raise ValueError("Crossover requires a retained feasible reference")
        clock = self.evaluations_used + evaluations
        if edge.parent_id == self.root_id:
            if edge.operator is not None:
                raise ValueError("Initial edges must have operator=None")
            direction = node.id
        else:
            parent = self.nodes.get(edge.parent_id)
            if parent is None or parent.fitness is None:
                raise ValueError("A new algorithm requires a retained feasible parent")
            if edge.operator not in ("Refine", "Crossover", "Explore"):
                raise ValueError("An evolutionary edge requires an operator")
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
        self.latest_node_id = node.id
        self.direction_evaluations[direction] = self.direction_evaluations.get(direction, 0) + evaluations


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
    """One generation attempt: its model calls, evaluations and final program.

    ``record`` holds method-specific information, such as trajectory allocation
    and the material shown to the model.
    """
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

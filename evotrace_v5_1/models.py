"""有界谱系森林模型、候选解析（analysis + code + design）与搜索记录结构。"""

from __future__ import annotations

import json
import math
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
    population_capacity: int = 24
    selection_lambda: float = 10.0
    explore_references: int = 5
    reflection_mask_prob: float = 0.0
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
        if self.population_capacity < 1 or self.explore_references < 1:
            raise ValueError("Population capacity and explore references must be positive")
        if not math.isfinite(self.selection_lambda) or self.selection_lambda < 0:
            raise ValueError("selection_lambda must be finite and nonnegative")
        if not math.isfinite(self.reflection_mask_prob) or not (
            0.0 <= self.reflection_mask_prob <= 1.0
        ):
            raise ValueError("reflection_mask_prob must be in [0.0, 1.0]")


@dataclass
class TreePopulation:
    """有界谱系森林：parents 存父子拓扑，idea_notes 存未兑现想法笔记，direct_evals 存直接投入评价数。"""

    parents: dict[int, int] = field(default_factory=dict)
    idea_notes: dict[int, str] = field(default_factory=dict)
    direct_evals: dict[int, int] = field(default_factory=dict)

    def children_of(self, node_id: int) -> list[int]:
        return sorted(cid for cid, pid in self.parents.items() if pid == node_id)

    def path_to(self, node_id: int) -> tuple[int, ...]:
        """从顶级谱系根（虚拟根直接子节点）到 node_id 的存活路径。"""
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

    def lineage_root(self, node_id: int) -> int:
        """拓扑不变量 L(v)：返回节点所在顶级子树的唯一根。"""
        path = self.path_to(node_id)
        if not path:
            raise KeyError(f"Node {node_id} is not in the active forest")
        return path[0]

    def active_lineages(self) -> dict[int, list[int]]:
        """按顶级谱系根分组返回全部存活节点。"""
        groups: dict[int, list[int]] = {}
        for nid in sorted(self.parents):
            root = self.lineage_root(nid)
            groups.setdefault(root, []).append(nid)
        return groups

    def snapshot(self) -> dict:
        return {
            "parents": self.parents.copy(),
            "idea_notes": self.idea_notes.copy(),
            "direct_evals": {nid: self.direct_evals.get(nid, 0) for nid in self.parents},
        }

    @classmethod
    def from_dict(cls, data: dict) -> TreePopulation:
        parents = {int(i): int(v) for i, v in data["parents"].items()}
        idea_notes = {
            int(i): t for i, v in data.get("idea_notes", {}).items() if (t := clean_text(v))
        }
        direct_evals = {int(i): int(v) for i, v in data.get("direct_evals", {}).items()}
        return cls(parents=parents, idea_notes=idea_notes, direct_evals=direct_evals)


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

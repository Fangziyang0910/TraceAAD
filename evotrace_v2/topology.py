"""一次保留选择、必要分叉闭包与反思区域划分。"""

import ast
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from difflib import unified_diff
from functools import lru_cache
import hashlib

from .models import Operator, ROOT_ID, TreePopulation


def _strip_docstrings(code: str) -> ast.AST:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


def bare_code(code: str) -> str:
    try:
        return ast.unparse(_strip_docstrings(code))
    except (SyntaxError, ValueError):
        return code.strip()


def compact_diff(old_code: str, new_code: str) -> str:
    a, b = bare_code(old_code).splitlines(), bare_code(new_code).splitlines()
    if a == b:
        return "(normalized code unchanged)"
    return "\n".join("..." if line.startswith("@@") else line for line in list(unified_diff(a, b, lineterm="", n=2))[2:])


@lru_cache(maxsize=4096)
def ast_shingles(code: str, k: int = 2) -> tuple[tuple[object, int], ...]:
    try:
        tree = _strip_docstrings(code)
    except (SyntaxError, ValueError):
        return tuple(sorted(Counter(code.split()).items()))

    tokens: list[str] = []
    counts: Counter[object] = Counter()

    def visit(node: ast.AST, parent_tag: str | None = None) -> None:
        tag = type(node).__name__
        if isinstance(node, ast.BinOp):
            tag = f"BinOp:{type(node.op).__name__}"
        elif isinstance(node, ast.UnaryOp):
            tag = f"UnaryOp:{type(node.op).__name__}"
        elif isinstance(node, ast.Compare):
            tag = "Compare:" + ",".join(type(op).__name__ for op in node.ops)
        elif isinstance(node, ast.AugAssign):
            tag = f"AugAssign:{type(node.op).__name__}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                tag = f"Call:{node.func.id}"
            elif isinstance(node.func, ast.Attribute):
                tag = f"CallAttr:{node.func.attr}"
        tokens.append(tag)
        if parent_tag is not None:
            counts[(parent_tag, tag)] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, tag)

    visit(tree)
    for idx in range(len(tokens) - k + 1):
        counts[tuple(tokens[idx : idx + k])] += 1
    return tuple(sorted(counts.items()))


@lru_cache(maxsize=4096)
def identity(code: str) -> str:
    try:
        text = ast.dump(_strip_docstrings(code), include_attributes=False)
    except (SyntaxError, ValueError):
        text = code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def code_distance(code_a: str, code_b: str) -> float:
    if code_a == code_b:
        return 0.0
    sh_a, sh_b = dict(ast_shingles(code_a)), dict(ast_shingles(code_b))
    keys = sh_a.keys() | sh_b.keys()
    union = sum(max(sh_a.get(k, 0), sh_b.get(k, 0)) for k in keys)
    if not union:
        return 0.0
    inter = sum(min(sh_a.get(k, 0), sh_b.get(k, 0)) for k in keys)
    return 1.0 - inter / union


@dataclass(frozen=True)
class TopologyConfig:
    cold_evaluations: int = 40
    quality_floor: float = 0.50
    clone_distance: float = 0.08
    same_fitness_clone_distance: float = 0.20

    def __post_init__(self) -> None:
        if type(self.cold_evaluations) is not int or self.cold_evaluations < 1:
            raise ValueError("cold_evaluations must be an integer >= 1")
        for name in ("quality_floor", "clone_distance", "same_fitness_clone_distance"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class PathContraction:
    parent_id: int
    child_id: int
    removed_ids: tuple[int, ...]
    merged_operator: Operator | None
    merged_reference_id: int | None
    side_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class BranchConsolidation:
    parent_id: int
    removed_ids: tuple[int, ...]


Region = PathContraction | BranchConsolidation


def quality_representatives(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> set[int]:
    """按适应度顺序选取未被同路线更优后代超越、且不与更优代表近重复的质量代表。"""
    feasible = sorted((i for i, n in tree.nodes.items() if n.fitness is not None),
                      key=lambda i: (tree.nodes[i].fitness, i))  # type: ignore[arg-type]
    kept: set[int] = set()
    overtaken: set[int] = set()
    for rank, i in enumerate(feasible):
        if 1 - rank / max(1, len(feasible) - 1) < config.quality_floor:
            break
        node = tree.nodes[i]
        represented = rank > 0 and i in overtaken
        for j in () if rank == 0 or represented else sorted(kept):
            other = tree.nodes[j]
            if other.fitness is None or other.fitness > node.fitness:  # type: ignore[operator]
                continue
            threshold = (max(config.clone_distance, config.same_fitness_clone_distance)
                         if abs(other.fitness - node.fitness) <= 1e-9 else config.clone_distance)  # type: ignore[operator]
            if identity(node.code) == identity(other.code) or code_distance(node.code, other.code) < threshold:
                represented = True
                break
        if not represented:
            kept.add(i)
        route = tree.direction_of(i)
        parent = tree.edges[i].parent_id
        while parent != ROOT_ID:
            p_fit = tree.nodes[parent].fitness
            if tree.direction_of(parent) == route and p_fit is not None and node.fitness < p_fit:  # type: ignore[operator]
                overtaken.add(parent)
            parent = tree.edges[parent].parent_id
    return kept


def plan_consolidation(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> tuple[Region, ...]:
    """保留质量代表与近期活跃个体，补齐分叉祖先，划分路径收缩与旁支区域。"""
    protected = quality_representatives(tree, config)

    def last_activity(i: int) -> int:
        parent = tree.edges[i].parent_id
        own = tree.nodes[i].last_used
        if parent == ROOT_ID or tree.direction_of(parent) != tree.direction_of(i):
            return own
        return max(own, tree.nodes[parent].last_used)

    recent = {i for i, node in tree.nodes.items()
              if node.fitness is not None and tree.evaluations_used - last_activity(i) < config.cold_evaluations}
    selected = protected | recent

    children = defaultdict(list)
    for i, edge in tree.edges.items():
        children[edge.parent_id].append(i)
    used = set(selected)
    for i in sorted(tree.nodes, reverse=True):
        if i in used:
            used.add(tree.edges[i].parent_id)
    forks = {i for i in used if i != ROOT_ID and sum(c in used for c in children[i]) >= 2}
    alive = selected | forks
    removed_all = tree.nodes.keys() - alive

    raw_contractions = []
    contracted_to_idx: dict[int, int] = {}
    for child in sorted(alive):
        parent = tree.edges[child].parent_id
        removed_chain: list[int] = []
        edges = [tree.edges[child]]
        while parent != ROOT_ID and parent not in alive:
            removed_chain.append(parent)
            edges.append(tree.edges[parent])
            parent = tree.edges[parent].parent_id
        if not removed_chain:
            continue
        removed_chain.reverse()
        edges.reverse()
        crosses = [e for e in edges if e.operator == "Crossover" or e.reference_id is not None]
        op: Operator | None = (
            None if parent == ROOT_ID
            else "Explore" if tree.direction_of(parent) != tree.direction_of(child)
            else "Crossover" if crosses else "Refine"
        )
        idx = len(raw_contractions)
        raw_contractions.append((parent, child, tuple(removed_chain), op, crosses[-1].reference_id if crosses else None))
        for mid in removed_chain:
            contracted_to_idx[mid] = idx

    edge_sides: dict[int, list[int]] = defaultdict(list)
    sides: dict[int, list[int]] = defaultdict(list)
    for i in sorted(removed_all - contracted_to_idx.keys()):
        anc = tree.edges[i].parent_id
        while anc != ROOT_ID and anc not in alive and anc not in contracted_to_idx:
            anc = tree.edges[anc].parent_id
        if anc in contracted_to_idx:
            edge_sides[contracted_to_idx[anc]].append(i)
        else:
            sides[anc].append(i)

    contractions = (
        PathContraction(p, c, rem, op, ref, tuple(edge_sides.get(idx, ())))
        for idx, (p, c, rem, op, ref) in enumerate(raw_contractions)
    )
    return (*contractions, *(BranchConsolidation(p, tuple(ids)) for p, ids in sorted(sides.items())))


def apply_consolidation(tree: TreePopulation, regions: Sequence[Region]) -> TreePopulation:
    """移除简化区域中的节点，重连收缩边并保留评价时钟与交叉记录。"""
    removed = {
        i for r in regions
        for i in ((*r.removed_ids, *r.side_ids) if isinstance(r, PathContraction) else r.removed_ids)
    }
    kept = sorted(tree.nodes.keys() - removed)
    kept_set = set(kept)
    contractions = {r.child_id: r for r in regions if isinstance(r, PathContraction)}
    return TreePopulation(
        root_selection_experience=tree.root_selection_experience,
        nodes={i: replace(tree.nodes[i]) for i in kept},
        edges={
            i: (replace(tree.edges[i], parent_id=contractions[i].parent_id, operator=contractions[i].merged_operator,
                        reference_id=contractions[i].merged_reference_id, contracted=True)
                if i in contractions else replace(tree.edges[i]))
            for i in kept
        },
        used_crossover_pairs=[pair for pair in tree.used_crossover_pairs if pair[0] in kept_set and pair[1] in kept_set],
        evaluations_used=tree.evaluations_used,
    )

"""统一算法选择、结构闭包与路径/旁支重构。"""

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
    """去除文档字符串后的标准化源码。"""
    try:
        return ast.unparse(_strip_docstrings(code))
    except (SyntaxError, ValueError):
        return code.strip()


def compact_diff(old_code: str, new_code: str) -> str:
    """计算两段标准化源码的紧凑 unified diff。"""
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
    base_capacity: int = 32
    trigger_capacity: int = 64
    quality_floor: float = 0.30
    clone_distance: float = 0.08
    same_fitness_clone_distance: float = 0.20
    min_protected_evaluations: int = 4
    protected_new_directions: int = 2
    recent_node_window: int = 16

    def __post_init__(self) -> None:
        if self.base_capacity < 2:
            raise ValueError("base_capacity must be at least 2")
        if self.trigger_capacity <= self.base_capacity:
            raise ValueError("trigger_capacity must exceed base_capacity")
        if not (0 <= self.quality_floor <= 1 and 0 <= self.clone_distance <= 1 and 0 <= self.same_fitness_clone_distance <= 1):
            raise ValueError("quality_floor, clone_distance, and same_fitness_clone_distance must be in [0, 1]")
        if self.min_protected_evaluations < 1:
            raise ValueError("min_protected_evaluations must be positive")
        if self.protected_new_directions < 0 or self.recent_node_window < 0:
            raise ValueError("Protection count and recent_node_window must be nonnegative")


@dataclass(frozen=True)
class PathContraction:
    parent_id: int
    child_id: int
    removed_ids: tuple[int, ...]
    merged_operator: Operator | None
    merged_reference_id: int | None


@dataclass(frozen=True)
class BranchConsolidation:
    parent_id: int
    removed_ids: tuple[int, ...]


Region = PathContraction | BranchConsolidation


def needs_consolidation(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> bool:
    return len(tree.nodes) >= config.trigger_capacity


def plan_consolidation(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> tuple[Region, ...]:
    """在统一容量上限下选择代表个体、补齐 LCA 与根入口骨架，并划分路径收缩与旁支区域。"""
    children: dict[int, list[int]] = defaultdict(list)
    paths: dict[int, tuple[int, ...]] = {}
    for i in sorted(tree.nodes):
        p = tree.edges[i].parent_id
        children[p].append(i)
        paths[i] = (() if p == ROOT_ID else paths[p]) + (i,)

    def closure(selected: set[int]) -> set[int]:
        used = {ROOT_ID, *(n for i in selected for n in paths[i])}
        lcas = {i for i in used if i != ROOT_ID and sum(c in used for c in children[i]) >= 2}
        boundaries = {i for i in used - {ROOT_ID} if tree.edges[i].parent_id == ROOT_ID}
        return selected | lcas | boundaries

    feasible = sorted((i for i, n in tree.nodes.items() if n.fitness is not None),
                      key=lambda i: (tree.nodes[i].fitness, i))  # type: ignore[arg-type]
    if len(feasible) <= config.base_capacity:
        kept = closure(set(feasible))
    else:
        denom = max(1, len(feasible) - 1)
        quality = {i: 1.0 - rank / denom for rank, i in enumerate(feasible)}
        kept = closure({feasible[0]})

        reps: dict[int, int] = {}
        for i in feasible:
            reps.setdefault(tree.direction_of(i), i)
        cutoff = tree.latest_node_id - config.recent_node_window
        young = sorted(
            (i for d, i in reps.items()
             if d != paths[i][0] and d >= cutoff and tree.direction_evaluations[d] < config.min_protected_evaluations),
            key=lambda i: (-quality[i], i),
        )
        protected = 0
        for cand in young:
            if protected >= config.protected_new_directions:
                break
            nxt = closure(kept | {cand})
            if len(nxt) <= config.base_capacity:
                kept = nxt
                protected += 1

        distances: dict[tuple[int, int], float] = {}

        def dist(a: int, b: int) -> float:
            if a == b:
                return 0.0
            key = (min(a, b), max(a, b))
            if key not in distances:
                distances[key] = code_distance(tree.nodes[a].code, tree.nodes[b].code)
            return distances[key]

        candidates = {i for i, q in quality.items() if q >= config.quality_floor} - kept
        nearest = {i: min(dist(i, j) for j in kept) for i in candidates}
        while len(kept) < config.base_capacity:
            winner, winner_set, winner_score = None, None, -1.0
            for i in sorted(candidates):
                if nearest[i] < config.clone_distance:
                    continue
                if config.same_fitness_clone_distance > config.clone_distance:
                    fi = float(tree.nodes[i].fitness)  # type: ignore[arg-type]
                    if any(abs(float(tree.nodes[j].fitness) - fi) <= 1e-9  # type: ignore[arg-type]
                           and dist(i, j) < config.same_fitness_clone_distance for j in kept):
                        continue
                score = quality[i] * (0.65 + 0.35 * nearest[i])
                if score > winner_score:
                    nxt = closure(kept | {i})
                    if len(nxt) <= config.base_capacity:
                        winner, winner_set, winner_score = i, nxt, score
            if winner is None or winner_set is None:
                break
            added = winner_set - kept
            kept = winner_set
            candidates -= kept
            for i in candidates:
                nearest[i] = min(nearest[i], *(dist(i, j) for j in added))

    contractions: list[PathContraction] = []
    contracted: set[int] = set()
    for child in sorted(kept):
        chain = [tree.edges[child]]
        removed: list[int] = []
        parent = tree.edges[child].parent_id
        while parent != ROOT_ID and parent not in kept:
            removed.append(parent)
            chain.append(tree.edges[parent])
            parent = tree.edges[parent].parent_id
        if removed:
            chain.reverse()
            removed.reverse()
            crossovers = [e for e in chain if e.operator == "Crossover" or e.reference_id is not None]
            op: Operator | None = (
                None if parent == ROOT_ID
                else "Explore" if tree.direction_of(parent) != tree.direction_of(child)
                else "Crossover" if crossovers else "Refine"
            )
            ref = crossovers[-1].reference_id if crossovers else None
            contractions.append(PathContraction(parent, child, tuple(removed), op, ref))
            contracted.update(removed)

    sides: dict[int, list[int]] = defaultdict(list)
    for i in sorted(tree.nodes.keys() - kept - contracted):
        parent = tree.edges[i].parent_id
        while parent != ROOT_ID and parent not in kept:
            parent = tree.edges[parent].parent_id
        sides[parent].append(i)

    return (*contractions, *(BranchConsolidation(p, tuple(ids)) for p, ids in sorted(sides.items())))


def apply_consolidation(tree: TreePopulation, regions: Sequence[Region]) -> TreePopulation:
    """移除简化区域中的节点，重连收缩边并保留存活方向的评价计数。"""
    removed = {i for r in regions for i in r.removed_ids}
    kept = sorted(tree.nodes.keys() - removed)
    kept_set = set(kept)
    contractions = {r.child_id: r for r in regions if isinstance(r, PathContraction)}
    nodes = {i: replace(tree.nodes[i]) for i in kept}
    edges = {
        i: (replace(tree.edges[i], parent_id=contractions[i].parent_id, operator=contractions[i].merged_operator,
                    reference_id=contractions[i].merged_reference_id, contracted=True)
            if i in contractions else replace(tree.edges[i]))
        for i in kept
    }
    active_directions = {n.direction_id for n in nodes.values()}
    return TreePopulation(
        root_selection_experience=tree.root_selection_experience,
        nodes=nodes,
        edges=edges,
        latest_node_id=tree.latest_node_id,
        direction_evaluations={d: c for d, c in tree.direction_evaluations.items() if d in active_directions},
        used_crossover_pairs=[pair for pair in tree.used_crossover_pairs if pair[0] in kept_set and pair[1] in kept_set],
    )

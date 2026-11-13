"""统一算法选择、结构闭包与路径/旁支重构。"""

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib

from .models import Edge, Node, Operator, TreePopulation


def _strip_docstrings(code: str) -> ast.AST:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


@lru_cache(maxsize=4096)
def ast_shingles(code: str, k: int = 2) -> tuple[tuple[object, int], ...]:
    """提取对局部变量名与数值常数不敏感的 AST 父子结构边与节点序列二元组特征。"""
    try:
        tree = _strip_docstrings(code)
    except (SyntaxError, ValueError):
        counts = Counter(code.split())
        return tuple(sorted(counts.items()))

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
    """Same parsed source, ignoring layout, comments and docstrings."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
                first = node.body[0]
                if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                        and isinstance(first.value.value, str)):
                    node.body = node.body[1:]
        text = ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError):
        text = code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def code_distance(code_a: str, code_b: str) -> float:
    """计算两段代码的归一化 AST 多重集 Jaccard 距离，取值 [0.0, 1.0]。"""
    if code_a == code_b:
        return 0.0
    sh_a = dict(ast_shingles(code_a))
    sh_b = dict(ast_shingles(code_b))
    if not sh_a and not sh_b:
        return 0.0
    keys = sh_a.keys() | sh_b.keys()
    inter = sum(min(sh_a.get(k, 0), sh_b.get(k, 0)) for k in keys)
    union = sum(max(sh_a.get(k, 0), sh_b.get(k, 0)) for k in keys)
    return 1.0 - (inter / union) if union else 0.0


@dataclass(frozen=True)
class TopologyConfig:
    """Population capacity and the shared quality/diversity selection rule."""

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
    """Connect retained endpoints across removed intermediate versions."""

    parent_id: int
    child_id: int
    removed_ids: tuple[int, ...]
    merged_operator: Operator | None
    merged_reference_id: int | None


@dataclass(frozen=True)
class BranchConsolidation:
    """Removed side branches owned by their nearest retained ancestor."""

    parent_id: int
    removed_ids: tuple[int, ...]


@dataclass(frozen=True)
class ConsolidationPlan:
    """One retained set, its selection reasons, and the regions to reflect on."""

    kept_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    lca_anchor_ids: tuple[int, ...]
    direction_anchor_ids: tuple[int, ...]
    protected_ids: tuple[int, ...] = ()
    protection_capacity_skips: tuple[int, ...] = ()
    path_contractions: tuple[PathContraction, ...] = ()
    branch_consolidations: tuple[BranchConsolidation, ...] = ()


def needs_consolidation(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> bool:
    return len(tree.nodes) >= config.trigger_capacity


class _TreeIndex:
    def __init__(self, tree: TreePopulation):
        self.tree = tree
        self.children: dict[int, list[int]] = defaultdict(list)
        self.paths: dict[int, tuple[int, ...]] = {}
        self.direction: dict[int, int] = {}
        if set(tree.nodes) != set(tree.edges) or tree.root_id in tree.nodes:
            raise ValueError("Each algorithm must have one incoming edge; the root is virtual")
        if tree.nodes and tree.latest_node_id < max(tree.nodes):
            raise ValueError("Use TreePopulation.add to record node IDs and actual evaluations")
        for i in sorted(tree.nodes):
            edge = tree.edges[i]
            p = edge.parent_id
            if edge.child_id != i or (p != tree.root_id and p not in self.paths):
                raise ValueError("Parent nodes must precede their children")
            if p != tree.root_id and tree.nodes[p].fitness is None:
                raise ValueError("Infeasible algorithms must remain leaves")
            self.children[p].append(i)
            self.paths[i] = (() if p == tree.root_id else self.paths[p]) + (i,)
            self.direction[i] = tree.direction_of(i)
        sizes = Counter(self.direction.values())
        if set(sizes) != set(tree.direction_evaluations) or any(
            tree.direction_evaluations[d] < count for d, count in sizes.items()
        ):
            raise ValueError("Direction counts must include every actual evaluation, including removed versions")

    def closure(self, selected: set[int]) -> tuple[set[int], set[int], set[int]]:
        used = {self.tree.root_id}
        for i in selected:
            used.update(self.paths[i])
        lcas = {i for i in used if i != self.tree.root_id
                and sum(c in used for c in self.children[i]) >= 2}
        boundaries: set[int] = set()
        for i in used - {self.tree.root_id}:
            edge = self.tree.edges[i]
            if edge.parent_id == self.tree.root_id:
                boundaries.add(i)
        return selected | lcas | boundaries, lcas, boundaries


def _partition(
    tree: TreePopulation, index: _TreeIndex, kept: set[int], selected: set[int],
    protected: list[int], capacity_skips: list[int],
) -> ConsolidationPlan:
    _, lcas, boundaries = index.closure(kept)
    contractions = []
    contracted: set[int] = set()
    for child in sorted(kept):
        chain = [tree.edges[child]]
        removed = []
        parent = tree.edges[child].parent_id
        while parent != tree.root_id and parent not in kept:
            removed.append(parent)
            chain.append(tree.edges[parent])
            parent = tree.edges[parent].parent_id
        if removed:
            chain.reverse()
            removed.reverse()
            crossovers = [e for e in chain if e.operator == "Crossover" or e.reference_id is not None]
            if index.direction[parent] != index.direction[child]:
                operator = "Explore"
            else:
                operator = "Crossover" if crossovers else "Refine"
            reference = crossovers[-1].reference_id if crossovers else None
            contractions.append(PathContraction(parent, child, tuple(removed), operator, reference))
            contracted.update(removed)
    sides: dict[int, list[int]] = defaultdict(list)
    for i in sorted(set(tree.nodes) - kept - contracted):
        parent = tree.edges[i].parent_id
        while parent != tree.root_id and parent not in kept:
            parent = tree.edges[parent].parent_id
        sides[parent].append(i)
    return ConsolidationPlan(
        kept_ids=tuple(sorted(kept)), selected_ids=tuple(sorted(selected)),
        lca_anchor_ids=tuple(sorted(lcas)), direction_anchor_ids=tuple(sorted(boundaries)),
        protected_ids=tuple(sorted(protected)), protection_capacity_skips=tuple(capacity_skips),
        path_contractions=tuple(contractions),
        branch_consolidations=tuple(BranchConsolidation(p, tuple(ids)) for p, ids in sorted(sides.items())),
    )


def plan_consolidation(
    tree: TreePopulation,
    config: TopologyConfig = TopologyConfig(),
) -> ConsolidationPlan:
    """Select algorithms and their required structure under one capacity limit.

    Internal versions and leaves use the same quality/distance rule. Structural
    nodes participate in subsequent distance comparisons. This function is pure;
    actual evaluation counts are maintained by TreePopulation.add.
    """
    index = _TreeIndex(tree)
    feasible = sorted((i for i, n in tree.nodes.items() if n.fitness is not None),
                      key=lambda i: (tree.nodes[i].fitness, i))
    denominator = max(1, len(feasible) - 1)
    quality = {i: 1.0 - rank / denominator for rank, i in enumerate(feasible)}
    protected: list[int] = []
    capacity_skips: list[int] = []
    if len(feasible) <= config.base_capacity:
        selected = set(feasible)
        kept, _, _ = index.closure(selected)
        return _partition(tree, index, kept, selected, protected, capacity_skips)

    champion = feasible[0]
    selected = {champion}
    kept, _, _ = index.closure(selected)
    # Retention protection delays removal; it does not reserve evaluation budget.
    representatives: dict[int, int] = {}
    for i in feasible:
        representatives.setdefault(index.direction[i], i)
    cutoff = tree.latest_node_id - config.recent_node_window
    young = sorted(
        (i for d, i in representatives.items()
         if d != index.paths[i][0] and d >= cutoff
         and tree.direction_evaluations[d] < config.min_protected_evaluations),
        key=lambda i: (-quality[i], i),
    )
    for candidate in young:
        if len(protected) >= config.protected_new_directions:
            break
        result, _, _ = index.closure(kept | {candidate})
        if len(result) > config.base_capacity:
            capacity_skips.append(candidate)
            continue
        protected.append(candidate)
        selected.add(candidate)
        kept = result

    distances: dict[tuple[int, int], float] = {}

    def distance(a: int, b: int) -> float:
        if a == b:
            return 0.0
        key = (min(a, b), max(a, b))
        if key not in distances:
            distances[key] = code_distance(tree.nodes[a].code, tree.nodes[b].code)
        return distances[key]

    candidates = {i for i in quality if quality[i] >= config.quality_floor} - kept
    nearest = {i: min(distance(i, j) for j in kept) for i in candidates}
    while len(kept) < config.base_capacity:
        winner = None
        winner_set = None
        winner_score = -1.0
        for i in sorted(candidates):
            if nearest[i] < config.clone_distance:
                continue
            if config.same_fitness_clone_distance > config.clone_distance:
                fi = float(tree.nodes[i].fitness)  # type: ignore[arg-type]
                if any(
                    abs(float(tree.nodes[j].fitness) - fi) <= 1e-9  # type: ignore[arg-type]
                    and distance(i, j) < config.same_fitness_clone_distance
                    for j in kept
                ):
                    continue
            score = quality[i] * (0.65 + 0.35 * nearest[i])
            if score <= winner_score:
                continue
            result, _, _ = index.closure(kept | {i})
            if len(result) <= config.base_capacity:
                winner, winner_set, winner_score = i, result, score
        if winner is None:
            break
        added = winner_set - kept
        selected.add(winner)
        kept = winner_set
        candidates -= kept
        for i in candidates:
            nearest[i] = min(nearest[i], *(distance(i, j) for j in added))
    return _partition(tree, index, kept, selected, protected, capacity_skips)


def apply_consolidation(tree: TreePopulation, plan: ConsolidationPlan) -> TreePopulation:
    """Reconnect retained versions, carrying forward experience and evaluation counts."""
    kept = set(plan.kept_ids)
    contractions = {c.child_id: c for c in plan.path_contractions}
    nodes = {i: replace(tree.nodes[i]) for i in plan.kept_ids}
    edges = {}
    for i in plan.kept_ids:
        edge = tree.edges[i]
        if i in contractions:
            c = contractions[i]
            edges[i] = replace(edge, parent_id=c.parent_id, operator=c.merged_operator,
                               reference_id=c.merged_reference_id, contracted=True)
        else:
            edges[i] = replace(edge)
    active_directions = {node.direction_id for node in nodes.values()}
    return TreePopulation(
        root_id=tree.root_id, root_selection_experience=tree.root_selection_experience,
        nodes=nodes, edges=edges, latest_node_id=tree.latest_node_id,
        direction_evaluations={
            d: count for d, count in tree.direction_evaluations.items()
            if d in active_directions
        },
        used_crossover_pairs=[
            pair for pair in tree.used_crossover_pairs if pair[0] in kept and pair[1] in kept
        ],
    )


def consolidate_topology(
    tree: TreePopulation,
    config: TopologyConfig = TopologyConfig(),
) -> tuple[TreePopulation, ConsolidationPlan]:
    """Select and reconnect; reflective_simplify additionally updates experience."""
    plan = plan_consolidation(tree, config)
    return apply_consolidation(tree, plan), plan

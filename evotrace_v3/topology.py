"""反思树简化（Reflective Tree Simplification）：路径收缩与子树剪枝。"""

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math

from .models import TreePopulation


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
        text = ast.dump(_strip_docstrings(code), include_attributes=False)
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
    """Representative coverage and batch size for reflective tree simplification.

    max_representatives: maximum number of mutually uncovered feasible representatives.
    cover_distance: AST shingle Jaccard distance below which a better representative
        covers a later candidate.
    region_size: minimum number of non-skeleton nodes accumulated in a path or subtree
        region before that region is simplified into edge or node experience.
    """

    max_representatives: int = 24
    cover_distance: float = 0.15
    region_size: int = 4

    def __post_init__(self):
        for name in ("max_representatives", "region_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if (
            isinstance(self.cover_distance, bool)
            or not isinstance(self.cover_distance, (int, float))
            or not math.isfinite(self.cover_distance)
            or not (0.0 <= self.cover_distance <= 1.0)
        ):
            raise ValueError("cover_distance must be a finite number in [0, 1]")


@dataclass(frozen=True)
class PathContraction:
    parent_id: int
    child_id: int
    removed_ids: tuple[int, ...]
    branch_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SubtreePruning:
    parent_id: int
    removed_ids: tuple[int, ...]


@dataclass(frozen=True)
class SimplificationPlan:
    kept_ids: tuple[int, ...]
    path_contractions: tuple[PathContraction, ...] = ()
    subtree_prunings: tuple[SubtreePruning, ...] = ()
    representative_ids: tuple[int, ...] = ()
    fork_ids: tuple[int, ...] = ()


def distinct_individuals(tree: TreePopulation) -> set[int]:
    """Collapse identical normalized source and measured score to the earliest retained node."""
    first = {}
    for i in sorted(tree.nodes):
        node = tree.nodes[i]
        if node.fitness is not None:
            first.setdefault((identity(node.code), node.fitness), i)
    return set(first.values())


def ranks(tree: TreePopulation, candidates) -> dict[int, float]:
    """0-based fitness rank among ``candidates`` (lower fitness is better); equal fitness shares the average rank."""
    ordered = sorted(candidates, key=lambda i: (tree.nodes[i].fitness, i))
    result = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and tree.nodes[ordered[end]].fitness == tree.nodes[ordered[start]].fitness:
            end += 1
        result.update((i, (start + end - 1) / 2) for i in ordered[start:end])
        start = end
    return result


def representatives(
    tree: TreePopulation,
    config: TopologyConfig = TopologyConfig(),
) -> tuple[int, ...]:
    """Select up to ``max_representatives`` mutually uncovered feasible individuals in fitness order."""
    ordered = sorted(
        distinct_individuals(tree),
        key=lambda i: (tree.nodes[i].fitness, i),  # type: ignore[arg-type]
    )
    selected: list[int] = []
    for i in ordered:
        code_i = tree.nodes[i].code
        if any(code_distance(code_i, tree.nodes[r].code) < config.cover_distance for r in selected):
            continue
        selected.append(i)
        if len(selected) >= config.max_representatives:
            break
    return tuple(selected)


def plan_simplification(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> SimplificationPlan:
    """Plan batched path contractions and subtree prunings on the evolutionary tree.

    1. Select the representative set ``R`` via greedy fitness-ordered coverage and close
       ``R`` under pairwise LCAs so branching structure between retained lineages is preserved.
    2. Partition remaining nodes into intermediate chains between retained endpoints
       (path contractions -> edge experience) and unretained subtrees under retained
       ancestors (subtree prunings -> node experience).
    3. Simplify only regions that have accumulated at least ``region_size`` nodes.
    """
    reps = set(representatives(tree, config))

    children = defaultdict(list)
    for i in sorted(tree.edges):
        children[tree.edges[i].parent_id].append(i)
    # A parent always has a smaller ID than its child, so reverse ID order visits children first.
    used = set(reps)
    for i in sorted(tree.nodes, reverse=True):
        if i in used:
            used.add(tree.edges[i].parent_id)
    forks = {i for i in used if i != tree.root_id and sum(c in used for c in children[i]) >= 2}
    retained_skeleton = reps | forks
    inactive = set(tree.nodes) - retained_skeleton

    raw_contractions = []
    contracted_to_idx: dict[int, int] = {}
    for child in sorted(retained_skeleton):
        parent = tree.edges[child].parent_id
        removed_chain = []
        while parent != tree.root_id and parent not in retained_skeleton:
            removed_chain.append(parent)
            parent = tree.edges[parent].parent_id
        if not removed_chain:
            continue
        removed_chain.reverse()
        idx = len(raw_contractions)
        raw_contractions.append((parent, child, tuple(removed_chain)))
        for mid in removed_chain:
            contracted_to_idx[mid] = idx

    path_branches: dict[int, list[int]] = defaultdict(list)
    subtrees: dict[int, list[int]] = defaultdict(list)
    for i in sorted(inactive - set(contracted_to_idx)):
        anc = tree.edges[i].parent_id
        while anc != tree.root_id and anc not in retained_skeleton and anc not in contracted_to_idx:
            anc = tree.edges[anc].parent_id
        if anc in contracted_to_idx:
            path_branches[contracted_to_idx[anc]].append(i)
        else:
            subtrees[anc].append(i)

    contractions = tuple(
        PathContraction(p, c, rem, tuple(path_branches.get(idx, ())))
        for idx, (p, c, rem) in enumerate(raw_contractions)
        if len(rem) + len(path_branches.get(idx, ())) >= config.region_size
    )
    prunings = tuple(
        SubtreePruning(p, tuple(ids))
        for p, ids in sorted(subtrees.items())
        if len(ids) >= config.region_size
    )
    removed = {
        *(i for c in contractions for i in (*c.removed_ids, *c.branch_ids)),
        *(i for b in prunings for i in b.removed_ids),
    }
    kept = tuple(sorted(set(tree.nodes) - removed))
    return SimplificationPlan(kept, contractions, prunings, tuple(sorted(reps)), tuple(sorted(forks)))


def apply_simplification(tree: TreePopulation, plan: SimplificationPlan) -> TreePopulation:
    """Apply planned path contractions and subtree prunings while keeping surviving effort counts."""
    kept = set(plan.kept_ids)
    contractions = {c.child_id: c for c in plan.path_contractions}
    nodes = {i: replace(tree.nodes[i]) for i in plan.kept_ids}
    edges = {}
    for i in plan.kept_ids:
        edge = tree.edges[i]
        if i in contractions:
            c = contractions[i]
            edges[i] = replace(edge, parent_id=c.parent_id, operator=None, reference_id=None, contracted=True)
        else:
            edges[i] = replace(edge)
    return TreePopulation(
        root_id=tree.root_id,
        root_experience=tree.root_experience,
        nodes=nodes,
        edges=edges,
        latest_node_id=tree.latest_node_id,
        used_crossover_pairs=[pair for pair in tree.used_crossover_pairs
                              if pair[0] in kept and pair[1] in kept],
        evaluations_used=tree.evaluations_used,
    )

"""一次保留选择、必要分叉闭包与反思区域划分。"""

import ast
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib

from .models import Operator, TreePopulation


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
    """Retire inactive versions while preserving quality and distinct computations."""

    cold_evaluations: int = 40
    quality_floor: float = 0.50
    clone_distance: float = 0.08
    same_fitness_clone_distance: float = 0.20

    def __post_init__(self):
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


@dataclass(frozen=True)
class ConsolidationPlan:
    kept_ids: tuple[int, ...]
    path_contractions: tuple[PathContraction, ...] = ()
    branch_consolidations: tuple[BranchConsolidation, ...] = ()
    triggers: tuple[dict, ...] = ()


def _select(tree, candidates, mandatory, config):
    """Select quality representatives before structural closure.

    A version stops representing its line once a better version developed from
    it is ranked; a near copy is represented by a version that is no worse. An
    untried node has the same eligibility as a developed node. Similarity is a
    selection heuristic, not a claim that future development is equivalent.
    """
    feasible = sorted((i for i in candidates if tree.nodes[i].fitness is not None),
                      key=lambda i: (tree.nodes[i].fitness, i))
    kept = set(mandatory)
    overtaken = set()
    for rank, i in enumerate(feasible):
        quality = 1 - rank / max(1, len(feasible) - 1)
        if quality < config.quality_floor:
            break
        node = tree.nodes[i]
        represented = rank > 0 and i in overtaken  # the champion always stays
        for j in () if rank == 0 or represented else sorted(kept):
            other = tree.nodes[j]
            if other.fitness is None or other.fitness > node.fitness:
                continue
            distance = code_distance(node.code, other.code)
            threshold = (max(config.clone_distance, config.same_fitness_clone_distance)
                         if abs(other.fitness - node.fitness) <= 1e-9 else config.clone_distance)
            if identity(node.code) == identity(other.code) or distance < threshold:
                represented = True
                break
        if not represented:
            kept.add(i)
        route = tree.direction_of(i)
        parent = tree.edges[i].parent_id
        while parent != tree.root_id:
            p_fit = tree.nodes[parent].fitness
            if tree.direction_of(parent) == route and p_fit is not None and node.fitness < p_fit:
                overtaken.add(parent)
            parent = tree.edges[parent].parent_id
    return kept


def quality_representatives(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> set[int]:
    """Quality/difference selection on this tree, before recency or fork closure."""
    return _select(tree, set(tree.nodes), set(), config)


def plan_consolidation(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> ConsolidationPlan:
    """Select once on the pre-update tree, then close and partition that same tree.

    A version stays while it, or its same-route tree parent, is still in local
    development. Quality representatives are fixed once before fork closure;
    side branches off contracted intermediates join that reconnected edge.
    """
    feasible = {i for i, node in tree.nodes.items() if node.fitness is not None}
    protected = quality_representatives(tree, config)
    def last_activity(i):
        parent = tree.edges[i].parent_id
        own = tree.nodes[i].last_used
        if parent == tree.root_id or tree.direction_of(parent) != tree.direction_of(i):
            return own
        return max(own, tree.nodes[parent].last_used)

    recent = {i for i in feasible
              if tree.evaluations_used - last_activity(i) < config.cold_evaluations}
    selected = protected | recent
    children = defaultdict(list)
    for i, edge in tree.edges.items():
        children[edge.parent_id].append(i)
    used = set(selected)
    for i in sorted(tree.nodes, reverse=True):
        if i in used:
            used.add(tree.edges[i].parent_id)
    forks = {i for i in used if i != tree.root_id
             and sum(c in used for c in children[i]) >= 2}
    alive = selected | forks
    removed_all = set(tree.nodes) - alive
    triggers = [{"kind": "inactive", "evaluations": tree.evaluations_used,
                 "protected_ids": sorted(protected), "recent_ids": sorted(recent),
                 "connection_ids": sorted(alive - selected), "removed_ids": sorted(removed_all)}] if removed_all else []
    # Partition on the original topology: intermediate chains form path contractions;
    # remaining side nodes join the first non-side ancestor (contracted edge or retained node/root).
    raw_contractions = []
    contracted_to_idx: dict[int, int] = {}
    for child in sorted(alive):
        parent = tree.edges[child].parent_id
        removed_chain = []
        edges = [tree.edges[child]]
        while parent != tree.root_id and parent not in alive:
            removed_chain.append(parent)
            edges.append(tree.edges[parent])
            parent = tree.edges[parent].parent_id
        if not removed_chain:
            continue
        removed_chain.reverse()
        edges.reverse()
        crosses = [e for e in edges if e.operator == "Crossover" or e.reference_id is not None]
        operator = (None if parent == tree.root_id else
                    "Explore" if tree.direction_of(parent) != tree.direction_of(child)
                    else "Crossover" if crosses else "Refine")
        idx = len(raw_contractions)
        raw_contractions.append((parent, child, tuple(removed_chain), operator,
                                 crosses[-1].reference_id if crosses else None))
        for mid in removed_chain:
            contracted_to_idx[mid] = idx
    edge_sides: dict[int, list[int]] = defaultdict(list)
    sides: dict[int, list[int]] = defaultdict(list)
    for i in sorted(removed_all - set(contracted_to_idx)):
        anc = tree.edges[i].parent_id
        while anc != tree.root_id and anc not in alive and anc not in contracted_to_idx:
            anc = tree.edges[anc].parent_id
        if anc in contracted_to_idx:
            edge_sides[contracted_to_idx[anc]].append(i)
        else:
            sides[anc].append(i)
    contractions = tuple(
        PathContraction(p, c, rem, op, ref, tuple(edge_sides.get(idx, ())))
        for idx, (p, c, rem, op, ref) in enumerate(raw_contractions)
    )
    return ConsolidationPlan(tuple(sorted(alive)), contractions,
                             tuple(BranchConsolidation(p, tuple(ids)) for p, ids in sorted(sides.items())),
                             tuple(triggers))


def apply_consolidation(tree: TreePopulation, plan: ConsolidationPlan) -> TreePopulation:
    """Reconnect the retained versions."""
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
    return TreePopulation(
        root_id=tree.root_id, root_selection_experience=tree.root_selection_experience,
        nodes=nodes, edges=edges, latest_node_id=tree.latest_node_id,
        direction_evaluations=dict(tree.direction_evaluations),
        used_crossover_pairs=[pair for pair in tree.used_crossover_pairs
                              if pair[0] in kept and pair[1] in kept],
        evaluations_used=tree.evaluations_used,
    )

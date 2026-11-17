"""覆盖代表集与树简化区域划分：路径收缩与子树剪枝。"""

import ast
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib

from .models import ROOT_ID, TreePopulation


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


@lru_cache(maxsize=4096)
def ast_shingles(code: str, k: int = 2) -> tuple[tuple[object, int], ...]:
    """提取对局部变量名与数值常数不敏感的 AST 父子边与遍历二元组特征。"""
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
    """忽略排版、注释与文档字符串后的语法树哈希。"""
    try:
        text = ast.dump(_strip_docstrings(code), include_attributes=False)
    except (SyntaxError, ValueError):
        text = code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def code_distance(code_a: str, code_b: str) -> float:
    """两段代码的 AST 多重集 Jaccard 距离，取值 [0.0, 1.0]。"""
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
    max_representatives: int = 24
    cover_distance: float = 0.15
    region_size: int = 4

    def __post_init__(self) -> None:
        for name in ("max_representatives", "region_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if type(self.cover_distance) not in (int, float) or not (0.0 <= self.cover_distance <= 1.0):
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


Region = PathContraction | SubtreePruning


def distinct_individuals(tree: TreePopulation) -> set[int]:
    """标准化源码与实测分数均相同的可行个体，只保留最早节点。"""
    first: dict[tuple[str, float], int] = {}
    for i in sorted(tree.nodes):
        node = tree.nodes[i]
        if node.fitness is not None:
            first.setdefault((identity(node.code), node.fitness), i)
    return set(first.values())


def ranks(tree: TreePopulation, candidates) -> dict[int, float]:
    """候选集合内的 0 起适应度名次（越小越优，并列共享平均名次）。"""
    ordered = sorted(candidates, key=lambda i: (tree.nodes[i].fitness, i))
    result: dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and tree.nodes[ordered[end]].fitness == tree.nodes[ordered[start]].fitness:
            end += 1
        result.update((i, (start + end - 1) / 2) for i in ordered[start:end])
        start = end
    return result


def representatives(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> tuple[int, ...]:
    """按 (fitness, id) 贪心选取至多 max_representatives 个互不覆盖的代表个体。"""
    ordered = sorted(distinct_individuals(tree), key=lambda i: (tree.nodes[i].fitness, i))  # type: ignore[arg-type]
    selected: list[int] = []
    for i in ordered:
        code_i = tree.nodes[i].code
        if any(code_distance(code_i, tree.nodes[r].code) < config.cover_distance for r in selected):
            continue
        selected.append(i)
        if len(selected) >= config.max_representatives:
            break
    return tuple(selected)


def plan_simplification(tree: TreePopulation, config: TopologyConfig = TopologyConfig()) -> tuple[Region, ...]:
    """补齐代表集的最近公共祖先骨架，返回达到 region_size 的路径收缩与子树剪枝区域。"""
    reps = set(representatives(tree, config))
    children = defaultdict(list)
    for i in sorted(tree.edges):
        children[tree.edges[i].parent_id].append(i)
    used = set(reps)
    for i in sorted(tree.nodes, reverse=True):
        if i in used:
            used.add(tree.edges[i].parent_id)
    forks = {i for i in used if i != ROOT_ID and sum(c in used for c in children[i]) >= 2}
    skeleton = reps | forks

    chains: list[tuple[int, int, tuple[int, ...]]] = []
    chain_of: dict[int, int] = {}
    for child in sorted(skeleton):
        parent = tree.edges[child].parent_id
        mid: list[int] = []
        while parent != ROOT_ID and parent not in skeleton:
            mid.append(parent)
            parent = tree.edges[parent].parent_id
        if mid:
            mid.reverse()
            idx = len(chains)
            chains.append((parent, child, tuple(mid)))
            for m in mid:
                chain_of[m] = idx

    branches: dict[int, list[int]] = defaultdict(list)
    subtrees: dict[int, list[int]] = defaultdict(list)
    for i in sorted(tree.nodes.keys() - skeleton - chain_of.keys()):
        anc = tree.edges[i].parent_id
        while anc != ROOT_ID and anc not in skeleton and anc not in chain_of:
            anc = tree.edges[anc].parent_id
        if anc in chain_of:
            branches[chain_of[anc]].append(i)
        else:
            subtrees[anc].append(i)

    contractions = (
        PathContraction(p, c, rem, tuple(branches.get(idx, ())))
        for idx, (p, c, rem) in enumerate(chains)
        if len(rem) + len(branches.get(idx, ())) >= config.region_size
    )
    prunings = (
        SubtreePruning(p, tuple(ids))
        for p, ids in sorted(subtrees.items())
        if len(ids) >= config.region_size
    )
    return (*contractions, *prunings)


def apply_simplification(tree: TreePopulation, regions: Sequence[Region]) -> TreePopulation:
    """移除简化区域中的节点，重连收缩边并保留存活节点的评价投入计数。"""
    removed = {
        i for r in regions
        for i in ((*r.removed_ids, *r.branch_ids) if isinstance(r, PathContraction) else r.removed_ids)
    }
    kept = sorted(tree.nodes.keys() - removed)
    kept_set = set(kept)
    rewired = {r.child_id: r.parent_id for r in regions if isinstance(r, PathContraction)}
    return TreePopulation(
        root_experience=tree.root_experience,
        nodes={i: replace(tree.nodes[i]) for i in kept},
        edges={
            i: (replace(tree.edges[i], parent_id=rewired[i], operator=None, reference_id=None, contracted=True)
                if i in rewired else replace(tree.edges[i]))
            for i in kept
        },
        used_crossover_pairs=[pair for pair in tree.used_crossover_pairs if pair[0] in kept_set and pair[1] in kept_set],
    )

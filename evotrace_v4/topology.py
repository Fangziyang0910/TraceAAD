"""核心保留集选取、LCA 连接骨架与路径/合并子树区域划分。"""

import ast
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import io
import tokenize

from .models import ROOT_ID, TreePopulation


def _docstring(node: ast.AST) -> ast.Expr | None:
    """模块、类或函数体开头的文档字符串语句。"""
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            return first
    return None


def _strip_docstrings(code: str) -> ast.AST:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if _docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def bare_code(code: str) -> str:
    """去掉文档字符串与注释、其余保持原写法的源码；不能整行去掉时原样返回。

    不用 ast.unparse 重排：它把 min(x for x in y) 写成 min((x for x in y))，模型照抄时会漏写右括号。
    """
    text = code.replace("\r\n", "\n").replace("\r", "\n")
    try:
        lines: list[str | None] = list(text.split("\n"))
        for token in tokenize.generate_tokens(io.StringIO(text).readline):
            if token.type == tokenize.COMMENT:
                row, col = token.start
                kept = (lines[row - 1] or "")[:col].rstrip()
                lines[row - 1] = kept if kept.strip() else None
        for node in ast.walk(ast.parse(text)):
            if (doc := _docstring(node)) is None:
                continue
            start, end = doc.lineno - 1, (doc.end_lineno or doc.lineno) - 1
            head, tail = lines[start] or "", lines[end] or ""
            if head[:doc.col_offset].strip() or tail.encode()[doc.end_col_offset or 0:].decode().strip():
                return text.strip()
            lines[start:end + 1] = [head[:doc.col_offset] + "pass" if len(node.body) == 1 else None] + [None] * (end - start)
        bare = "\n".join(line for line in lines if line is not None).strip()
        ast.parse(bare)
        return bare
    except (SyntaxError, ValueError, tokenize.TokenError):
        return text.strip()


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


def feasible_individuals(tree: TreePopulation) -> tuple[int, ...]:
    """返回当前存活于树上的全部可行个体编号（保留各谱系位置上的精确副本）。"""
    return tuple(i for i in sorted(tree.nodes) if tree.nodes[i].fitness is not None)


def distinct_individuals(tree: TreePopulation) -> set[int]:
    """标准化源码与实测分数均相同的可行个体，仅保留最早入树节点作为核心集候选。"""
    first: dict[tuple[str, float], int] = {}
    for i in sorted(tree.nodes):
        node = tree.nodes[i]
        if node.fitness is not None:
            first.setdefault((identity(node.code), node.fitness), i)
    return set(first.values())


def ranks(tree: TreePopulation, candidates) -> dict[int, float]:
    """候选集合内从 0 起算的平均适应度名次（越小越优，并列组共享整数或半整数平均名次）。"""
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
    """按 (fitness, id) 贪心扫描去重可行集，选出至多 max_representatives 个互不覆盖的核心保留个体。"""
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


def plan_simplification(
    tree: TreePopulation,
    config: TopologyConfig = TopologyConfig(),
    *,
    reps: Sequence[int] | None = None,
) -> tuple[Region, ...]:
    """补齐核心保留集的 LCA 骨架，划分相邻骨架间的路径区与各骨架节点下的合并子树区，返回满块区域。"""
    reps_set = set(representatives(tree, config) if reps is None else reps)
    children = defaultdict(list)
    for i in sorted(tree.edges):
        children[tree.edges[i].parent_id].append(i)
    used = set(reps_set)
    for i in sorted(tree.nodes, reverse=True):
        if i in used:
            used.add(tree.edges[i].parent_id)
    forks = {i for i in used if i != ROOT_ID and sum(c in used for c in children[i]) >= 2}
    skeleton = reps_set | forks

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
    """移除简化区域中的节点，重连收缩边（含重连至虚拟根）并保留存活节点的开发计数不变。"""
    removed = {
        i for r in regions
        for i in ((*r.removed_ids, *r.branch_ids) if isinstance(r, PathContraction) else r.removed_ids)
    }
    kept = sorted(tree.nodes.keys() - removed)
    rewired = {r.child_id: r.parent_id for r in regions if isinstance(r, PathContraction)}
    return TreePopulation(
        nodes={i: replace(tree.nodes[i]) for i in kept},
        edges={
            i: (replace(tree.edges[i], parent_id=rewired[i], operator=None, reference_id=None, contracted=True)
                if i in rewired else replace(tree.edges[i]))
            for i in kept
        },
    )

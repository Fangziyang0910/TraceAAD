"""有界谱系森林拓扑归约：源码清洗、规范 AST 精确去重、就地升级与单度容量淘汰。"""

import ast
import hashlib
import io
import math
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from .models import ROOT_ID, Program, TreePopulation, clean_text

ReductionKind = Literal["upgrade", "evict_leaf", "retire_root", "evict_single_child"]


def better(left: float, right: float) -> bool:
    """严格最小化比较，允许末位浮点舍入误差视为持平。"""
    return right - left > 8 * max(math.ulp(left), math.ulp(right))


def beats(left: Program, right: Program) -> bool:
    """严格优劣判据：双方可行且 left 的目标值严格更优。"""
    return left.valid and right.valid and better(left.fitness, right.fitness)


def _docstring(node: ast.AST) -> ast.Expr | None:
    if (
        isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and node.body
    ):
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            return first
    return None


def bare_code(code: str) -> str:
    """去掉文档字符串与注释、其余保持原写法的源码；不能整行去掉时原样返回。"""
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
            if (
                head[: doc.col_offset].strip()
                or tail.encode()[doc.end_col_offset or 0 :].decode().strip()
            ):
                return text.strip()
            replacement = head[: doc.col_offset] + "pass" if len(node.body) == 1 else None
            lines[start : end + 1] = [replacement] + [None] * (end - start)
        bare = "\n".join(line for line in lines if line is not None).strip()
        ast.parse(bare)
        return bare
    except SyntaxError, ValueError, tokenize.TokenError:
        return text.strip()


@lru_cache(maxsize=4096)
def code_identity(code: str) -> str:
    """忽略排版、注释与文档字符串后的规范 AST 哈希，用于精确源码去重。"""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if _docstring(node) is not None:
                node.body = node.body[1:] or [ast.Pass()]
        text = ast.dump(tree, include_attributes=False)
    except (SyntaxError, ValueError):
        text = code.strip()
    return hashlib.sha256(text.encode()).hexdigest()


def duplicate_of(program: Program, active_programs) -> int | None:
    """检查新程序是否与存活树上的某个节点规范 AST 完全一致。"""
    if not program.valid:
        return None
    key = code_identity(program.code)
    return next(
        (
            other.id
            for other in active_programs
            if other.id != program.id and other.valid and code_identity(other.code) == key
        ),
        None,
    )


@dataclass(frozen=True)
class Reduction:
    kind: ReductionKind
    anchor: int
    target: int
    removed: tuple[int, ...]


def next_reduction(
    tree: TreePopulation, archive: dict[int, Program], capacity: int = 24
) -> Reduction | None:
    """查找下一步确定性树归约：
    1. 就地升级（Deterministic Crowding）：无分叉单链 A -> v_1..v_k -> D（允许 A = v_0）满足 D 严格优于所有中间节点；
    2. 超容单度淘汰（|V| > capacity）：在非全局最优且出度 <= 1 的节点中移出全局目标值最差者（同分取较晚编号）。
    """
    if not tree.parents:
        return None

    children: dict[int, list[int]] = defaultdict(list)
    for child, parent in tree.parents.items():
        children[parent].append(child)

    # 后序遍历（叶端优先）保证确定性扫描顺序
    order: list[int] = []
    stack = sorted(children[ROOT_ID], reverse=True)
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(sorted(children[node], reverse=True))
    downstream_first = order[::-1]

    # 步骤 1：就地升级（允许收缩到虚拟根 ROOT_ID）
    for end in downstream_first:
        middle: list[int] = []
        current = tree.parents[end]
        while (
            current != ROOT_ID
            and len(children[current]) == 1
            and beats(archive[end], archive[current])
        ):
            middle.append(current)
            current = tree.parents[current]
        if middle:
            return Reduction("upgrade", current, end, tuple(reversed(middle)))

    # 步骤 2：超容单度淘汰（由两行度之和定理保证 |V| >= 2 时必然存在合法淘汰对象）
    if len(tree.parents) <= capacity:
        return None

    champion = min(tree.parents, key=lambda nid: (archive[nid].fitness, nid))
    eligible = [nid for nid in tree.parents if nid != champion and len(children[nid]) <= 1]
    if not eligible:
        return None
    worst = max(eligible, key=lambda nid: (archive[nid].fitness, nid))
    parent = tree.parents[worst]
    kids = children[worst]
    if not kids:
        if parent == ROOT_ID:
            return Reduction("retire_root", ROOT_ID, ROOT_ID, (worst,))
        return Reduction("evict_leaf", parent, parent, (worst,))
    return Reduction("evict_single_child", parent, kids[0], (worst,))


def apply_reduction(tree: TreePopulation, event: Reduction, note: str) -> None:
    """应用拓扑重连、移除淘汰节点，并将更新后的未兑现想法笔记写回目标节点。"""
    if event.kind in ("upgrade", "evict_single_child"):
        tree.parents[event.target] = event.anchor
    for old in event.removed:
        tree.parents.pop(old, None)
        tree.idea_notes.pop(old, None)
        tree.direct_evals.pop(old, None)
    if event.kind == "retire_root":
        return
    cleaned = clean_text(note)
    if cleaned:
        tree.idea_notes[event.target] = cleaned
    else:
        tree.idea_notes.pop(event.target, None)

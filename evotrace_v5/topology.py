"""纯黑盒树状种群局部归约：源码清洗/去重，以及无状态的分支剪枝与路径收缩。"""

import ast
import hashlib
import io
import math
import tokenize
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .models import ROOT_ID, Program, TreePopulation, clean_text


def better(left: float, right: float) -> bool:
    """严格最小化比较，允许末位浮点舍入误差视为持平。"""
    return right - left > 8 * max(math.ulp(left), math.ulp(right))


def beats(left: Program, right: Program) -> bool:
    """纯黑盒胜负判据：双方可行且 left 的目标值严格更优。"""
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
    """忽略排版、注释与文档字符串后的语法树哈希，用于精确去重。"""
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
    """检查新程序是否与现役树上的某个节点源码等价（忽略注释、文档字符串与排版）。"""
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
    kind: str
    anchor: int
    target: int
    removed: tuple[int, ...]


def next_reduction(tree: TreePopulation, archive: dict[int, Program]) -> Reduction | None:
    """无状态、叶端优先查找下一次局部归约：先分支剪枝（至少两弱叶），再路径收缩。"""
    children: dict[int, list[int]] = defaultdict(list)
    for child, parent in tree.parents.items():
        children[parent].append(child)

    order: list[int] = []
    stack = sorted(children[ROOT_ID], reverse=True)
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(sorted(children[node], reverse=True))
    downstream_first = order[::-1]

    for parent in downstream_first:
        weak = sorted(
            i for i in children[parent] if not children[i] and beats(archive[parent], archive[i])
        )
        if len(weak) >= 2:
            return Reduction("branch", parent, parent, tuple(weak))

    for end in downstream_first:
        middle: list[int] = []
        current = tree.parents[end]
        while (
            current != ROOT_ID
            and tree.parents[current] != ROOT_ID
            and len(children[current]) == 1
            and beats(archive[end], archive[current])
        ):
            middle.append(current)
            current = tree.parents[current]
        if middle:
            return Reduction("path", current, end, tuple(reversed(middle)))
    return None


def _rewire(parents: dict[int, int], event: Reduction) -> None:
    if event.kind == "path":
        parents[event.target] = event.anchor
    for old in event.removed:
        del parents[old]


def reduction_plan(tree: TreePopulation, archive: dict[int, Program]) -> list[Reduction]:
    """在当前树上迭代执行无状态局部归约直至不动点，返回有序归约事件列表。"""
    view = TreePopulation(parents=tree.parents.copy())
    events: list[Reduction] = []
    while (event := next_reduction(view, archive)) is not None:
        events.append(event)
        _rewire(view.parents, event)
    return events


def apply_reduction(tree: TreePopulation, event: Reduction, note: str) -> None:
    """应用拓扑重连并把凝练后的经验写回对应节点或收缩边。"""
    _rewire(tree.parents, event)
    cleaned = clean_text(note)
    target_map = tree.node_notes if event.kind == "branch" else tree.edge_notes
    if cleaned:
        target_map[event.target] = cleaned
    else:
        target_map.pop(event.target, None)
    for old in event.removed:
        tree.edge_notes.pop(old, None)
        tree.node_notes.pop(old, None)

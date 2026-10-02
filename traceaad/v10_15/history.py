"""Observed changes on a node's formation path; Ideas remain intentions."""

import ast
import difflib


def path(node, archive):
    items = [node]
    seen = {node["id"]}
    while items[-1]["parent_id"] is not None:
        parent = archive[items[-1]["parent_id"]]
        if parent["id"] in seen:
            raise ValueError("cyclic formation path")
        items.append(parent)
        seen.add(parent["id"])
    return list(reversed(items))


def score_text(score):
    return format(score, ".6g")


def verdict(parent, child, higher_is_better):
    delta = child - parent if higher_is_better else parent - child
    tolerance = 1e-9 * max(1.0, abs(parent))
    return "improved" if delta > tolerance else "worse" if delta < -tolerance else "same score"


class _MaskNumeric(ast.NodeTransformer):
    def visit_UnaryOp(self, node):
        if (isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in (int, float, complex)):
            return ast.copy_location(ast.Constant(value=0), node)
        return self.generic_visit(node)

    def visit_Constant(self, node):
        if type(node.value) in (int, float, complex):
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _numbers(code):
    tree = ast.parse(code)
    locations = []

    def visit(node, owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and owner == "<module>":
            owner = node.name
        if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
                and isinstance(node.operand, ast.Constant)
                and type(node.operand.value) in (int, float, complex)):
            value = node.operand.value * (-1 if isinstance(node.op, ast.USub) else 1)
            locations.append((node.lineno, node.col_offset, owner, repr(value)))
            return
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
            locations.append((node.lineno, node.col_offset, owner, repr(node.value)))
        for child in ast.iter_child_nodes(node):
            visit(child, owner)

    visit(tree, "<module>")
    return sorted(locations)


def numeric_changes(before, after):
    left, right = ast.parse(before), ast.parse(after)
    if ast.dump(_MaskNumeric().visit(left)) != ast.dump(_MaskNumeric().visit(right)):
        return None
    old, new = _numbers(before), _numbers(after)
    if len(old) != len(new):
        return None
    return [(a[2], a[3], b[3]) for a, b in zip(old, new) if a[3] != b[3]]


def _changed_top_level(before, after):
    def units(code):
        result = {}
        for node in ast.parse(code).body:
            label = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else "<module>"
            result.setdefault(label, []).append(ast.dump(node))
        return result

    left, right = units(before), units(after)
    return sorted(name for name in left.keys() | right.keys() if left.get(name) != right.get(name))[:4]


def change_summary(before, after):
    numbers = numeric_changes(before, after)
    if numbers:
        names = ", ".join(dict.fromkeys(name for name, _, _ in numbers))
        pairs = "; ".join(f"{old} → {new}" for _, old, new in numbers[:4])
        return f"numeric constants only, in {names}: {pairs}"[:400]
    old, new = before.splitlines(), after.splitlines()
    removed, added = [], []
    for op, i, j, k, end in difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes():
        if op != "equal":
            removed.extend(old[i:j])
            added.extend(new[k:end])
    names = ", ".join(_changed_top_level(before, after)) or "<module>"

    def ends(lines):
        if not lines:
            return "none"
        clean = [line.strip()[:110] for line in lines]
        return f"`{clean[0]}` | `{clean[-1]}`"

    return (f"+{len(added)}/−{len(removed)} lines in {names}; "
            f"removed: {ends(removed)}; added: {ends(added)}")[:400]


def code_diff(before, after):
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(), n=2, lineterm=""))[2:]
    return "\n".join(lines)


def edges(node, archive, count=8):
    sequence = path(node, archive)
    return sequence[0], [(sequence[i - 1], sequence[i]) for i in range(max(1, len(sequence) - count), len(sequence))]

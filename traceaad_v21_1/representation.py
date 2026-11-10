"""Normalize displayed source and describe actual parent-child code changes."""

import ast
from difflib import SequenceMatcher, unified_diff
from functools import lru_cache

REWRITE_BELOW = .5


def _tree(code):
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return tree


@lru_cache(maxsize=None)
def bare(code):
    """The source without comments or docstrings; the record keeps the raw text."""
    try:
        return ast.unparse(_tree(code))
    except (SyntaxError, ValueError):
        return code


def changes(before, after):
    """The changed lines of normalized source, or the fact that most of the code was replaced."""
    try:
        old, new = ast.unparse(_tree(before)).splitlines(), ast.unparse(_tree(after)).splitlines()
    except (SyntaxError, ValueError):
        return None
    matcher = SequenceMatcher(None, old, new, autojunk=False)
    if matcher.ratio() < REWRITE_BELOW:
        kept = sum(block.size for block in matcher.get_matching_blocks())
        return f"Changed computation: most of the code was replaced; {kept} of {len(new)} lines are unchanged."
    rows = []
    for line in list(unified_diff(old, new, lineterm="", n=1))[2:]:
        rows.append("..." if line.startswith("@@") else line)
    return "Changed computation:\n" + "\n".join(rows[1:])

"""Normalized source for display; a step's actual change as one unified diff of the normalized parent
and child source; and where a failed program stopped, as the candidate's own source line."""

import ast
from difflib import unified_diff
from functools import lru_cache

from traceaad_base.prompts import TIMEOUT_PREFIX, _parse_failure
from traceaad_base.selection import identity

CONTEXT_LINES = 3


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


def _normalized(code):
    return ast.unparse(_tree(code)).splitlines()


def changes(parent, program):
    """The changed lines with three lines of context, whatever the size of the change."""
    if identity(parent.code) == identity(program.code):
        return "Code change: none; the normalized source is unchanged."
    try:
        old, new = _normalized(parent.code), _normalized(program.code)
        header = "Code change:"
    except (SyntaxError, ValueError):
        old, new = parent.code.splitlines(), program.code.splitlines()
        header = ("Code change: the candidate code could not be parsed; "
                  f"the raw source texts are compared.")
    rows = ["..." if line.startswith("@@") else line
            for line in list(unified_diff(old, new, lineterm="", n=CONTEXT_LINES))[2:]]
    return header + "\n" + "\n".join(rows[1:])


def failure(evaluation):
    """The execution status, the exception, and the candidate's own line where it was raised."""
    text = evaluation.failure or ""
    if text.startswith(TIMEOUT_PREFIX):
        return f"Result: {evaluation.status}; the evaluation did not finish within the allowed time."
    message, blocks = _parse_failure(text)
    lines = [f"Result: {evaluation.status}; {message}"]
    if blocks:
        frame = blocks[-1]
        function = frame[0].rsplit(", in ", 1)[-1] if ", in " in frame[0] else None
        source = next((line.strip() for line in frame[1:] if line.strip() and set(line.strip()) - set("~^ ")), None)
        if source:
            lines.append(f"Raised in {function}: {source}" if function else f"Raised at: {source}")
    return "\n".join(lines)

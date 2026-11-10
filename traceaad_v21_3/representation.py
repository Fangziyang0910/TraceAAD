"""Every step shows its actual change: one unified diff of the normalized parent and child source."""

import ast
from difflib import unified_diff

from traceaad_v21_1.representation import _tree
from traceaad_base.selection import identity

CONTEXT_LINES = 3


def _normalized(code):
    return ast.unparse(_tree(code)).splitlines()


def changes(parent, program):
    """The changed lines with three lines of context, whatever the size of the change."""
    if identity(parent.code) == identity(program.code):
        return f"Changed computation: none; the code is identical to Algorithm {parent.id}."
    try:
        old, new = _normalized(parent.code), _normalized(program.code)
        header = "Changed computation:"
    except (SyntaxError, ValueError):
        old, new = parent.code.splitlines(), program.code.splitlines()
        header = (f"Changed computation: the code of Algorithm {program.id} could not be parsed; "
                  f"the raw source texts are compared.")
    rows = ["..." if line.startswith("@@") else line
            for line in list(unified_diff(old, new, lineterm="", n=CONTEXT_LINES))[2:]]
    return header + "\n" + "\n".join(rows[1:])

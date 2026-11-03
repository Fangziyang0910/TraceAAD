"""Code identity and code distance.

Identity ignores comments, docstrings and layout, so a reply that reproduces an
existing program with new comments is recognized. Distance is 1 minus the
Jaccard similarity of 5-token shingles, with every number read as one token;
it measures how much code two programs share, not how differently they behave.
"""

import ast
import hashlib
import io
import tokenize

SHINGLE = 5


def canonical(code):
    """The program without comments, docstrings and layout; None if it does not parse."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def identity(code):
    text = canonical(code)
    return None if text is None else hashlib.sha256(text.encode()).hexdigest()


def shingles(code):
    text = canonical(code) or code
    try:
        tokens = [
            "NUM" if t.type == tokenize.NUMBER else t.string
            for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type
            not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            )
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        tokens = text.split()
    return {tuple(tokens[i : i + SHINGLE]) for i in range(max(1, len(tokens) - SHINGLE + 1))}


def distance(a, b):
    shared = len(a & b)
    union = len(a) + len(b) - shared
    return 1.0 - shared / union if union else 0.0

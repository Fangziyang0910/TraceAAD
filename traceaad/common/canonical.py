"""Program identity and the lexical reference-distance proxy."""

import ast
import hashlib
import io
import tokenize


class _StripDocstrings(ast.NodeTransformer):
    def _body(self, node):
        node = self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        if not node.body and not isinstance(node, ast.Module):
            node.body = [ast.Pass()]
        return node

    def visit_Module(self, node):
        return self._body(node)

    def visit_FunctionDef(self, node):
        return self._body(node)

    def visit_AsyncFunctionDef(self, node):
        return self._body(node)

    def visit_ClassDef(self, node):
        return self._body(node)


def canonical(code: str) -> str:
    tree = _StripDocstrings().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree).rstrip() + "\n"


def key(code: str) -> str:
    """The caller passes canonical source, avoiding two normalization paths."""
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def token_set(code: str) -> frozenset[str]:
    return frozenset(token.string for token in tokenize.generate_tokens(io.StringIO(code).readline)
                     if token.type in (tokenize.NAME, tokenize.NUMBER, tokenize.OP))


def similarity(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / len(a | b) if a or b else 1.0

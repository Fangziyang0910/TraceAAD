"""Exact, simultaneous edits; source identity is never semantic similarity."""

import ast
from dataclasses import asdict, dataclass
import difflib
import hashlib
import json
import re

from traceaad.v10_13.parsing import template_target


def source_id(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Edit:
    search: str
    replacement: str


class SourceError(ValueError):
    """A complete payload that failed syntax/interface checks, retained for repair."""

    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


def apply_edits(code, edits):
    """Validate all preimages against one base, reject overlap, commit atomically."""
    if not edits:
        raise ValueError("empty edit bundle")
    spans = []
    for edit in edits:
        if not isinstance(edit.search, str) or not edit.search or not isinstance(edit.replacement, str):
            raise ValueError("edits require nonempty search and string replacement")
        if code.count(edit.search) != 1:
            raise ValueError("edit search must match exactly once")
        start = code.index(edit.search)
        spans.append((start, start + len(edit.search), edit.replacement))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("overlapping edits")
    for start, end, replacement in reversed(spans):
        code = code[:start] + replacement + code[end:]
    return code


def inverse(edits):
    return tuple(Edit(e.replacement, e.search) for e in edits)


def extract_edits(before, after):
    """Conservative line bundles, adding context only for insertions/deletions.

    Ambiguity and overlapping context deliberately make a bundle ineligible.
    Full output remains a normal search candidate even when this fails.
    """
    left, right = before.splitlines(keepends=True), after.splitlines(keepends=True)
    edits = []
    for tag, i, j, k, end in difflib.SequenceMatcher(a=left, b=right, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if i == j or k == end:
            if i > 0 and k > 0 and left[i-1] == right[k-1]:
                i, k = i-1, k-1
            elif j < len(left) and end < len(right) and left[j] == right[end]:
                j, end = j+1, end+1
            else:
                raise ValueError("insertion/deletion lacks stable context")
        edits.append(Edit("".join(left[i:j]), "".join(right[k:end])))
    edits = tuple(edits)
    if apply_edits(before, edits) != after or apply_edits(after, inverse(edits)) != before:
        raise ValueError("bundle does not round-trip")
    return edits


def close_trace(p00, p10, p11):
    a, b = extract_edits(p00, p10), extract_edits(p10, p11)
    p01 = apply_edits(p11, inverse(a))
    if apply_edits(p00, b) != p01 or apply_edits(p01, a) != p11:
        raise ValueError("edits cannot be exchanged exactly")
    if apply_edits(p01, inverse(b)) != p00:
        raise ValueError("background does not round-trip")
    return p01, {"a": [asdict(e) for e in a], "b": [asdict(e) for e in b],
                 "snapshots": [source_id(p) for p in (p00, p10, p11, p01)]}


def symbols(code):
    try:
        return sorted({n.name for n in ast.walk(ast.parse(code))
                       if isinstance(n, (ast.FunctionDef, ast.ClassDef))})
    except SyntaxError:
        return []


def changed_symbols(before, after):
    def units(code):
        result = {}
        for node in ast.parse(code).body:
            name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else "<module>"
            result.setdefault(name, []).append(ast.dump(node))
        return result
    try:
        left, right = units(before), units(after)
        return sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    except SyntaxError:
        return symbols(after)


def numeric_parameters(code):
    """Only explicit numeric assignments qualify Tune, not incidental literals."""
    def numeric(node):
        return (isinstance(node, ast.Constant) and type(node.value) in (int, float)
                or isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
                and numeric(node.operand))
    names = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and numeric(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.extend(t.id for t in targets if isinstance(t, ast.Name))
    return sorted(set(names))


def observed_change(before, after):
    """A syntactic diagnostic, deliberately not a semantic operator classifier."""
    if before == after:
        return "identical_source"
    try:
        left, right = ast.parse(before), ast.parse(after)
        if ast.dump(left) == ast.dump(right):
            return "presentation_only"

        class MaskNumbers(ast.NodeTransformer):
            def visit_Constant(self, node):
                return ast.Constant(value=0) if type(node.value) in (int, float) else node

        if ast.dump(MaskNumbers().visit(left)) == ast.dump(MaskNumbers().visit(right)):
            return "numeric_literals_only"
    except SyntaxError:
        return "invalid_syntax"
    return "structural_syntax_change"


def validate_source(code, template):
    tree = ast.parse(code)
    interface, _ = template_target(template)
    name, _, signature = interface
    targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(targets) != 1:
        raise ValueError(f"expected one top-level function {name}")
    args = targets[0].args
    actual = ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
              [a.arg for a in args.kwonlyargs], args.vararg.arg if args.vararg else None,
              args.kwarg.arg if args.kwarg else None)
    if actual != signature:
        raise ValueError("target signature differs from the task interface")
    compile(code, "<candidate>", "exec")


def parse_response(text, finish_reason, template, *, base=None, mode="full"):
    """Return (exact evaluated source, optional note); refuse ambiguous delivery."""
    if finish_reason not in {None, "unknown", "stop"}:
        raise ValueError(f"incomplete completion: {finish_reason}")
    if not isinstance(text, str):
        raise ValueError("response must be text")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "<think>" in text.lower():
        raise ValueError("unclosed reasoning block")
    note = re.search(r"(?im)^\s*(?:#+\s*)?Idea\s*:\s*(.+)", text)
    idea = note.group(1).strip()[:1000] if note else ""
    if mode == "edit" and base is not None:
        try:
            payload = json.loads(text)
            if payload.get("mode") != "edit":
                raise ValueError("expected the configured edit payload")
            edits = tuple(Edit(**e) for e in payload["edits"])
            code = apply_edits(base, edits)
            idea = str(payload.get("idea", ""))[:1000]
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("invalid edit payload") from exc
    else:
        if "```" in text:
            blocks = re.findall(r"```(?:python|py)\s*\n(.*?)\n```", text, re.S | re.I)
            if len(blocks) != 1 or text.count("```") != 2:
                raise ValueError("expected exactly one closed Python block")
            code = blocks[0]
        else:
            code = text
        # Preserve the submitted source; only add absent task imports explicitly.
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError) as exc:
            raise SourceError(str(exc), code) from exc
        existing = {ast.dump(n) for n in tree.body}
        imports = [ast.get_source_segment(template, n) for n in ast.parse(template).body
                   if isinstance(n, (ast.Import, ast.ImportFrom)) and ast.dump(n) not in existing]
        if imports:
            # Future imports must precede ordinary imports; preserve all source
            # bytes apart from the explicitly logged template import insertion.
            prefix_end = 0
            for node in tree.body:
                if (isinstance(node, ast.ImportFrom) and node.module == "__future__"
                        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)):
                    prefix_end = node.end_lineno
                else:
                    break
            lines = code.splitlines(keepends=True)
            prefix = "".join(lines[:prefix_end])
            code = prefix + ("\n" if prefix and not prefix.endswith("\n") else "") + "\n".join(imports) + "\n" + "".join(lines[prefix_end:])
    try:
        validate_source(code, template)
    except (SyntaxError, ValueError) as exc:
        raise SourceError(str(exc), code) from exc
    return code, idea

"""Extract and validate one Python implementation from a model response."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
PYTHON_BLOCK_RE = re.compile(
    r"```(?:python|py)\s*\n(.*?)(?:\n```|\Z)", re.DOTALL | re.IGNORECASE,
)
FULL_OUTPUT_FORMAT = """Return the complete implementation in one Python code block.
You may precede it with one short Idea sentence."""
OUTPUT_FORMAT = FULL_OUTPUT_FORMAT + """ For a small change to the shown program,
you may instead return exactly one JSON object:
{"mode":"edit","edits":[{"search":"exact source","replacement":"new source"}]}"""


@dataclass
class ParsedCandidate:
    idea: str
    program_code: str


def _signature(args):
    return (
        [arg.arg for arg in args.posonlyargs],
        [arg.arg for arg in args.args],
        [arg.arg for arg in args.kwonlyargs],
        args.vararg.arg if args.vararg else None,
        args.kwarg.arg if args.kwarg else None,
    )


def template_target(template_program):
    """Return the template function interface and a compact prompt stub."""
    tree = ast.parse(template_program)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        raise ValueError("evaluation template must define exactly one function")
    function = functions[0]
    args_text = ast.unparse(function.args)
    returns = f" -> {ast.unparse(function.returns)}" if function.returns else ""
    stub = f"def {function.name}({args_text}){returns}:\n    pass"
    return (function.name, args_text, _signature(function.args)), stub


def _apply_edits(code, edits):
    if not isinstance(edits, list) or not edits:
        raise ValueError("edits must be a nonempty list")
    for edit in edits:
        if not isinstance(edit, dict):
            raise ValueError("each edit must be an object")
        search = edit.get("search")
        replacement = edit.get("replacement")
        if not isinstance(search, str) or not search or not isinstance(replacement, str):
            raise ValueError("each edit needs nonempty search text and replacement text")
        if code.count(search) != 1:
            raise ValueError("each search must match exactly once")
        code = code.replace(search, replacement, 1)
    return code


def _idea(text):
    match = re.search(r"(?:^|\n)\s*(?:#+\s*)?Idea\s*:\s*(.+)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _merge_with_template(template_program, generated_tree, target):
    """Keep task imports/helpers and replace its target implementation."""
    template_tree = ast.parse(template_program)
    template_target = next(
        node for node in template_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    template_body = [node for node in template_tree.body if node is not template_target]
    generated_body = [node for node in generated_tree.body if node is not target]
    module = ast.Module(body=template_body + generated_body + [target], type_ignores=[])
    ast.fix_missing_locations(module)
    return ast.unparse(module).strip()


def parse_candidate(response, finish_reason, interface, template_program, *, base_code=None):
    """Parse a full Python response or one exact edit of the parent program."""
    del finish_reason
    text = THINK_BLOCK_RE.sub("", response).strip()
    idea = _idea(text)
    mode = "full"

    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        payload = None

    if isinstance(payload, dict):
        mode = payload.get("mode", "full")
        if isinstance(payload.get("idea"), str):
            idea = payload["idea"].strip()
        if mode == "edit":
            if base_code is None:
                return None, "edit_error: no parent program is available"
            try:
                code = _apply_edits(base_code, payload.get("edits"))
            except ValueError as exc:
                return None, f"edit_error: {exc}"
        elif mode == "full" and isinstance(payload.get("code"), str):
            code = payload["code"]
        else:
            return None, "format_error: expected Python code or one edit object"
    else:
        block = PYTHON_BLOCK_RE.search(text)
        code = block.group(1) if block else text

    code = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not code:
        return None, "format_error: empty code"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        return None, f"syntax_error: {exc}"

    name, args_text, expected_signature = interface
    targets = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(targets) != 1:
        return None, f"target_error: expected one top-level {name} function"
    if _signature(targets[0].args) != expected_signature:
        return None, f"signature_error: {name} must use ({args_text})"

    program = ast.unparse(tree).strip() if mode == "edit" else _merge_with_template(
        template_program, tree, targets[0],
    )
    try:
        compile(program, "<candidate>", "exec")
    except (SyntaxError, ValueError) as exc:
        return None, f"syntax_error: {exc}"
    return ParsedCandidate(idea, program), None

"""Idea parsing and target-function extraction for TraceAAD V10.12."""

import ast
import re

from dataclasses import dataclass

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*\r?$", re.MULTILINE | re.IGNORECASE)
OUTPUT = (
    "Return exactly:\n"
    "Idea: <one short paragraph describing how the algorithm computes its output and the main change from any supplied algorithms>\n"
    "Code:\n```python\n"
    "<the complete Python implementation, including the target function "
    "and any supporting code>\n```\n\n"
    "Keep the Idea to no more than 200 words.\n"
    "Use the exact target function name and signature shown above. "
    "Include imports and helper definitions as needed."
)
MAX_IDEA_CHARS = 2000


@dataclass
class ParsedCandidate:
    idea: str
    program_code: str


def normalize_code(text: str) -> str:
    return "\n".join(text.replace("\r\n", "\n").replace("\r", "\n").splitlines()).strip()


def signature(args):
    return ([arg.arg for arg in args.posonlyargs], [arg.arg for arg in args.args],
            [arg.arg for arg in args.kwonlyargs],
            args.vararg.arg if args.vararg else None,
            args.kwarg.arg if args.kwarg else None)


def template_target(template_program):
    try:
        tree = ast.parse(template_program)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("evaluation template must define exactly one function") from exc
    funcs = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) != 1:
        raise ValueError("evaluation template must define exactly one function")
    func = funcs[0]
    args_text = ast.unparse(func.args)
    returns = f" -> {ast.unparse(func.returns)}" if func.returns else ""
    lines = [f"def {func.name}({args_text}){returns}:"]
    docstring = ast.get_docstring(func)
    if docstring:
        lines.append(f'    """{docstring}"""')
    lines.append("    pass")
    return (func.name, args_text, signature(func.args)), "\n".join(lines)


def _target_functions(tree, name):
    return [node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]


def _merge_candidate(template_program, generated_tree, target):
    """Keep the fixed template preface and add generated module dependencies."""
    template_tree = ast.parse(template_program)
    template_targets = _target_functions(template_tree, target.name)
    if len(template_targets) != 1:
        return None
    template_fn = template_targets[0]
    preface = [node for node in template_tree.body if node is not template_fn]
    seen = {ast.dump(node, include_attributes=False) for node in preface}
    dependencies = []
    for node in generated_tree.body:
        if node is target:
            continue
        key = ast.dump(node, include_attributes=False)
        if key not in seen:
            dependencies.append(node)
            seen.add(key)
    module = ast.Module(body=preface + dependencies + [target], type_ignores=[])
    ast.fix_missing_locations(module)
    return normalize_code(ast.unparse(module))


def parse_candidate(response, finish_reason, interface, template_program):
    if finish_reason not in ("stop", "length", "unknown"):
        return None, "unsupported finish reason"
    text = THINK_BLOCK_RE.sub("", response)
    fences = list(FENCE_RE.finditer(text))
    if len(fences) != 2:
        return None, "format_error: expected exactly one Python code block"
    prefix = text[:fences[0].start()]
    if text[fences[1].end():].strip():
        return None, "format_error: no text is allowed after the code block"
    idea_match = re.fullmatch(r"[ \t]*Idea:[ \t]*(.*?)[ \t]*\n[ \t]*Code:[ \t]*\n?", prefix,
                              re.IGNORECASE | re.DOTALL)
    if idea_match is None or not idea_match.group(1).strip():
        return None, "Idea is required"
    idea = idea_match.group(1).strip()
    if len(idea) > MAX_IDEA_CHARS:
        return None, f"idea_length_error: Idea exceeds {MAX_IDEA_CHARS} characters"
    code = text[fences[0].end():fences[1].start()]
    canonical = normalize_code(code)
    if not canonical:
        return None, "The code block is empty"
    try:
        tree = ast.parse(canonical)
    except (SyntaxError, ValueError) as exc:
        return None, f"syntax_error: {type(exc).__name__}: {exc}"
    name, args_text, expected = interface
    targets = _target_functions(tree, name)
    if len(targets) != 1:
        return None, f"target_function_error: expected exactly one top-level `{name}(...)` function"
    if signature(targets[0].args) != expected:
        return None, f"signature_error: `{name}` must declare the parameters ({args_text})"
    rebuilt = _merge_candidate(template_program, tree, targets[0])
    if rebuilt is None:
        return None, "template_error: could not find one target function in the task template"
    return ParsedCandidate(idea, rebuilt), None


def repair_prompt(task_contract, response, event):
    message = (event.get("error") or event.get("reason") or "Evaluation failed").strip()
    error_type = event.get("error_type") or "Error"
    return (f"{task_contract}\n\n# Failed output\n{THINK_BLOCK_RE.sub('', response)}\n\n"
            f"# Failure\n{error_type}: {message[:2000]}\n\n"
            "# Repair\nCorrect the failure while preserving the intended decision method, then return the required output format.\n\n"
            + OUTPUT)

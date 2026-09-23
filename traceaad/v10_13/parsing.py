"""Code-first full/edit parsing with optional idea metadata."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass, field

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"^[ \t]*```(?:python|py)?[ \t]*\r?$", re.MULTILINE | re.IGNORECASE)
FULL_OUTPUT_FORMAT = '''Return one JSON object. Focus on implementing a good algorithm.
Full implementation: {"mode":"full","code":"complete Python source"}.
You may add an optional top-level idea string or object:
"idea": {"mechanism":"main decision principle", "change":"what changed",
"transfer":"a potentially reusable computation"}. Keep each included field to one
short sentence; omit fields that add no useful information.
You may add donor_id: the offered reference actually used, or null. These descriptions
are hypotheses, not proofs. Preserve the exact target function interface.'''
OUTPUT_FORMAT = FULL_OUTPUT_FORMAT + '''
Local edit: {"mode":"edit","base_hash":"shown parent hash",
"edits":[{"search":"exact nonempty source block","replacement":"replacement block"}]}.
Use either full or edit, not both. Supply 1 to 32 edits, applied sequentially to the
shown parent; each search block must match exactly once, including whitespace.
A full implementation is always allowed. Optional idea and donor_id also apply to edit mode.'''


def code_hash(code):
    return hashlib.sha256(code.encode()).hexdigest()


@dataclass
class ParsedCandidate:
    idea: str
    program_code: str
    idea_fields: dict[str, str] = field(default_factory=dict)
    mode: str = "full"
    donor_id: int | None = None
    base_hash: str | None = None


def response_object(response):
    text = THINK_BLOCK_RE.sub('', response).strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def idea_metadata(value):
    # Metadata never gates evaluation; the unabridged response is journaled.
    if isinstance(value, str):
        return value.strip(), {}
    if isinstance(value, dict):
        fields = {key: value[key].strip() for key in ('mechanism', 'change', 'transfer')
                  if isinstance(value.get(key), str) and value[key].strip()}
        return fields.get('mechanism') or fields.get('change', ''), fields
    return '', {}


def apply_edits(base_code, expected_hash, edits):
    if expected_hash != code_hash(base_code):
        raise ValueError('base_hash does not match the supplied parent')
    if not isinstance(edits, list) or not edits or len(edits) > 32:
        raise ValueError('edits must contain 1 to 32 replacements')
    code = base_code
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {'search', 'replacement'}:
            raise ValueError('each edit needs only search and replacement')
        source, target = edit['search'], edit['replacement']
        if not isinstance(source, str) or not source or not isinstance(target, str):
            raise ValueError('search must be nonempty text and replacement must be text')
        if code.count(source) != 1:
            raise ValueError('search block must match exactly once; use more surrounding code')
        code = code.replace(source, target, 1)
    return code


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


def _extract_idea(prefix: str) -> str | None:
    lines = prefix.replace("\r\n", "\n").splitlines()
    cleaned = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        label = re.match(r"^(?:[*_`#\s]*)(idea|code)\s*:?[\s*`]*$", stripped, re.I)
        inline = re.match(r"^(?:[*_`#\s]*)idea\s*:\s*(.*?)\s*$", stripped, re.I)
        if inline:
            if inline.group(1).strip():
                cleaned.append(re.sub(r"^[*_`#\s]+", "", inline.group(1)).strip())
            collecting = True
            continue
        if label:
            collecting = label.group(1).lower() == "idea"
            continue
        if collecting and stripped:
            cleaned.append(stripped)
    if not cleaned:
        prose = [line.strip() for line in lines if line.strip() and
                 not re.match(r"^(?:[*_`#\s]*)code\s*:?[\s*`]*$", line.strip(), re.I)]
        if prose and not any(re.search(r"\bidea\b", line, re.I) for line in prose):
            return " ".join(prose)
    return re.sub(r"^[*_`#\s]+", "", " ".join(cleaned).strip()).strip() or None


def parse_candidate(response, finish_reason, interface, template_program, *,
                    base_code=None, allowed_donor_ids=()):
    if finish_reason not in ("stop", "length", "unknown"):
        return None, "unsupported finish reason"
    text = THINK_BLOCK_RE.sub("", response)
    payload = response_object(text)
    fields, mode, donor_id, base_hash = {}, 'full', None, None
    if payload is not None:
        idea, fields = idea_metadata(payload.get('idea'))
        mode = payload.get('mode', 'full')
        declared = payload.get('donor_id')
        if type(declared) is int and declared in allowed_donor_ids:
            donor_id = declared
        if mode == 'edit':
            if base_code is None:
                return None, 'edit_error: no parent is available for editing'
            if 'code' in payload:
                return None, 'edit_error: return edits or full code, not both'
            base_hash = payload.get('base_hash')
            try:
                canonical = apply_edits(base_code, base_hash, payload.get('edits'))
            except ValueError as exc:
                return None, f'edit_error: {exc}'
        elif mode == 'full':
            if 'edits' in payload or not isinstance(payload.get('code'), str):
                return None, 'format_error: full mode requires code and no edits'
            canonical = payload['code']
        else:
            return None, 'format_error: expected full or edit mode'
    else:
        # Backwards-compatible code-first extraction, including code without Idea.
        fences = list(FENCE_RE.finditer(text))
        if len(fences) == 2:
            idea = _extract_idea(text[:fences[0].start()]) or ''
            canonical = text[fences[0].end():fences[1].start()]
        elif not fences:
            idea, canonical = '', text
        else:
            return None, 'format_error: ambiguous Python code blocks'
    canonical = normalize_code(canonical)
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
    if isinstance(targets[0], ast.AsyncFunctionDef):
        return None, 'signature_error: target must be synchronous'
    # An edit operates on the entire displayed program. Do not reinsert template
    # statements or reorder its top-level code after applying the exact edits.
    rebuilt = (normalize_code(ast.unparse(tree)) if mode == 'edit' else
               _merge_candidate(template_program, tree, targets[0]))
    if rebuilt is None:
        return None, "template_error: could not find one target function in the task template"
    try:
        compile(rebuilt, '<candidate>', 'exec')
    except (SyntaxError, ValueError) as exc:
        return None, f'syntax_error: {exc}'
    return ParsedCandidate(idea, rebuilt, fields, mode, donor_id, base_hash), None


def build_repair_prompt(task_contract, response, event, *, base_code=None,
                        failed_program=None):
    message = (event.get("error") or event.get("reason") or "Evaluation failed").strip()
    error_type = event.get("error_type") or "Error"
    payload = response_object(response)
    needs_base = payload is not None and payload.get('mode') == 'edit' and not failed_program
    parent = f'\n\n# Edit base\n```python\n{base_code}\n```' if needs_base and base_code else ''
    failed = (f'```python\n{failed_program}\n```' if failed_program else
              json.dumps({k: v for k, v in payload.items() if k in
                          ('mode', 'code', 'base_hash', 'edits')}, ensure_ascii=False)
              if payload is not None else THINK_BLOCK_RE.sub('', response))
    return (f"{task_contract}{parent}\n\n# Failed output\n{failed}\n\n"
            f"# Failure\n{error_type}: {message[:2000]}\n\n"
            "# Repair\nCorrect the reported failure while preserving the intended algorithmic "
            "idea. Return a full implementation in JSON (mode=full); do not request context.\n\n"
            + FULL_OUTPUT_FORMAT)

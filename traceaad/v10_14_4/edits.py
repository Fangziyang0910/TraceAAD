"""Exact, simultaneous edits; source identity is never semantic similarity."""

import ast
import builtins
import copy
import symtable
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
    interface = task_interface(template)
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


def task_interface(template):
    """Multiple-function templates require an explicit '# Target: name' contract."""
    target = re.search(r"(?im)^\s*#\s*Target:\s*(\w+)\s*$", template)
    if target:
        nodes = [n for n in ast.parse(template).body
                 if isinstance(n, ast.FunctionDef) and n.name == target.group(1)]
        if len(nodes) != 1:
            raise ValueError("invalid explicit template target")
        interface, _ = template_target(ast.unparse(nodes[0]))
    else:
        interface, _ = template_target(template)
    return interface


def defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {a.asname or (a.name.split('.')[0] if isinstance(node, ast.Import) else a.name)
                for a in node.names}
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def global_dependencies(code):
    table = symtable.symtable(code, "<candidate>", "exec")
    found = set()

    def visit(scope):
        found.update(s.get_name() for s in scope.get_symbols() if s.is_referenced()
                     and (scope is table or s.is_global()))
        for child in scope.get_children():
            visit(child)
    visit(table)
    bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
    return found - bound - set(vars(builtins)) - {"__name__", "__file__"}


def complete_template_dependencies(code, template):
    """Supply required contract dependencies only; never overwrite submitted names.

    Unknown names remain unknown and are exposed by normal evaluation/repair.
    Supplement statements retain template order; eager dependencies must already
    be bound. This avoids inventing a program by merging alternative drafts.
    """
    nodes = ast.parse(template).body
    target = task_interface(template)[0]
    generated = ast.parse(code).body
    bound = set().union(*(defined_names(n) for n in generated)) if generated else set()
    candidates = {name: n for n in nodes if not (isinstance(n, ast.FunctionDef) and n.name == target)
                  for name in defined_names(n) if name not in bound}
    needed, included = global_dependencies(code), set()
    while needed:
        name = needed.pop()
        node = candidates.get(name)
        if node is None or id(node) in included:
            continue
        if defined_names(node) & bound:
            raise SourceError("template dependency would overwrite submitted binding", code)
        included.add(id(node))
        segment = ast.get_source_segment(template, node)
        needed.update(global_dependencies(segment) - bound)
    additions = [n for n in nodes if id(n) in included]
    # Imports/definitions may use names provided by earlier supplements, but
    # cannot eagerly reference candidate definitions that have not executed yet.
    available = set(vars(builtins)) | {"__name__"}
    for node in additions:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            eager = [*node.decorator_list, *node.args.defaults,
                     *(x for x in node.args.kw_defaults if x is not None)]
            eager += [a.annotation for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                      if a.annotation is not None]
            if node.returns:
                eager.append(node.returns)
            required = {x.id for expr in eager for x in ast.walk(expr)
                        if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            required = set()
        else:
            required = global_dependencies(ast.get_source_segment(template, node))
        if required - available:
            raise SourceError("template dependency order is not self-contained: " + ', '.join(sorted(required - available)), code)
        available.update(defined_names(node))
    if not additions:
        return code, []
    prefix_end = 0
    for node in generated:
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__"
                or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            prefix_end = node.end_lineno
        else:
            break
    lines = code.splitlines(keepends=True)
    prefix = ''.join(lines[:prefix_end])
    segments = [ast.get_source_segment(template, n) for n in additions]
    return prefix + ('\n' if prefix and not prefix.endswith('\n') else '') + '\n'.join(segments) + '\n' + ''.join(lines[prefix_end:]), sorted(set().union(*(defined_names(n) for n in additions)))


IDEA_LABEL = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?(?:Idea|算法思路|思路)(?:\*\*)?[ \t]*(?:[:：](?:\*\*)?[ \t]*|[ \t]*$)")
FINAL_LABEL = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?(?:Final implementation|Final code|Final candidate|最终实现|最终代码)(?::?\*\*:|:?\*\*|:)?[ \t]*$")
PART_LABEL = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?Candidate:[ \t]*(\w+)[ \t]*;[ \t]*Part[ \t]+(\d+)/(\d+)[ \t]*$")
CODE_LABEL = re.compile(r"(?im)^[ \t]*(?:#+[ \t]*)?(?:\*\*)?(?:Code|Implementation|Python|代码|实现)(?:[:：]?\*\*[:：]?|[:：])?[ \t]*$")


def extract_idea(text):
    """Read Idea sections; fall back to prose outside the Python code blocks.

    This collects the model's words without judging which draft they explain.
    """
    regions = re.split(r"```(?:python|py)?[ \t]*\n.*?\n[ \t]*```", text, flags=re.S | re.I)
    if len(regions) == 1:
        return "", []
    labelled, unlabelled = [], []
    for i, region in enumerate(regions):
        position = "before" if i == 0 else "after" if i == len(regions)-1 else "between"
        region = '\n'.join(line for line in region.splitlines()
                           if not any(label.fullmatch(line) for label in (FINAL_LABEL, PART_LABEL, CODE_LABEL)))
        label = IDEA_LABEL.search(region)
        if label and (value := IDEA_LABEL.sub("", region[label.end():]).strip()):
            labelled.append((position, "explicit_label", value))
        if value := IDEA_LABEL.sub("", region).strip():
            unlabelled.append((position, "unlabelled_prose", value))
    chosen = labelled or unlabelled
    return '\n\n'.join(value for _, _, value in chosen), [
        {"position": position, "kind": kind} for position, kind, _ in chosen]


def idea_metadata(idea, count_tokens, limit=500):
    """Raw text is immutable; only its prompt/display view can be shortened."""
    if not idea:
        return {"raw_tokens": 0, "display": "", "display_tokens": 0, "truncated": False, "limit": limit}
    if count_tokens is None:
        return {"raw_tokens": None, "display": idea, "display_tokens": None,
                "truncated": False, "limit": limit, "token_count_status": "not_measured_offline"}
    try:
        total = count_tokens(idea)
    except Exception as exc:
        # Metadata service failure must not throw away a complete algorithm.
        return {"raw_tokens": None, "display": "", "display_tokens": 0,
                "truncated": True, "limit": limit, "token_count_status": "unavailable",
                "token_count_error": f"{type(exc).__name__}: {exc}"}
    if total <= limit:
        return {"raw_tokens": total, "display": idea, "display_tokens": total, "truncated": False, "limit": limit}
    lo, hi = 0, len(idea)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(idea[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    display = idea[:lo]
    return {"raw_tokens": total, "display": display, "display_tokens": count_tokens(display) if display else 0,
            "truncated": True, "limit": limit, "token_count_status": "serving_tokenizer"}


def validate_assembly_order(code):
    """Reject forward eager dependencies; function bodies can use later helpers."""
    class EagerOnly(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.body = [ast.Pass()]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

    available = set(vars(builtins)) | {"__name__"}
    for original in ast.parse(code).body:
        node = EagerOnly().visit(copy.deepcopy(original))
        needed = global_dependencies(ast.unparse(node))
        if needed - available:
            raise SourceError("candidate part execution order has unresolved eager dependencies: "
                              + ', '.join(sorted(needed - available)), code)
        available.update(defined_names(original))


def _delivery(text, template):
    blocks = list(re.finditer(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```", text, re.S | re.I))
    if not blocks:
        if '```' in text:
            raise ValueError("no complete Python block")
        return text, {"strategy": "bare_source", "block_indices": []}, 0
    if text.count('```') != 2 * len(blocks):
        raise ValueError("unclosed or non-Python code block")
    target = task_interface(template)[0]
    contexts = [text[blocks[i-1].end() if i else 0:b.start()] for i, b in enumerate(blocks)]
    finals = [i for i, c in enumerate(contexts) if FINAL_LABEL.search(c)]
    if len(finals) > 1:
        raise ValueError("multiple explicit final implementations")
    if finals:
        i = finals[0]
        return blocks[i].group(1), {"strategy": "explicit_final", "block_indices": [i]}, blocks[i].start()
    parts = [PART_LABEL.search(c) for c in contexts]
    if any(parts):
        if not all(parts):
            raise ValueError("mixed grouped and ungrouped candidate blocks")
        labels = [p.groups() for p in parts]
        if (len({x[0] for x in labels}) != 1 or any(int(x[2]) != len(blocks) for x in labels)
                or [int(x[1]) for x in labels] != list(range(1, len(blocks)+1))):
            raise ValueError("candidate parts require one group in explicit execution order")
        seen = set()
        for b in blocks:
            for node in ast.parse(b.group(1)).body:
                names = defined_names(node)
                if seen & names:
                    raise ValueError("conflicting definitions in candidate parts")
                seen.update(names)
        code = '\n\n'.join(b.group(1) for b in blocks)
        # Undefined names cannot be guessed from other alternatives or reordered.
        completed, _ = complete_template_dependencies(code, template)
        if global_dependencies(completed):
            raise SourceError("unresolved dependency in explicitly grouped candidate", code)
        validate_assembly_order(completed)
        return code, {"strategy": "explicit_ordered_parts", "block_indices": list(range(len(blocks)))}, blocks[0].start()
    complete = []
    for i, b in enumerate(blocks):
        try:
            tree = ast.parse(b.group(1))
            if sum(isinstance(n, ast.FunctionDef) and n.name == target for n in tree.body) == 1:
                complete.append(i)
        except SyntaxError:
            pass
    if not complete:
        if len(blocks) == 1:
            return blocks[0].group(1), {"strategy": "single_payload_for_repair", "block_indices": [0]}, blocks[0].start()
        raise ValueError("no complete candidate implementation")
    i = complete[-1]
    return blocks[i].group(1), {"strategy": "last_complete_candidate", "block_indices": [i]}, blocks[i].start()


def parse_response(text, finish_reason, template, *, base=None, mode="full",
                   metadata=None, count_tokens=None, idea_limit=500):
    """Choose one complete deliverable and preserve its full, multiline Idea."""
    if finish_reason not in {None, "unknown", "stop"}:
        raise ValueError(f"incomplete completion: {finish_reason}")
    if not isinstance(text, str):
        raise ValueError("response must be text")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "<think>" in text.lower():
        raise ValueError("unclosed reasoning block")
    info = metadata if metadata is not None else {}
    if mode == "edit" and base is not None:
        try:
            payload = json.loads(text)
            if payload.get("mode") != "edit":
                raise ValueError("expected configured edit payload")
            code = apply_edits(base, tuple(Edit(**e) for e in payload["edits"]))
            idea = str(payload.get("idea", ""))
            info.update(strategy="simultaneous_edits", block_indices=[],
                        idea_sources=[{"position": "edit_payload", "kind": "explicit_field"}] if idea.strip() else [])
        except (TypeError, KeyError, AttributeError) as exc:
            raise ValueError("invalid edit payload") from exc
    else:
        idea, info["idea_sources"] = extract_idea(text)
    info["idea_status"] = "present" if idea.strip() else "missing_in_response"
    info["idea_raw"] = idea
    info["idea"] = idea_metadata(idea, count_tokens, idea_limit)
    if mode != "edit" or base is None:
        # A malformed code envelope must not erase an already supplied Idea.
        code, delivery, _ = _delivery(text, template)
        info.update(delivery)
    try:
        code, additions = complete_template_dependencies(code, template)
        info["template_additions"] = additions
        validate_source(code, template)
    except (SyntaxError, ValueError) as exc:
        raise SourceError(str(exc), code) from exc
    return code, idea

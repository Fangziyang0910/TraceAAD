"""One full-program deliverable, with the model's Idea kept verbatim."""

import ast
import builtins
import re
import symtable

from traceaad.v10_13.parsing import template_target


class SourceError(ValueError):
    def __init__(self, message, code, submitted_code=None):
        super().__init__(message)
        self.code = code
        self.submitted_code = submitted_code if submitted_code is not None else code


class DeliveryError(ValueError):
    pass


BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```", re.S | re.I)
IDEA = re.compile(r"(?im)^[ \t]*Idea[ \t]*:[ \t]*")
CODE = re.compile(r"(?im)^[ \t]*Code[ \t]*:[ \t]*$")


def target_name(template):
    return template_target(template)[0][0]


def extract_idea(text):
    label = IDEA.search(text)
    if not label:
        return ""
    block = BLOCK.search(text, label.end())
    stop = block.start() if block else len(text)
    code_label = CODE.search(text, label.end(), stop)
    if code_label:
        stop = code_label.start()
    return text[label.end():stop].strip()


def _delivery(text, template):
    blocks = list(BLOCK.finditer(text))
    if not blocks:
        if "```" in text:
            raise DeliveryError("no complete Python code block")
        label = IDEA.search(text)
        if label:
            code_label = CODE.search(text, label.end())
            if code_label:
                text = text[code_label.end():]
        return text.strip(), {"strategy": "bare_source", "block_indices": []}
    if text.count("```") != len(blocks) * 2:
        raise DeliveryError("unclosed or unsupported code block")
    target = target_name(template)
    complete = []
    for index, block in enumerate(blocks):
        try:
            tree = ast.parse(block.group(1))
        except SyntaxError:
            continue
        if sum(isinstance(node, ast.FunctionDef) and node.name == target for node in tree.body) == 1:
            complete.append(index)
    if complete:
        index = complete[-1]
        return blocks[index].group(1), {"strategy": "last_complete_candidate", "block_indices": [index]}
    if len(blocks) == 1:
        return blocks[0].group(1), {"strategy": "single_payload_for_repair", "block_indices": [0]}
    raise DeliveryError("no complete candidate implementation")


def complete_template_dependencies(code, template):
    """Freeze V10.14's dependency completion: include only needed bindings."""
    nodes = ast.parse(template).body
    generated = ast.parse(code).body
    bound = set().union(*(defined_names(n) for n in generated)) if generated else set()
    candidates = {name: n for n in nodes
                  if not (isinstance(n, ast.FunctionDef) and n.name == target_name(template))
                  for name in defined_names(n) if name not in bound}
    needed, included = global_dependencies(code), set()
    while needed:
        node = candidates.get(needed.pop())
        if node is None or id(node) in included:
            continue
        if defined_names(node) & bound:
            raise SourceError("template dependency would overwrite submitted binding", code)
        included.add(id(node))
        needed.update(global_dependencies(ast.get_source_segment(template, node)) - bound)
    additions = [node for node in nodes if id(node) in included]
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
            raise SourceError("template dependency order is not self-contained: "
                              + ", ".join(sorted(required - available)), code)
        available.update(defined_names(node))
    if not additions:
        return code, []
    prefix_end = 0
    for node in generated:
        if (isinstance(node, ast.ImportFrom) and node.module == "__future__") or (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)):
            prefix_end = node.end_lineno
        else:
            break
    lines = code.splitlines(keepends=True)
    prefix = "".join(lines[:prefix_end])
    segments = [ast.get_source_segment(template, node) for node in additions]
    completed = prefix + ("\n" if prefix and not prefix.endswith("\n") else "") + "\n".join(segments) + "\n" + "".join(lines[prefix_end:])
    return completed, sorted(set().union(*(defined_names(node) for node in additions)))


def defined_names(node):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or (alias.name.split(".")[0] if isinstance(node, ast.Import) else alias.name)
                for alias in node.names}
    return {child.id for child in ast.walk(node)
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)}


def global_dependencies(code):
    table = symtable.symtable(code, "<candidate>", "exec")
    found = set()

    def visit(scope):
        found.update(symbol.get_name() for symbol in scope.get_symbols()
                     if symbol.is_referenced() and (scope is table or symbol.is_global()))
        for child in scope.get_children():
            visit(child)

    visit(table)
    bound = {symbol.get_name() for symbol in table.get_symbols()
             if symbol.is_assigned() or symbol.is_imported()}
    return found - bound - set(vars(builtins)) - {"__name__", "__file__"}


def validate_source(code, template):
    tree = ast.parse(code)
    name, _, signature = template_target(template)[0]
    targets = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(targets) != 1:
        raise ValueError(f"expected one top-level function {name}")
    args = targets[0].args
    actual = ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
              [a.arg for a in args.kwonlyargs], args.vararg.arg if args.vararg else None,
              args.kwarg.arg if args.kwarg else None)
    if actual != signature:
        raise ValueError("target signature differs from the task interface")
    compile(code, "<candidate>", "exec")


def parse_response(text, finish_reason, template):
    if finish_reason != "stop":
        raise DeliveryError(f"incomplete completion: {finish_reason}")
    if not isinstance(text, str):
        raise DeliveryError("response must be text")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "<think>" in text.lower():
        raise DeliveryError("unclosed reasoning block")
    idea = extract_idea(text)
    code, metadata = _delivery(text, template)
    if not code.strip() or not BLOCK.search(text) and not re.search(r"(?m)^\s*(?:def |import |from |class |@)", code):
        raise DeliveryError("no usable code")
    submitted_code = code
    try:
        code, additions = complete_template_dependencies(code, template)
        validate_source(code, template)
    except SourceError as exc:
        exc.submitted_code = submitted_code
        raise
    except (SyntaxError, ValueError) as exc:
        raise SourceError(str(exc), code, submitted_code) from exc
    metadata["submitted_code"] = submitted_code
    metadata["template_additions"] = additions
    return code, idea, metadata

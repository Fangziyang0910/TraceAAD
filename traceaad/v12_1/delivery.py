"""Extract the final submitted program and check its task interface.

The evaluator runs the submitted source, without adding template dependencies.
Missing imports and helpers follow the same evaluation/Repair path as other
program errors. The original reply remains in the generation record.
"""

import ast
import json
import re


class DeliveryError(ValueError):
    pass


BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```", re.S | re.I)
OPENER = re.compile(r"```(?:python|py)?[ \t]*\n", re.I)
IDEA = re.compile(r"(?im)^[ \t]*(?:Design|Idea|Thought)[ \t]*:[ \t]*")
CODE = re.compile(r"(?im)^[ \t]*Code[ \t]*:[ \t]*$")


def target(template):
    return next(
        node for node in ast.parse(template).body if isinstance(node, ast.FunctionDef)
    )


def extract_idea(text):
    labels = list(IDEA.finditer(text))
    if not labels:
        return ""
    label = labels[-1]
    block = BLOCK.search(text, label.end()) or OPENER.search(text, label.end())
    stop = block.start() if block else len(text)
    code_label = CODE.search(text, label.end(), stop)
    return text[label.end() : code_label.start() if code_label else stop].strip()


def arguments(function):
    args = function.args
    return (
        [a.arg for a in args.posonlyargs],
        [a.arg for a in args.args],
        [a.arg for a in args.kwonlyargs],
        args.vararg.arg if args.vararg else None,
        args.kwarg.arg if args.kwarg else None,
    )


def signature(function):
    return f"{function.name}({ast.unparse(function.args)})"


def located(exc):
    """A syntax error stated with the line it is on."""
    line = (exc.text or "").strip()
    return SyntaxError(
        f"{exc.msg} at line {exc.lineno}" + (f": `{line}`" if line else "")
    )


def validate_source(code, template):
    expected = target(template)
    try:
        functions = [
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name == expected.name
        ]
    except SyntaxError as exc:
        raise located(exc) from exc
    if len(functions) != 1 or arguments(functions[0]) != arguments(expected):
        found = ", ".join(f"`{signature(f)}`" for f in functions) or "none"
        raise ValueError(
            f"the program must define one function `{signature(expected)}`; found: {found}"
        )
    try:
        compile(code, "<candidate>", "exec")
    except SyntaxError as exc:
        raise located(exc) from exc


# Every model request asks for one JSON object.
JSON_REPLY = {"type": "json_object"}


def read_json(text, finish_reason, fields):
    """The JSON object of a complete reply, with the given string fields."""
    if finish_reason != "stop" or not isinstance(text, str):
        raise DeliveryError(f"incomplete or non-text completion: {finish_reason}")
    try:
        reply = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DeliveryError(f"reply is not JSON: {exc}") from exc
    if not isinstance(reply, dict) or not all(
        isinstance(reply.get(f), str) for f in fields
    ):
        raise DeliveryError("reply needs string fields " + ", ".join(fields))
    return reply


def parse_json_response(text, finish_reason, fields=("design", "code")):
    """Parse a program reply; syntax and execution are individual evaluation states."""
    reply = read_json(text, finish_reason, fields)
    code = reply["code"]
    if not code.strip():
        raise DeliveryError("no usable code")
    idea = reply["design"].strip() if "design" in fields else ""
    analysis = reply.get("analysis")
    analysis = analysis.strip() if isinstance(analysis, str) else None
    return code, idea, {"strategy": "json", "analysis": analysis}

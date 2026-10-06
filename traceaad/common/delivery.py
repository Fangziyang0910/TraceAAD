"""Extract the final submitted program and check its task interface.

The evaluator runs the submitted source, without adding template dependencies.
Missing imports and helpers follow the same evaluation/Repair path as other
program errors. The original reply remains in the generation record.
"""

import ast
import re


class SourceError(ValueError):
    def __init__(self, message, code):
        super().__init__(message)
        self.code = code


class DeliveryError(ValueError):
    pass


BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```", re.S | re.I)
OPENER = re.compile(r"```(?:python|py)?[ \t]*\n", re.I)
IDEA = re.compile(r"(?im)^[ \t]*(?:Design|Idea|Thought)[ \t]*:[ \t]*")
CODE = re.compile(r"(?im)^[ \t]*Code[ \t]*:[ \t]*$")


def target(template):
    return next(node for node in ast.parse(template).body if isinstance(node, ast.FunctionDef))


def extract_idea(text):
    labels = list(IDEA.finditer(text))
    if not labels:
        return ""
    label = labels[-1]
    block = BLOCK.search(text, label.end()) or OPENER.search(text, label.end())
    stop = block.start() if block else len(text)
    code_label = CODE.search(text, label.end(), stop)
    return text[label.end():code_label.start() if code_label else stop].strip()


def arguments(function):
    args = function.args
    return ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
            [a.arg for a in args.kwonlyargs], args.vararg.arg if args.vararg else None,
            args.kwarg.arg if args.kwarg else None)


def validate_source(code, template):
    expected = target(template)
    functions = [node for node in ast.parse(code).body
                 if isinstance(node, ast.FunctionDef) and node.name == expected.name]
    if len(functions) != 1 or arguments(functions[0]) != arguments(expected):
        raise ValueError("expected one target function with the task's argument names")
    compile(code, "<candidate>", "exec")


def parse_response(text, finish_reason, template):
    if finish_reason != "stop" or not isinstance(text, str):
        raise DeliveryError(f"incomplete or non-text completion: {finish_reason}")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    if "<think>" in text.lower():
        raise DeliveryError("unclosed reasoning block")
    idea = extract_idea(text)
    labels = list(CODE.finditer(text))
    payload = text[labels[-1].end():] if labels else text
    blocks = list(BLOCK.finditer(payload))
    tail = OPENER.search(payload, blocks[-1].end()) if blocks else None
    if tail:
        code, strategy = payload[tail.end():].rstrip(), "unclosed_final_block"
    elif blocks:
        code, strategy = blocks[-1].group(1), "last_code_block"
    elif opener := OPENER.search(payload):
        code, strategy = payload[opener.end():].rstrip(), "unclosed_final_block"
    else:
        code, strategy = payload.strip(), "bare_source"
    if not code.strip() or (not blocks and not re.search(r"(?m)^\s*(?:def |import |from |class |@)", code)):
        raise DeliveryError("no usable code")
    try:
        validate_source(code, template)
    except (SyntaxError, ValueError) as exc:
        raise SourceError(str(exc), code) from exc
    if labels:
        strategy = "after_last_code_label:" + strategy
    return code, idea, {"strategy": strategy}

"""Python source representations used by the comparison methods."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class Function:
    name: str
    args: str
    body: str
    return_type: str | None = None
    docstring: str | None = None
    score: Any | None = None
    evaluate_time: float | None = None
    sample_time: float | None = None
    operator: str | None = "Unknown"
    source_prefix: str = ""
    source_suffix: str = ""
    raw_source: str | None = None
    algorithm = ""

    def __str__(self):
        if self.raw_source is not None:
            return self.raw_source
        returns = f" -> {self.return_type}" if self.return_type else ""
        code = f"def {self.name}({self.args}){returns}:\n"
        if self.docstring:
            code += f'    """{self.docstring}"""' + ("\n" if self.body else "")
        return self.source_prefix + code + self.body + "\n\n" + self.source_suffix

    def __setattr__(self, name, value):
        if name == "body":
            value = value.strip("\n")
        if name == "docstring" and value is not None and '"""' in value:
            value = value.strip().replace('"""', "")
        if name in {"name", "args", "body", "return_type", "docstring", "source_prefix", "source_suffix"}:
            if name in self.__dict__ and self.__dict__[name] != value:
                super().__setattr__("raw_source", None)
        super().__setattr__(name, value)

    def __eq__(self, other):
        if not isinstance(other, Function):
            return NotImplemented
        fields = ("name", "args", "return_type", "body", "source_prefix", "source_suffix")
        return all(getattr(self, name) == getattr(other, name) for name in fields)


@dataclass(frozen=True)
class Program:
    preface: str
    functions: list[Function]

    def __str__(self):
        return (self.preface + "\n" if self.preface else "") + "\n".join(map(str, self.functions))


def parse_program(source: str) -> Program | None:
    try:
        tree = ast.parse(source)
    except SyntaxError, ValueError:
        return None
    lines, functions, preface = source.splitlines(), [], ""
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if not functions:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            preface = "\n".join(lines[: start - 1])
        doc = ast.get_docstring(node, clean=False)
        statements = node.body[1:] if doc is not None else node.body
        start = node.body[0].end_lineno if doc is not None else node.body[0].lineno - 1
        prefix = lines[node.body[0].lineno - 1][: node.body[0].col_offset]
        if prefix.strip():
            body = (
                "    " + "; ".join(ast.get_source_segment(source, stmt) for stmt in statements)
                if statements
                else ""
            )
        else:
            body = "\n".join(lines[start : node.end_lineno])
        functions.append(
            Function(
                name=node.name,
                args=ast.unparse(node.args),
                body=body,
                return_type=ast.unparse(node.returns) if node.returns else None,
                docstring=doc,
            )
        )
    return Program(preface=preface, functions=functions)


def parse_function(program: str | Program) -> Function | None:
    if isinstance(program, str):
        program = parse_program(program)
    if program is None:
        return None
    if len(program.functions) != 1:
        raise ValueError(f"Expected one function, got {len(program.functions)}")
    return copy.deepcopy(program.functions[0])


def replace_function(function: str | Function, template: str | Program) -> Program | None:
    function = parse_function(function) if isinstance(function, str) else copy.deepcopy(function)
    template = parse_program(template) if isinstance(template, str) else copy.deepcopy(template)
    if function is None or template is None:
        return None
    if len(template.functions) != 1:
        raise ValueError(f"Expected one template function, got {len(template.functions)}")
    if function.source_prefix or function.source_suffix:
        return Program(preface="", functions=[function])
    template.functions[0].body = function.body
    return template


def function_from_source(source: str, entrypoint: str) -> Function | None:
    """Represent a method's accepted module without changing its callable contract."""
    try:
        tree = ast.parse(source)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entrypoint)
        function = parse_function(ast.get_source_segment(source, node))
        lines = source.splitlines(keepends=True)
        function.source_prefix = ''.join(lines[:node.lineno - 1]) or '\n'
        function.source_suffix = ''.join(lines[node.end_lineno:])
        function.raw_source = source
        return function
    except (SyntaxError, ValueError, StopIteration):
        return None

# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
"""Response trimming and text-to-Function conversion shared by baseline samplers."""

from __future__ import annotations

import ast
import copy
import re
from typing import Any, List

from core import Function, LLM, Program, TextFunctionProgramConverter


class SampleTrimmer:
    def __init__(self, llm: LLM):
        self.llm = llm

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """Get a sample based on the provided 'LLM' instance.
        Preserve the full response for the task-aware program parser.
        """
        generated_code = self.llm.draw_sample(prompt, *args, **kwargs)
        return generated_code

    def draw_samples(self, prompts: List[str | Any], *args, **kwargs) -> List[str]:
        """Get samples based on the provided 'Sampler' instance.
        Preserve the full response for the task-aware program parser.
        """
        ret = self.llm.draw_samples(prompts, *args, **kwargs)
        return ret

    @classmethod
    def _check_indent_if_code_completion(cls, generated_code: str) -> bool:
        """Judge if the content is generated through code completion model or instruct model.
        """
        generated_code = generated_code.strip('\n')
        line = generated_code.splitlines()[0]
        if line.startswith('\t'):
            return True
        if line.startswith(' ' * 2):
            return True
        if line.startswith(' ' * 4):
            return True
        return False

    @classmethod
    def trim_preface_of_function(cls, generated_code: str):
        """Trim the redundant descriptions/symbols/'def' declaration BEFORE the function body.
        Example of a generated content from an LLM:
        --------------------------------------------------------------------------
        This is the optimized function ...

        def priority_v2(...) -> ...:
            a = random.random()
            return a * a

        This function aims to ...
        --------------------------------------------------------------------------
        Example return of this function:
        --------------------------------------------------------------------------
            a = random.random()
            return a * a

        This function aims to ...
        --------------------------------------------------------------------------
        """
        lines = generated_code.splitlines()
        func_body_lineno = 0
        find_def_declaration = False
        for lineno, line in enumerate(lines):
            # find the first 'def' statement in the given code
            if line[:3] == 'def':
                func_body_lineno = lineno
                find_def_declaration = True
                break
        if find_def_declaration:
            code = ''
            for line in lines[func_body_lineno + 1:]:
                code += line + '\n'
            return code
        return generated_code

    @classmethod
    def auto_trim(cls, generated_code: str) -> str:
        """Automatically trim the preface of the generated content.
        """
        is_code_complete = cls._check_indent_if_code_completion(generated_code)
        if is_code_complete:
            return generated_code
        generated_code = cls.trim_preface_of_function(generated_code)
        return generated_code

    @classmethod
    def sample_to_function(cls, generated_code: str, template_program: str | Program) -> Function | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        The returned entrypoint retains its module imports and helpers.
        """
        program = cls.sample_to_program(generated_code, template_program)
        if program is None:
            return None
        return TextFunctionProgramConverter.program_to_function(program)

    @classmethod
    def sample_to_program(cls, generated_code: str, template_program: str | Program) -> Program | None:
        """Convert the generated content (with redundant component)
        to a Function instance. If the convert fails, return None.
        """
        if isinstance(template_program, str):
            template_program = TextFunctionProgramConverter.text_to_program(template_program)
        if template_program is None:
            return None
        target = template_program.functions[0]
        response = re.sub(r"<think>.*?</think>", "", generated_code, flags=re.DOTALL)
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
        sources = blocks or [response]
        for source in sources:
            lines = source.splitlines()
            if not blocks:
                starts = [i for i, line in enumerate(lines)
                          if re.match(r"^(?:def |class |import |from |@|[A-Za-z_]\w*\s*(?::[^=]+)?=)", line)]
                if starts:
                    lines = lines[starts[0]:]
            while lines:
                try:
                    tree = ast.parse("\n".join(lines))
                    break
                except SyntaxError:
                    lines.pop()
            else:
                continue
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
            matches = [n for n in functions if re.fullmatch(re.escape(target.name) + r"(?:_v\d+)?", n.name)]
            if not matches:
                continue
            node = matches[-1]
            old_name = node.name
            # Versioned responses retain helper functions, imports and module state.
            class Rename(ast.NodeTransformer):
                def visit_Name(self, item):
                    if item.id == old_name:
                        item.id = target.name
                    return item
            tree = Rename().visit(tree)
            node.name = target.name
            ast.fix_missing_locations(tree)
            code = ast.unparse(tree)
            tree = ast.parse(code)
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == target.name)
            function = TextFunctionProgramConverter.text_to_function(ast.get_source_segment(code, node))
            if function is None:
                continue
            # The experiment supplies the callable contract, as in the previous adapter.
            function.name, function.args, function.return_type = target.name, target.args, target.return_type
            function.docstring = target.docstring
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            code_lines = code.splitlines(keepends=True)
            prefix = ''.join(code_lines[:start])
            imports = '\n'.join(line for line in template_program.preface.splitlines() if line not in prefix.splitlines())
            futures = [line for line in prefix.splitlines() if line.startswith('from __future__ import ')]
            rest = '\n'.join(line for line in prefix.splitlines() if not line.startswith('from __future__ import '))
            decorators = ''.join(code_lines[start:node.lineno - 1])
            function.source_prefix = '\n'.join(part for part in ('\n'.join(futures), imports, rest) if part)
            if function.source_prefix:
                function.source_prefix += '\n'
            function.source_prefix += decorators
            function.source_suffix = ''.join(code_lines[node.end_lineno:])
            return Program(preface='', functions=[function])
        # FunSearch also accepts a completion containing only an indented body.
        if generated_code.startswith((' ', '\t')):
            body = cls.trim_function_body(generated_code)
            if body:
                function = copy.deepcopy(target)
                function.body = body
                return Program(preface=template_program.preface, functions=[function])
        return None

    @classmethod
    def trim_function_body(cls, generated_code: str) -> str | None:
        """Extracts the body of the generated function, trimming anything after it.
        """
        try:
            if not generated_code:
                return ''
            code = f'def fake_function_header():\n{generated_code}'

            # keep trying and deleting code from the end until the parser succeeds
            tree = None
            while tree is None:
                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    # "e.lineno - 1" locates the line number of the lost python code
                    code = '\n'.join(code.splitlines()[:e.lineno - 1])

            if not code:
                # Nothing could be saved from `generated_code`
                return ''

            visitor = _FunctionLineVisitor('fake_function_header')
            visitor.visit(tree)
            body_lines = code.splitlines()[1:visitor.function_end_line]
            return '\n'.join(body_lines) + '\n\n'
        except:
            return None

    @classmethod
    def remove_docstrings(cls, func: Function | str):
        func_ = copy.deepcopy(func)
        func_ = TextFunctionProgramConverter.text_to_function(str(func_))  # convert to Function instance
        docstring = func_.docstring
        while not (docstring == "" or docstring is None):
            func_.docstring = ""
            func_str = str(func_)
            func_ = TextFunctionProgramConverter.text_to_function(func_str)
            docstring = func_.docstring

        if isinstance(func, Function):
            for key, value in func.__dict__.items():
                if key != 'docstring' and key != 'body':
                    setattr(func_, key, value)
            return func_
        else:
            return str(func_)


class _FunctionLineVisitor(ast.NodeVisitor):
    """Visitor that finds the last line number of a function with a given name."""

    def __init__(self, target_function_name: str) -> None:
        self._target_function_name: str = target_function_name
        self._function_end_line: int | None = None

    def visit_FunctionDef(self, node: Any) -> None:  # pylint: disable=invalid-name
        """Collects the end line number of the target function."""
        if node.name == self._target_function_name:
            self._function_end_line = node.end_lineno
        self.generic_visit(node)

    @property
    def function_end_line(self) -> int:
        """Line number of the final line of function `target_function_name`."""
        assert self._function_end_line is not None  # Check internal correctness.
        return self._function_end_line

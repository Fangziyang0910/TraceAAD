from __future__ import annotations

import ast
import copy
import re

from .graph import PathWiseAction, PathWiseNode
from core import Function, LLM, Program, TextFunctionProgramConverter
from baselines.sampling import SampleTrimmer


GLOBAL_IMPORTS = [
    "import numpy as np",
    "import random",
    "import math",
    "import scipy",
    "try:\n    import torch\nexcept Exception:\n    torch = None",
]


class PathWiseSampler:
    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        return self.llm.draw_sample(prompt, *args, **kwargs)

    @staticmethod
    def with_global_imports(program: str | Program) -> Program:
        if isinstance(program, str):
            program = TextFunctionProgramConverter.text_to_program(program)
        else:
            program = copy.deepcopy(program)
        existing = set(line.strip() for line in program.preface.splitlines())
        additions = [line for line in GLOBAL_IMPORTS if line not in existing]
        preface_lines = [line for line in program.preface.splitlines() if line.strip()]
        return Program(preface="\n".join(additions + preface_lines), functions=program.functions)

    @classmethod
    def strip_thinking(cls, text: str) -> str:
        text = text or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        return text.strip()

    @classmethod
    def fenced_blocks(cls, text: str) -> list[str]:
        text = cls.strip_thinking(text)
        pattern = r"```(?:python)?\s*(.*?)```"
        return [block.strip() for block in re.findall(pattern, text, flags=re.DOTALL | re.IGNORECASE)]

    @classmethod
    def _function_source_from_valid_python(cls, source: str) -> str | None:
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                segment = ast.get_source_segment(source, node)
                if segment:
                    return segment.strip()
        return None

    @classmethod
    def _function_source_from_prefix(cls, source: str) -> str | None:
        lines = source.splitlines()
        starts = [idx for idx, line in enumerate(lines) if line.lstrip().startswith("def ")]
        for start in starts:
            for end in range(len(lines), start, -1):
                snippet = "\n".join(lines[start:end])
                found = cls._function_source_from_valid_python(snippet)
                if found:
                    return found
        return None

    @classmethod
    def extract_function_source(cls, response: str) -> str | None:
        text = cls.strip_thinking(response)
        candidates = cls.fenced_blocks(text)
        candidates.append(text)
        for candidate in candidates:
            found = cls._function_source_from_valid_python(candidate)
            if found:
                return found
            found = cls._function_source_from_prefix(candidate)
            if found:
                return found
        return None

    @classmethod
    def response_to_function(cls, response: str, template_program: str | Program) -> Function | None:
        function_source = cls.extract_function_source(response)
        if function_source is None:
            return None
        program = SampleTrimmer.sample_to_program(function_source, template_program)
        if program is None:
            func = TextFunctionProgramConverter.text_to_function(function_source)
            if func is None:
                return None
            program = TextFunctionProgramConverter.function_to_program(func, template_program)
        if program is None:
            return None
        return TextFunctionProgramConverter.program_to_function(program)

    @classmethod
    def _line_value(cls, response: str, labels: list[str]) -> str:
        for line in cls.strip_thinking(response).splitlines():
            stripped = line.strip().strip("*").strip()
            stripped = stripped.replace("```", "").strip()
            for label in labels:
                if stripped.lower().startswith(label.lower()):
                    return stripped[len(label):].strip().strip("[]")
        return ""

    @classmethod
    def parse_initialization_response(cls, response: str, template_program: str | Program) -> tuple[Function, str, str] | None:
        func = cls.response_to_function(response, template_program)
        if func is None:
            return None
        description = cls._line_value(response, ["Description:", "# Description:", "Algorithm description:"])
        rationale = cls._line_value(response, ["Derivation Rationale:", "# Derivation Rationale:", "Rationale:"])
        if not description:
            description = "Initial PathWise heuristic."
        if not rationale:
            rationale = "Initial population sample."
        return func, description, rationale

    @classmethod
    def parse_world_model_response(cls, response: str, template_program: str | Program) -> tuple[Function, str] | None:
        func = cls.response_to_function(response, template_program)
        if func is None:
            return None
        description = cls._line_value(
            response,
            ["Description:", "# Description:", "Algorithm description:", "Algorithmic description:"],
        )
        if not description:
            description = "PathWise world-model rollout."
        return func, description

    @classmethod
    def parse_policy_response(cls, response: str, state: list[PathWiseNode]) -> PathWiseAction | None:
        available = [node.node_id for node in state]
        parents: list[str] = []
        rationale_lines: list[str] = []
        collecting_rationale = False

        for raw_line in cls.strip_thinking(response).splitlines():
            line = raw_line.strip()
            upper = line.upper()
            if upper.startswith("PARENTS:"):
                parent_part = line.split(":", 1)[1].strip()
                parent_tokens = [p.strip() for p in parent_part.replace("[", "").replace("]", "").replace(",", " ").split()]
                invalid = [token for token in parent_tokens if token and token not in available]
                if invalid:
                    return None
                parents = [node_id for node_id in available if node_id in parent_tokens or node_id in parent_part]
                collecting_rationale = False
            elif upper.startswith("DIRECTIVE:") or upper.startswith("RATIONALE:"):
                rationale_lines.append(line.split(":", 1)[1].strip())
                collecting_rationale = True
            elif collecting_rationale and line:
                rationale_lines.append(line)

        if not parents:
            return None
        rationale = " ".join(part for part in rationale_lines if part).strip()
        if not rationale:
            rationale = "Refine the selected parent heuristic to improve performance."
        return PathWiseAction(parents=parents, rationale=rationale)

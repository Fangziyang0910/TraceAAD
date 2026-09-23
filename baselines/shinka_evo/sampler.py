from __future__ import annotations

import ast
import copy
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from llm4ad.base import Function, LLM, Program, TextFunctionProgramConverter
from .population import ShinkaProgram


EVOLVE_START = re.compile(r"^[^\S\r\n]*#\s*EVOLVE-BLOCK-START\s*$", re.MULTILINE)
EVOLVE_END = re.compile(r"^[^\S\r\n]*#\s*EVOLVE-BLOCK-END\s*$", re.MULTILINE)
PATCH_PATTERN = re.compile(
    r"<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}\s*REPLACE\s*",
    re.DOTALL,
)


@dataclass
class PatchResult:
    function: Function | None
    program: Program | None
    code_diff: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.function is not None and self.program is not None and self.error is None


class ShinkaSampler:
    BASE_SYSTEM_MSG = (
        "You are an expert optimization and algorithm design assistant. "
        "Improve the program while preserving correctness and immutable regions."
    )

    DIFF_FORMAT = (
        "\n\n# Patch Format\n"
        "Return one or more SEARCH/REPLACE blocks. Only edit text between EVOLVE-BLOCK-START\n"
        "and EVOLVE-BLOCK-END.\n\n"
        + "<" * 7 + " SEARCH\n"
        "old code\n"
        + "=" * 7 + "\n"
        "new code\n"
        + ">" * 7 + " REPLACE\n"
    )

    FULL_FORMAT = """

# Patch Format
Return a complete Python function or program in a ```python code block```.
Keep immutable code, imports, function name, arguments, return type, and docstring
semantics unchanged. Only the implementation inside the evolve block may change.
"""

    CROSS_FORMAT = """

# Patch Format
Return a complete Python function or program in a ```python code block```.
Combine useful ideas from the current parent and inspiration programs while
preserving the original function signature and immutable regions.
"""

    FIX_FORMAT = """

# Patch Format
Return a complete corrected Python function or program in a ```python code block```.
Make the program correct first; performance improvements are secondary.
"""

    def __init__(
            self,
            llm: LLM,
            template_program: Program,
            *,
            task_description: str,
            patch_types: Sequence[str] = ("diff", "full", "cross"),
            patch_type_probs: Sequence[float] = (0.6, 0.3, 0.1),
            use_text_feedback: bool = False,
            inspiration_sort_order: str = "ascending",
            task_sys_msg: str | None = None,
    ):
        self.llm = llm
        self.template_program = copy.deepcopy(template_program)
        self.template_function = copy.deepcopy(template_program.functions[0])
        self.task_description = task_description
        self.patch_types = list(patch_types)
        self.patch_type_probs = list(patch_type_probs)
        self.use_text_feedback = use_text_feedback
        self.inspiration_sort_order = inspiration_sort_order
        self.task_sys_msg = task_sys_msg or self.BASE_SYSTEM_MSG
        prob_sum = sum(self.patch_type_probs)
        if len(self.patch_types) != len(self.patch_type_probs):
            raise ValueError("patch_types and patch_type_probs must have the same length.")
        if abs(prob_sum - 1.0) > 1e-6:
            raise ValueError(f"patch_type_probs must sum to 1.0, got {prob_sum:.6f}.")

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> tuple[str, float]:
        start = time.time()
        response = self.llm.draw_sample(prompt, *args, **kwargs)
        return response, time.time() - start

    def sample_patch_type(
            self,
            archive_inspirations: Sequence[ShinkaProgram],
            top_k_inspirations: Sequence[ShinkaProgram],
    ) -> str:
        patch_types = self.patch_types
        probs = self.patch_type_probs
        if not archive_inspirations and not top_k_inspirations:
            filtered = [(ptype, prob) for ptype, prob in zip(patch_types, probs) if ptype != "cross"]
            if filtered:
                patch_types, probs = zip(*filtered)
                prob_sum = sum(probs)
                probs = [prob / prob_sum for prob in probs]
        import numpy as np
        return str(np.random.choice(list(patch_types), p=list(probs)))

    def build_prompt(
            self,
            parent: ShinkaProgram,
            archive_inspirations: Sequence[ShinkaProgram],
            top_k_inspirations: Sequence[ShinkaProgram],
            patch_type: str,
            *,
            meta_recommendations: str | None = None,
            previous_error: str | None = None,
            fix_mode: bool = False,
    ) -> str:
        system = self.task_sys_msg
        if meta_recommendations and meta_recommendations != "none" and patch_type != "cross":
            system += (
                "\n\n# Potential Recommendations\n"
                "The following are potential recommendations for the next program generation:\n"
                f"{meta_recommendations}"
            )
        if fix_mode:
            system += self.FIX_FORMAT
        elif patch_type == "diff":
            system += self.DIFF_FORMAT
        elif patch_type == "full":
            system += self.FULL_FORMAT
        elif patch_type == "cross":
            system += self.CROSS_FORMAT
        else:
            raise ValueError(f"Unsupported patch type: {patch_type}")

        user_parts = [
            "# Task",
            self.task_description or "Improve the given algorithm.",
            "",
            "# Current Program",
            "```python",
            self.to_marked_program(parent.function),
            "```",
            "",
            "# Current Performance",
            self._performance_block(parent),
        ]
        inspiration_block = self._inspiration_block(archive_inspirations, top_k_inspirations)
        if inspiration_block:
            user_parts.extend(["", inspiration_block])
        if previous_error:
            user_parts.extend([
                "",
                "# Previous Patch Error",
                previous_error,
                "Try again with a valid patch that preserves immutable regions.",
            ])
        if fix_mode:
            user_parts.extend([
                "",
                "# Error Context",
                str(parent.metadata.get("error", parent.text_feedback or "The current program did not pass evaluation.")),
            ])
        return system.strip() + "\n\n" + "\n".join(user_parts).strip()

    def _performance_block(self, program: ShinkaProgram) -> str:
        lines = [
            f"combined_score: {program.combined_score}",
            f"correct: {program.correct}",
        ]
        if program.public_metrics:
            lines.append(f"public_metrics: {program.public_metrics}")
        if self.use_text_feedback and program.text_feedback:
            lines.append(f"text_feedback: {program.text_feedback}")
        return "\n".join(lines)

    def _inspiration_block(
            self,
            archive_inspirations: Sequence[ShinkaProgram],
            top_k_inspirations: Sequence[ShinkaProgram],
    ) -> str:
        inspirations = list(archive_inspirations) + list(top_k_inspirations)
        if not inspirations:
            return ""
        if self.inspiration_sort_order == "ascending":
            inspirations = sorted(inspirations, key=lambda p: p.combined_score)
        elif self.inspiration_sort_order == "chronological":
            inspirations = sorted(inspirations, key=lambda p: p.generation)
        blocks = ["# Inspiration Programs"]
        for idx, program in enumerate(inspirations, start=1):
            blocks.extend([
                f"## Inspiration {idx}",
                f"score: {program.combined_score}; correct: {program.correct}; island: {program.island_idx}",
                "```python",
                program.program.strip(),
                "```",
            ])
        return "\n".join(blocks)

    def apply_response(
            self,
            response: str,
            parent: ShinkaProgram,
            patch_type: str,
            *,
            fix_mode: bool = False,
    ) -> PatchResult:
        if fix_mode or patch_type in {"full", "cross"}:
            return self._apply_full_response(response, parent, patch_type="fix" if fix_mode else patch_type)
        if patch_type == "diff":
            return self._apply_diff_response(response, parent)
        return PatchResult(None, None, error=f"Unsupported patch type: {patch_type}")

    def to_marked_program(self, function: Function) -> str:
        template_function = copy.deepcopy(self.template_function)
        template_function.body = function.body
        program = TextFunctionProgramConverter.function_to_program(template_function, self.template_program)
        if program is None:
            return str(function)
        return self._insert_markers(str(program))

    def _insert_markers(self, program_str: str) -> str:
        program = TextFunctionProgramConverter.text_to_program(program_str)
        if program is None or len(program.functions) != 1:
            return program_str
        func = program.functions[0]
        marked = copy.deepcopy(func)
        body = marked.body.strip("\n")
        body_lines = body.splitlines() if body else ["    pass"]
        marked.body = "\n".join(["    # EVOLVE-BLOCK-START"] + body_lines + ["    # EVOLVE-BLOCK-END"])
        marked_program = TextFunctionProgramConverter.function_to_program(marked, program)
        return str(marked_program) if marked_program is not None else program_str

    def _apply_full_response(self, response: str, parent: ShinkaProgram, *, patch_type: str) -> PatchResult:
        extracted = self.extract_python_code(response)
        if not extracted:
            return PatchResult(None, None, error="Could not extract Python code from response.")
        source = self.extract_function_source(extracted) or self.extract_function_source(response)
        if not source:
            return PatchResult(None, None, error="Could not extract a Python function from response.")
        parsed = TextFunctionProgramConverter.text_to_function(source)
        if parsed is None:
            return PatchResult(None, None, error="Extracted function is not parseable.")
        func = self._canonicalize_function(parsed)
        program = TextFunctionProgramConverter.function_to_program(func, self.template_program)
        if program is None:
            return PatchResult(None, None, error="Could not rebuild program from generated function.")
        return PatchResult(func, program, code_diff=self._simple_diff(parent.program, str(program)), metadata={"patch_type": patch_type})

    def _apply_diff_response(self, response: str, parent: ShinkaProgram) -> PatchResult:
        matches = PATCH_PATTERN.findall(response or "")
        if not matches:
            return PatchResult(None, None, error="No SEARCH/REPLACE block found.")
        marked = self.to_marked_program(parent.function)
        mutable_ranges = self.mutable_ranges(marked)
        if not mutable_ranges:
            return PatchResult(None, None, error="No EVOLVE-BLOCK region found.")
        updated = marked
        for search_text, replace_text in matches:
            search_text = search_text.rstrip("\n")
            replace_text = replace_text.rstrip("\n")
            position = updated.find(search_text)
            if position < 0:
                return PatchResult(None, None, error="SEARCH text was not found in the current program.")
            span = (position, position + len(search_text))
            if not self._inside_mutable(span, mutable_ranges):
                return PatchResult(None, None, error="SEARCH text touches immutable code outside EVOLVE-BLOCK.")
            updated = updated[:position] + replace_text + updated[position + len(search_text):]
            mutable_ranges = self.mutable_ranges(updated)
        cleaned = self.strip_markers(updated)
        program = TextFunctionProgramConverter.text_to_program(cleaned)
        if program is None or len(program.functions) != 1:
            return PatchResult(None, None, error="Patched program is not parseable.")
        parsed = program.functions[0]
        func = self._canonicalize_function(parsed)
        rebuilt = TextFunctionProgramConverter.function_to_program(func, self.template_program)
        if rebuilt is None:
            return PatchResult(None, None, error="Could not rebuild program from diff result.")
        return PatchResult(func, rebuilt, code_diff=self._simple_diff(parent.program, str(rebuilt)), metadata={"patch_type": "diff"})

    def _canonicalize_function(self, parsed: Function) -> Function:
        func = copy.deepcopy(self.template_function)
        func.body = parsed.body
        return func

    @staticmethod
    def extract_python_code(text: str) -> str | None:
        text = ShinkaSampler.strip_thinking(text)
        blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            return blocks[0].strip()
        return text.strip() if text and text.strip() else None

    @staticmethod
    def strip_thinking(text: str) -> str:
        text = text or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        return text.strip()

    @classmethod
    def extract_function_source(cls, source: str) -> str | None:
        candidates = [source]
        for block in re.findall(r"```(?:python)?\s*(.*?)```", source or "", flags=re.DOTALL | re.IGNORECASE):
            candidates.insert(0, block)
        for candidate in candidates:
            found = cls._function_source_from_valid_python(candidate)
            if found:
                return found
            found = cls._function_source_from_prefix(candidate)
            if found:
                return found
        return None

    @staticmethod
    def _function_source_from_valid_python(source: str) -> str | None:
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
        lines = (source or "").splitlines()
        starts = [idx for idx, line in enumerate(lines) if line.lstrip().startswith("def ")]
        for start in starts:
            for end in range(len(lines), start, -1):
                found = cls._function_source_from_valid_python("\n".join(lines[start:end]))
                if found:
                    return found
        return None

    @staticmethod
    def mutable_ranges(text: str) -> list[tuple[int, int]]:
        markers = []
        for match in EVOLVE_START.finditer(text):
            markers.append((match.end(), "start"))
        for match in EVOLVE_END.finditer(text):
            markers.append((match.start(), "end"))
        markers.sort(key=lambda item: item[0])
        ranges = []
        stack = []
        for pos, marker_type in markers:
            if marker_type == "start":
                stack.append(pos)
            elif marker_type == "end" and stack:
                ranges.append((stack.pop(), pos))
        return ranges

    @staticmethod
    def _inside_mutable(span: tuple[int, int], mutable_ranges: Sequence[tuple[int, int]]) -> bool:
        return any(span[0] >= start and span[1] <= end for start, end in mutable_ranges)

    @staticmethod
    def strip_markers(text: str) -> str:
        lines = []
        for line in text.splitlines():
            if EVOLVE_START.match(line) or EVOLVE_END.match(line):
                continue
            lines.append(line)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _simple_diff(before: str, after: str) -> str:
        if before == after:
            return ""
        import difflib
        return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm=""))


class ShinkaPrompt:
    """Compatibility wrapper for code that expects a prompt helper class."""

    @staticmethod
    def compose(system: str, user: str) -> str:
        return f"{system.strip()}\n\n{user.strip()}"

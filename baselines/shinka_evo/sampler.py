from __future__ import annotations

import ast
import copy
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .edit.apply_full import apply_full_patch
from .edit.apply_diff import apply_diff_patch
from core.code import function_from_source

from core import Function, LLM, Program
from .population import ShinkaProgram
from .prompts import messages, prompt_record
from .prompts.prompts_base import BASE_SYSTEM_MSG, construct_eval_history_msg, perf_str, format_text_feedback_section
from .prompts.prompts_diff import DIFF_SYS_FORMAT, DIFF_ITER_MSG
from .prompts.prompts_full import FULL_SYS_FORMATS, FULL_ITER_MSG
from .prompts.prompts_cross import CROSS_SYS_FORMAT, CROSS_ITER_MSG, get_cross_component
from .prompts.prompts_fix import FIX_SYS_FORMAT, FIX_ITER_MSG, format_error_output_section
from core.code import replace_function


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
    BASE_SYSTEM_MSG = 'You are an expert optimization and algorithm design assistant. Improve the program while preserving correctness and immutable regions.'

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
        system = self.task_sys_msg + '\n\n' + self.task_description
        if meta_recommendations not in (None, 'none') and patch_type != 'cross':
            system += ('\n\n# Potential Recommendations\n'
                       'The following are potential recommendations for the next program generation:\n'
                       f'\n{meta_recommendations}')
        record = prompt_record(parent, self.to_marked_program(parent.function))
        feedback = format_text_feedback_section(parent.text_feedback) if self.use_text_feedback else ''
        values = dict(language='python', code_content=record.code,
                      performance_metrics=perf_str(record.combined_score, record.public_metrics),
                      text_feedback_section=feedback)
        if fix_mode:
            system += FIX_SYS_FORMAT
            values['error_output_section'] = format_error_output_section(
                stderr_log=str(parent.metadata.get('error', parent.text_feedback or '')))
            user = FIX_ITER_MSG.format(**values)
        elif patch_type == 'diff':
            system += DIFF_SYS_FORMAT
            user = DIFF_ITER_MSG.format(**values)
        elif patch_type == 'full':
            import numpy as np
            system += FULL_SYS_FORMATS[np.random.randint(len(FULL_SYS_FORMATS))]
            user = FULL_ITER_MSG.format(**values)
        elif patch_type == 'cross':
            system += CROSS_SYS_FORMAT
            user = CROSS_ITER_MSG.format(**values)
            user += '\n\n' + get_cross_component(
                [prompt_record(p) for p in archive_inspirations],
                [prompt_record(p) for p in top_k_inspirations])
        else:
            raise ValueError(f'Unsupported patch type: {patch_type}')
        inspirations = list(archive_inspirations) + list(top_k_inspirations)
        if self.inspiration_sort_order == 'ascending':
            inspirations.sort(key=lambda p: p.combined_score, reverse=True)
        elif self.inspiration_sort_order == 'chronological':
            inspirations.sort(key=lambda p: p.timestamp)
        history = construct_eval_history_msg([prompt_record(p) for p in inspirations],
                    include_text_feedback=self.use_text_feedback) if inspirations else ''
        if previous_error:
            user += '\n\n# Previous Patch Error\n' + previous_error
        return messages(system, history + '\n' + user)

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
            inspirations = sorted(inspirations, key=lambda p: p.combined_score, reverse=True)
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
        program = replace_function(function, self.template_program)
        if program is None:
            return str(function)
        return self._insert_markers(str(program))

    def _insert_markers(self, program_str: str) -> str:
        # A complete-solver task permits edits to helpers and imports too.
        if self.template_function.name == 'solve':
            return '# EVOLVE-BLOCK-START\n' + program_str + '\n# EVOLVE-BLOCK-END'
        function = function_from_source(program_str, self.template_function.name)
        if function is None:
            return program_str
        function.body = '\n'.join(['    # EVOLVE-BLOCK-START', function.body, '    # EVOLVE-BLOCK-END'])
        return str(function)

    def _apply_patch(self, response, parent, patch_type, apply):
        updated, count, _, error, _, _ = apply(
            response, original_str=self.to_marked_program(parent.function),
            language='python', verbose=False)
        if error or count == 0:
            return PatchResult(None, None, error=error or 'No patch applied.')
        function = function_from_source(self.strip_markers(updated), self.template_function.name)
        if function is None:
            return PatchResult(None, None, error='Patch does not contain a valid task entrypoint.')
        program = replace_function(function, self.template_program)
        return PatchResult(function, program,
                           code_diff=self._simple_diff(parent.program, str(program)),
                           metadata={'patch_type': patch_type})

    def _apply_full_response(self, response, parent, *, patch_type):
        return self._apply_patch(response, parent, patch_type, apply_full_patch)

    def _apply_diff_response(self, response, parent):
        return self._apply_patch(response, parent, 'diff', apply_diff_patch)

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


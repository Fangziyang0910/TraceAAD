# Module Name: FunSearch
#
# Reference:
#   - Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog,
#       M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, et al.
#       "Mathematical discoveries from program search with large language models."
#       Nature 625, no. 7995 (2024): 468-475.
#   - Reference implementation: https://github.com/google-deepmind/funsearch (Apache 2.0).
"""FunSearch: an island model over score clusters, prompted with versioned programs.

The search loop follows ``google-deepmind/funsearch`` (``sampler.py``,
``evaluator.py``, ``programs_database.py``):

1. The template program is evaluated and registered in every island.
2. A prompt is built from one random island: up to two programs from
   score-weighted clusters, sorted from worse to better and renamed
   ``f_v0``, ``f_v1``, followed by the empty header of ``f_v2``.
3. Each prompt is sampled ``samples_per_prompt`` times. Each sample becomes a
   full program by taking the generated body; a body that calls an earlier
   version ``f_v*`` is rejected.
4. Valid programs go back into the island that produced the prompt. Every
   ``reset_period`` seconds the weaker half of the islands is reset.

Two adaptations for an instruction-tuned chat model, both outside the
evolutionary algorithm: the prompt ends with one sentence asking for the
code of the last function only, and the task description is the module
docstring of the prompted program (the original specifications describe the
problem there). The response is parsed for the ``f_v{k}`` definition.

Scores: tasks here return a minimized fitness. The database keeps the
original higher-is-better convention and receives the negated fitness.
"""

from __future__ import annotations

import re
import textwrap
import time
import traceback

from baselines.observability import (
    close_llm,
    finish_profiler,
    init_observability,
    is_search_aborted,
    log_event,
    log_llm_call,
    log_state,
    record_sample_failure,
    reset_sample_failures,
)
from baselines.profiler import ProfilerBase
from baselines.sampling import SampleTrimmer
from core import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter

from . import code_manipulation
from .config import ProgramsDatabaseConfig
from .programs_database import ProgramsDatabase

OPERATOR = 'funsearch'
_FENCE = re.compile(r'```[ \t]*(?:python|py)?[ \t]*\n(.*?)```', re.S)


def chat_instruction(function_name: str, version: int) -> str:
    return (f'Complete the function `{function_name}_v{version}` above. '
            f'Only output the Python code of `{function_name}_v{version}`, no descriptions.')


class FunSearch:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase | None = None,
                 max_sample_nums: int | None = 1000,
                 samples_per_prompt: int = 4,
                 database_config: ProgramsDatabaseConfig | None = None,
                 *,
                 max_consecutive_sample_failures: int = 20,
                 debug_mode: bool = False,
                 **kwargs):
        """
        Args:
            llm: the model client.
            evaluation: the task; its score is minimized.
            profiler: records every evaluated program; ``None`` disables recording.
            max_sample_nums: budget counted in model samples, including samples
                whose code cannot be parsed. ``None`` disables the limit.
            samples_per_prompt: independent samples drawn for each prompt.
            database_config: programs database settings (paper defaults).
            **kwargs: passed to ``core.SecureEvaluator``.
        """
        self._llm = llm
        self._max_sample_nums = max_sample_nums
        self._samples_per_prompt = samples_per_prompt
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._config = database_config or ProgramsDatabaseConfig()

        self._template_program: Program = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
        self._function_to_evolve: Function = self._template_program.functions[0]
        self._function_name: str = self._function_to_evolve.name
        description = evaluation.task_description.strip().replace('"""', "'''")
        preface = self._template_program.preface.strip('\n')
        prompt_template = Program(preface=f'"""{description}"""\n\n{preface}\n' if preface else f'"""{description}"""\n',
                                  functions=[self._function_to_evolve])
        self._database = ProgramsDatabase(self._config, prompt_template, self._function_name)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler

        self._tot_sample_nums = 0
        init_observability(self, max_consecutive_sample_failures)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    # ------------------------------------------------------------------ parsing
    def _extract_code(self, response: str, version: int) -> str:
        """The fenced block that defines `f_v{version}`, else the first block, else the response."""
        blocks = _FENCE.findall(response)
        target = f'def {self._function_name}_v{version}'
        for block in blocks:
            if target in block:
                return block
        return blocks[0] if blocks else response

    def _body_after_header(self, code: str, version: int) -> str:
        """Text after the header of the generated function, dedented to function-body level."""
        lines = code.splitlines()
        patterns = (rf'^(\s*)def\s+{re.escape(self._function_name)}_v{version}\s*\(',
                    rf'^(\s*)def\s+{re.escape(self._function_name)}\s*\(',
                    r'^()def\s+\w+\s*\(')
        for pattern in patterns:
            for index, line in enumerate(lines):
                match = re.match(pattern, line)
                if not match:
                    continue
                indent = len(match.group(1))
                end = index
                while end < len(lines) and not lines[end].split('#')[0].rstrip().endswith(':'):
                    end += 1
                body = '\n'.join(item[indent:] if item[:indent].strip() == '' else item
                                 for item in lines[end + 1:])
                return body
        # A completion-style response continues the header directly.
        return code if code.startswith((' ', '\t')) else textwrap.indent(code, '    ')

    def sample_to_program(self, response: str, version: int) -> Program | None:
        """Returns the runnable program for one model response, or None if no function body is found."""
        body = self._body_after_header(self._extract_code(response, version), version)
        try:
            # A recursive generated function calls itself by its versioned name.
            body = code_manipulation.rename_function_calls(body, f'{self._function_name}_v{version}',
                                                           self._function_name)
        except Exception:
            pass
        return SampleTrimmer.sample_to_program(body, self._template_program)

    def calls_ancestor(self, program: Program) -> bool:
        """Whether the generated function calls an earlier version `f_v*` (rejected, as in the original)."""
        try:
            called = code_manipulation.get_functions_called(str(program))
        except Exception:
            return False
        return any(name.startswith(f'{self._function_name}_v') for name in called)

    # ------------------------------------------------------------------ search
    def _has_budget(self) -> bool:
        return not is_search_aborted(self) and (
            self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums)

    def _register(self, function: Function, program: Program, score, eval_time, sample_time,
                  island_id: int | None, operator: str) -> None:
        function.score = score
        function.evaluate_time = eval_time
        function.sample_time = sample_time
        function.operator = operator
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program))
        if score is not None:
            self._database.register_program(function, island_id, -float(score))

    def _sample_once(self, prompt) -> None:
        sample_order = self._tot_sample_nums + 1
        text = f'{prompt.code}\n{chat_instruction(self._function_name, prompt.version_generated)}'
        start = time.time()
        try:
            response = self._llm.draw_sample(text)
        except Exception as exc:
            record_sample_failure(self, exc, stage='sample', operator=OPERATOR, sample_order=sample_order,
                                  prompt=text, counts_budget=False, island_id=prompt.island_id)
            return
        sample_time = time.time() - start
        reset_sample_failures(self)

        program = self.sample_to_program(response, prompt.version_generated)
        log_llm_call(self, stage='generate', operator=OPERATOR, sample_order=sample_order, prompt=text,
                     response=response, island_id=prompt.island_id,
                     version_generated=prompt.version_generated, function_parse_success=program is not None)
        self._tot_sample_nums += 1
        if program is None:
            log_event(self, event='sample_rejected', status='parse_failed', operator=OPERATOR,
                      sample_order=sample_order, island_id=prompt.island_id, counts_budget=True)
            return

        function = TextFunctionProgramConverter.program_to_function(program)
        if self.calls_ancestor(program):
            score, eval_time, status = None, 0.0, 'calls_ancestor'
        else:
            score, eval_time = self._evaluator.evaluate_program_record_time(program)
            status = 'registered' if score is not None else 'invalid'
        self._register(function, program, score, eval_time, sample_time, prompt.island_id, OPERATOR)
        log_event(self, event='database_register', status=status, operator=OPERATOR, sample_order=sample_order,
                  island_id=prompt.island_id, version_generated=prompt.version_generated, score=score,
                  counts_budget=True)
        log_state(self, phase='programs_database', method=OPERATOR, sample_count=self._tot_sample_nums,
                  island_program_counts=[island.num_programs for island in self._database.islands],
                  island_cluster_counts=[len(island.clusters) for island in self._database.islands],
                  island_best_scores=[-s if s != -float('inf') else None
                                      for s in self._database.best_score_per_island],
                  island_resets=self._database.num_resets)

    def run(self) -> None:
        try:
            # The template must run: it seeds every island.
            score, eval_time = self._evaluator.evaluate_program_record_time(self._template_program)
            if score is None:
                raise RuntimeError('The score of the template program must not be None.')
            seed = TextFunctionProgramConverter.program_to_function(self._template_program)
            self._register(seed, self._template_program, score, eval_time, 0.0, None, 'seed')

            while self._has_budget():
                prompt = self._database.get_prompt()
                log_event(self, event='prompt_sampled', status='scheduled', operator=OPERATOR,
                          island_id=prompt.island_id, version_generated=prompt.version_generated,
                          sample_order=self._tot_sample_nums + 1)
                for _ in range(self._samples_per_prompt):
                    if not self._has_budget():
                        break
                    try:
                        self._sample_once(prompt)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        if self._debug_mode:
                            traceback.print_exc()
                            raise
                        record_sample_failure(self, exc, stage='evaluate', operator=OPERATOR,
                                              island_id=prompt.island_id, counts_budget=False)
        except KeyboardInterrupt:
            pass
        finally:
            finish_profiler(self, status='aborted' if is_search_aborted(self) else 'finished',
                            island_resets=self._database.num_resets)
            close_llm(self._llm)

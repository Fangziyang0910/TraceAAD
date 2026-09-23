# Module Name: EoH
# Last Revision: 2025/2/16
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Fei Liu, Tong Xialiang, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang.
#       "Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model."
#       In Forty-first International Conference on Machine Learning (ICML). 2024.
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

from __future__ import annotations

import concurrent.futures
import random
import time
import traceback
from threading import Lock, Thread
from typing import Optional, Literal

from .population import Population
from .profiler import EoHProfiler
from .prompt import EoHPrompt
from .sampler import EoHSampler
from .observability import (
    call_sampler_get_thought_and_function,
    close_sampler_llm,
    finish_profiler,
    init_observability,
    is_search_aborted,
    log_event,
    log_state,
    record_sample_failure,
    reset_sample_failures,
    shutdown_executor,
)
from llm4ad.base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from llm4ad.tools.profiler import ProfilerBase


class EoH:
    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: Optional[int] = 10,
                 max_sample_nums: Optional[int] = 100,
                 pop_size: Optional[int] = 5,
                 selection_num=2,
                 use_e2_operator: bool = True,
                 use_m1_operator: bool = True,
                 use_m2_operator: bool = True,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 *,
                 use_m3_operator: bool = False,
                 operators: Optional[list[str]] = None,
                 operator_weights: Optional[list[float]] = None,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 max_consecutive_sample_failures: int = 20,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 **kwargs):
        """Evolutionary of Heuristics.
        Args:
            llm             : an instance of 'llm4ad.base.LLM', which provides the way to query LLM.
            evaluation      : an instance of 'llm4ad.base.Evaluator', which defines the way to calculate the score of a generated function.
            profiler        : an instance of 'baselines.eoh.EoHProfiler'. If you do not want to use it, you can pass a 'None'.
            max_generations : terminate after evolving 'max_generations' generations when 'max_sample_nums' is None,
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums valid generated functions,
                              pass 'None' to disable this termination condition.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            use_e2_operator : if use e2 operator.
            use_m1_operator : if use m1 operator.
            use_m2_operator : if use m2 operator.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._use_e2_operator = use_e2_operator
        self._use_m1_operator = use_m1_operator
        self._use_m2_operator = use_m2_operator
        self._use_m3_operator = use_m3_operator
        self._operators, self._operator_weights = self._build_operator_config(
            operators,
            operator_weights,
        )

        # samplers and evaluators
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        llm.debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        # function to be evolved
        self._function_to_evolve: Function = TextFunctionProgramConverter.text_to_function(self._template_program_str)
        self._function_to_evolve_name: str = self._function_to_evolve.name
        self._template_program: Program = TextFunctionProgramConverter.text_to_program(self._template_program_str)

        # adjust population size
        self._adjust_pop_size()

        # population, sampler, and evaluator
        self._population = Population(pop_size=self._pop_size)
        self._profiler = profiler
        self._sampler = EoHSampler(llm, self._template_program_str, profiler=self._profiler)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)

        # statistics
        self._tot_sample_nums = 0
        self._sample_lock = Lock()
        init_observability(self, max_consecutive_sample_failures)

        # reset _initial_sample_nums_max
        self._initial_sample_nums_max = 2 * self._pop_size

        # multi-thread executor for evaluation
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # pass parameters to profiler
        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)  # ZL: necessary

    def _build_operator_config(self, operators, operator_weights):
        valid_operators = {"e1", "e2", "m1", "m2", "m3"}
        if operators is None:
            operators = ["e1"]
            if self._use_e2_operator:
                operators.append("e2")
            if self._use_m1_operator:
                operators.append("m1")
            if self._use_m2_operator:
                operators.append("m2")
            if self._use_m3_operator:
                operators.append("m3")
        else:
            operators = list(operators)

        unknown = [op for op in operators if op not in valid_operators]
        if unknown:
            raise ValueError(f"Unknown EoH operators: {unknown}")
        if not operators:
            raise ValueError("EoH requires at least one evolution operator.")

        if operator_weights is None or len(operator_weights) != len(operators):
            operator_weights = [1.0] * len(operators)
        else:
            operator_weights = list(operator_weights)
        return operators, operator_weights

    def _adjust_pop_size(self):
        # adjust population size
        if self._max_sample_nums is None:
            if self._pop_size is None:
                self._pop_size = 5
        elif self._max_sample_nums >= 10000:
            if self._pop_size is None:
                self._pop_size = 40
            elif abs(self._pop_size - 40) > 20:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 40.')
        elif self._max_sample_nums >= 1000:
            if self._pop_size is None:
                self._pop_size = 20
            elif abs(self._pop_size - 20) > 10:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 20.')
        elif self._max_sample_nums >= 200:
            if self._pop_size is None:
                self._pop_size = 10
            elif abs(self._pop_size - 10) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 10.')
        else:
            if self._pop_size is None:
                self._pop_size = 5
            elif abs(self._pop_size - 5) > 5:
                print(f'Warning: population size {self._pop_size} '
                      f'is not suitable, please reset it to 5.')

    def _sample_evaluate_register(self, prompt, *, register=True, count_sample=True, operator=None):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        sample_order = self._tot_sample_nums + 1
        try:
            thought, func = call_sampler_get_thought_and_function(
                self._sampler,
                prompt,
                operator=operator,
                sample_order=sample_order,
            )
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage='sample',
                operator=operator,
                sample_order=sample_order,
                prompt=prompt,
                counts_budget=False,
            )
            return None
        sample_time = time.time() - sample_start
        reset_sample_failures(self)
        if thought is None or func is None:
            log_event(
                self,
                event='sample_rejected',
                status='parse_failed',
                operator=operator,
                sample_order=sample_order,
                counts_budget=False,
            )
            return
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            log_event(
                self,
                event='sample_rejected',
                status='program_parse_failed',
                operator=operator,
                sample_order=sample_order,
                counts_budget=False,
            )
            return
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        if not Population._is_valid_score(score):
            return None
        # register to profiler
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator or 'Unknown'
        should_advance_generation = False
        if count_sample:
            sample_order = self._try_increment_sample_count()
            if sample_order is None:
                return None
            should_advance_generation = sample_order % self._pop_size == 0

        accepted = True
        if register:
            accepted = self._population.register_function(func, increment_generation=False)
            if should_advance_generation:
                self._population.advance_generation()

        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, EoHProfiler):
                self._profiler.register_population(self._population)
        log_event(
            self,
            event='sample_registered',
            status='accepted' if accepted else 'rejected',
            operator=operator,
            sample_order=self._tot_sample_nums,
            score=func.score,
            generation=self._population.generation,
            counts_budget=count_sample,
        )
        log_state(
            self,
            phase='population',
            method='eoh',
            generation=self._population.generation,
            population_size=len(self._population),
            sample_count=self._tot_sample_nums,
        )

        return func if accepted else None

    def _sample_budget(self):
        if self._max_sample_nums is not None:
            return self._max_sample_nums
        if self._max_generations is not None:
            return self._max_generations * self._pop_size
        return None

    def _try_increment_sample_count(self) -> int | None:
        lock = getattr(self, "_sample_lock", None)
        if lock is not None:
            lock.acquire()
        try:
            budget = self._sample_budget()
            if budget is not None and self._tot_sample_nums >= budget:
                return None
            self._tot_sample_nums += 1
            return self._tot_sample_nums
        finally:
            if lock is not None:
                lock.release()

    def _continue_loop(self) -> bool:
        if is_search_aborted(self):
            return False
        budget = self._sample_budget()
        if budget is None:
            return True
        return self._tot_sample_nums < budget

    def _select_operator(self):
        return random.choices(self._operators, weights=self._operator_weights, k=1)[0]

    def _build_operator_prompt(self, operator: str):
        if operator == "e1":
            indivs = [self._population.selection() for _ in range(self._selection_num)]
            return EoHPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
        if operator == "e2":
            indivs = [self._population.selection() for _ in range(self._selection_num)]
            return EoHPrompt.get_prompt_e2(self._task_description_str, indivs, self._function_to_evolve)
        if operator == "m1":
            indiv = self._population.selection()
            return EoHPrompt.get_prompt_m1(self._task_description_str, indiv, self._function_to_evolve)
        if operator == "m2":
            indiv = self._population.selection()
            return EoHPrompt.get_prompt_m2(self._task_description_str, indiv, self._function_to_evolve)
        if operator == "m3":
            indiv = self._population.selection()
            return EoHPrompt.get_prompt_m3(self._task_description_str, indiv, self._function_to_evolve)
        raise ValueError(f"Unknown EoH operator: {operator}")

    def _run_one_evolution_sample(self) -> bool:
        operator = self._select_operator()
        prompt = self._build_operator_prompt(operator)
        if self._debug_mode:
            print(f'{operator.upper()} Prompt: {prompt}')
        log_event(
            self,
            event='operator_start',
            status='scheduled',
            operator=operator,
            sample_order=self._tot_sample_nums,
            generation=self._population.generation,
        )
        return self._sample_evaluate_register(
            prompt,
            register=True,
            count_sample=True,
            operator=operator,
        ) is not None

    def _iteratively_use_eoh_operator(self):
        while self._continue_loop():
            try:
                self._run_one_evolution_sample()
            except KeyboardInterrupt:
                break
            except Exception as e:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        # shutdown evaluation_executor
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except:
            pass

    def _initialize_population(self):
        """Generate a fixed 2 * pop_size i1 candidate pool, then keep the top pop_size."""
        raw_population = []
        initial_sample_nums_max = getattr(self, "_initial_sample_nums_max", 2 * self._pop_size)
        for _ in range(initial_sample_nums_max):
            try:
                prompt = EoHPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, register=False, count_sample=False, operator='i1')
                if func is not None:
                    raw_population.append(func)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue
        self._population = Population(pop_size=self._pop_size)
        self._population.survival(raw_population, increment_generation=False)
        if self._profiler is not None and isinstance(self._profiler, EoHProfiler):
            self._profiler.register_population(self._population)
        if len(self._population) < self._pop_size:
            print(
                f'Note: During initialization, EoH gets {len(self._population)} algorithms '
                f'after {initial_sample_nums_max} trails.')

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In EoH, `fn` can be `self._iteratively_init_population` or `self._iteratively_use_eoh_operator`.
        """
        # threads for sampling
        sampler_threads = [
            Thread(target=fn, args=args, kwargs=kwargs)
            for _ in range(self._num_samplers)
        ]
        for t in sampler_threads:
            t.start()
        for t in sampler_threads:
            t.join()

    def run(self):
        try:
            if not self._resume_mode:
                # do initialization
                self._initialize_population()
                # terminate searching if
                if len(self._population) == 0:
                    print(
                        f'The search is terminated since EoH unable to obtain any feasible algorithm during initialization. '
                        f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                        f'Please also check your evaluation implementation and LLM implementation.')
                    return

            # evolutionary search
            self._multi_threaded_sampling(self._iteratively_use_eoh_operator)
        finally:
            finish_profiler(
                self,
                status='aborted' if is_search_aborted(self) else 'finished',
            )
            close_sampler_llm(self._sampler)
            shutdown_executor(self._evaluation_executor)

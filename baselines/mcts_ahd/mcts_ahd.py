# Module Name: MCTS_AHD
# Last Revision: 2025/7/22
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Zheng, Z., Xie, Z., Wang, Z., & Hooi, B. (2025). Monte carlo tree search for
#       comprehensive exploration in llm-based automatic heuristic design. (ICML). 2024.
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
import heapq
import concurrent.futures
import copy
import random
import time
import traceback
from threading import Thread
from typing import Optional, Literal

from .population import Population
from .mcts import MCTS, MCTSNode
from .profiler import MAProfiler
from .prompt import MAPrompt
from .sampler import MASampler
from .observability import close_sampler_llm, finish_profiler, shutdown_executor
from llm4ad.base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from llm4ad.tools.profiler import ProfilerBase


class MCTS_AHD:
    E1_MIN_REFS = 2
    E1_MAX_REFS = 5
    DEFAULT_OPERATORS = ('e1', 'e2', 'm1', 'm2', 's1')
    DEFAULT_OPERATOR_WEIGHTS = (0, 1, 2, 2, 1)

    def __init__(self,
                 llm: LLM,
                 evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_sample_nums: Optional[int] = 1000,
                 init_size: Optional[float] = 4,
                 pop_size: Optional[int] = 10,
                 selection_num: int = 2,
                 num_samplers: int = 1,
                 num_evaluators: int = 1,
                 alpha: float = 0.5,
                 lambda_0: float = 0.1,
                 *,
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
                              pass 'None' to disable this termination condition.
            max_sample_nums : terminate after evaluating max_sample_nums functions (no matter the function is valid or not) or reach 'max_generations',
                              pass 'None' to disable this termination condition.
            init_size       : population size, if set to 'None', EoH will automatically adjust this parameter.
            pop_size        : population size, if set to 'None', EoH will automatically adjust this parameter.
            selection_num   : number of selected individuals while crossover.
            alpha           : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            lambda_0        : a parameter for the UCT formula, which is used to balance exploration and exploitation.
            resume_mode     : in resume_mode, randsample will not evaluate the template_program, and will skip the init process. TODO: More detailed usage.
            debug_mode      : if set to True, we will print detailed information.
            max_consecutive_sample_failures: stop the search after this many consecutive LLM/sample failures.
            multi_thread_or_process_eval: use 'concurrent.futures.ThreadPoolExecutor' or 'concurrent.futures.ProcessPoolExecutor' for the usage of
                multi-core CPU while evaluation. Please note that both settings can leverage multi-core CPU. As a result on my personal computer (Mac OS, Intel chip),
                setting this parameter to 'process' will faster than 'thread'. However, I do not sure if this happens on all platform so I set the default to 'thread'.
                Please note that there is one case that cannot utilize multi-core CPU: if you set 'safe_evaluate' argument in 'evaluator' to 'False',
                and you set this argument to 'thread'.
            **kwargs                    : some args pass to 'llm4ad.base.SecureEvaluator'. Such as 'fork_proc'.
        """
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self.lambda_0 = lambda_0
        self.alpha = alpha
        self._init_pop_size = init_size
        self._pop_size = pop_size
        self._selection_num = selection_num
        self._e1_min_refs = self.E1_MIN_REFS
        self._e1_max_refs = self.E1_MAX_REFS
        self._operators = list(self.DEFAULT_OPERATORS)
        self._operator_weights = list(self.DEFAULT_OPERATOR_WEIGHTS)

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
        self._population = Population(init_pop_size=init_size, pop_size=self._pop_size)
        self._profiler = profiler
        self._sampler = MASampler(llm, self._template_program_str, profiler=self._profiler)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)

        # statistics
        self._tot_sample_nums = 0
        self._consecutive_sample_failures = 0
        self._max_consecutive_sample_failures = max(1, max_consecutive_sample_failures)
        self._search_aborted = False

        # reset _initial_sample_nums_max
        if self._max_sample_nums is None:
            self._initial_sample_nums_max = 10 * self._init_pop_size
        else:
            self._initial_sample_nums_max = min(
                self._max_sample_nums,
                10 * self._init_pop_size
            )

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

    def _log_message(self, message: str):
        if isinstance(self._profiler, MAProfiler):
            self._profiler.log_message(message)
        else:
            print(message)

    def _log_mcts_state(self, mcts: MCTS, phase: str, selected_node: MCTSNode | None = None):
        if isinstance(self._profiler, MAProfiler):
            self._profiler.log_mcts_state(
                phase=phase,
                sample_order=self._tot_sample_nums,
                max_sample_nums=self._max_sample_nums,
                mcts=mcts,
                selected_node=selected_node,
            )

    def _log_mcts_event(self, **payload):
        if isinstance(self._profiler, MAProfiler):
            self._profiler.log_mcts_event(**payload)

    def _log_llm_call(self, **payload):
        logger = getattr(self._profiler, 'log_llm_call', None)
        if callable(logger):
            try:
                logger(**payload)
            except Exception:
                pass

    @staticmethod
    def _node_score(node: MCTSNode):
        raw_info = getattr(node, 'raw_info', None)
        if raw_info is not None:
            score = getattr(raw_info, 'score', None)
            if score is not None:
                return score
        individual = getattr(node, 'individual', None)
        if individual is not None:
            score = getattr(individual, 'score', None)
            if score is not None:
                return score
        return getattr(node, 'Q', None)

    def _adjust_pop_size(self):
        # adjust population size
        if self._pop_size is None:
            self._pop_size = 10
            return
        if self._max_sample_nums is None:
            return
        if self._max_sample_nums >= 10000:
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

    def _sample_evaluate_register(self, prompt, func_only=False, operator=None):
        """Perform following steps:
        1. Sample an algorithm using the given prompt.
        2. Evaluate it by submitting to the process/thread pool, and get the results.
        3. Add the function to the population and register it to the profiler.
        """
        sample_start = time.time()
        sample_order = self._tot_sample_nums + 1
        try:
            thought, func = self._sampler.get_thought_and_function(
                self._task_description_str,
                prompt,
                operator=operator,
                sample_order=sample_order,
            )
        except Exception as exc:
            if getattr(self, '_debug_mode', False):
                raise
            self._record_sample_failure(
                exc,
                prompt=prompt,
                operator=operator,
                sample_order=sample_order,
            )
            return False
        sample_time = time.time() - sample_start
        self._consecutive_sample_failures = 0
        if thought is None or func is None:
            return False
        # convert to Program instance
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            return False
        # evaluate
        score, eval_time = self._evaluation_executor.submit(
            self._evaluator.evaluate_program_record_time,
            program
        ).result()
        # register to profiler
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator or 'Unknown'
        self._tot_sample_nums += 1
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program))
            if isinstance(self._profiler, MAProfiler):
                self._profiler.register_population(self._population)
        if func_only:
            return func
        if func.score is None:
            return False
        # register to the population
        self._population.register_function(func)

        return True

    def _continue_loop(self) -> bool:
        if getattr(self, '_search_aborted', False):
            return False
        if self._max_sample_nums is None:
            return True
        else:
            return self._tot_sample_nums < self._max_sample_nums

    def _record_sample_failure(self, exc: Exception, *, prompt, operator=None, sample_order=None):
        if sample_order is None:
            sample_order = self._tot_sample_nums + 1
        self._consecutive_sample_failures = getattr(self, '_consecutive_sample_failures', 0) + 1
        max_failures = getattr(self, '_max_consecutive_sample_failures', 20)
        error_message = str(exc)
        if len(error_message) > 500:
            error_message = error_message[:497] + '...'

        self._log_llm_call(
            stage='sample_error',
            operator=operator,
            sample_order=sample_order,
            prompt=prompt,
            error_type=type(exc).__name__,
            error=error_message,
            consecutive_failures=self._consecutive_sample_failures,
        )
        self._log_mcts_event(
            event='sample_error',
            status='error',
            reason='llm_sample_exception',
            operator=operator,
            sample_order=self._tot_sample_nums,
            error_type=type(exc).__name__,
            error=error_message,
            consecutive_failures=self._consecutive_sample_failures,
            max_consecutive_failures=max_failures,
        )
        logger = getattr(self._profiler, 'log_error', None)
        if callable(logger):
            logger(
                'sample',
                exc,
                operator=operator,
                sample_order=sample_order,
                prompt=prompt,
                counts_budget=False,
                consecutive_failures=self._consecutive_sample_failures,
                max_consecutive_failures=max_failures,
            )

        if self._consecutive_sample_failures >= max_failures:
            self._search_aborted = True
            self._log_message(
                f'MCTS_AHD stops after {self._consecutive_sample_failures} consecutive sample failures. '
                f'Last error: {type(exc).__name__}: {error_message}'
            )

    def check_duplicate(self, population, code):
        for ind in population:
            ind_code = ind.code if isinstance(ind, MCTSNode) else str(ind)
            if code == ind_code:
                return True
        return False

    def check_duplicate_obj(self, population, score):
        for ind in population:
            if isinstance(ind, MCTSNode):
                ind_score = ind.individual.score if ind.individual is not None else ind.Q
            else:
                ind_score = ind.score
            if score == ind_score:
                return True
        return False

    def _eval_remain_ratio(self) -> float:
        if self._max_sample_nums is None or self._max_sample_nums <= 0:
            return 1.0
        return max(1 - self._tot_sample_nums / self._max_sample_nums, 0)

    def _sample_e1_references_from_population(self, population, allow_single=False):
        candidates = list(population)
        if len(candidates) < self._e1_min_refs:
            return copy.deepcopy(candidates) if allow_single else []
        ref_count = random.randint(self._e1_min_refs, min(self._e1_max_refs, len(candidates)))
        return copy.deepcopy(random.choices(candidates, k=ref_count))

    def _sample_e1_references_from_root(self, mcts: MCTS, allow_single=False):
        candidates = [
            random.choice(child.subtree).individual
            for child in mcts.root.children
            if len(child.subtree) > 0
        ]
        if len(candidates) < self._e1_min_refs:
            if not allow_single:
                return []
            return copy.deepcopy(candidates)
        ref_count = random.randint(self._e1_min_refs, min(self._e1_max_refs, len(candidates)))
        return copy.deepcopy(random.choices(candidates, k=ref_count))

    @staticmethod
    def _individual_from_entry(entry):
        if isinstance(entry, MCTSNode):
            return entry.individual
        return entry

    def population_management(self, pop_input, size):
        pop = []
        for entry in pop_input:
            individual = self._individual_from_entry(entry)
            if individual is not None and individual.score is not None:
                pop.append(individual)
        if size > len(pop):
            size = len(pop)

        unique_pop = []
        unique_scores = []
        for individual in pop:
            if individual.score not in unique_scores:
                unique_pop.append(individual)
                unique_scores.append(individual.score)

        return heapq.nlargest(size, unique_pop, key=lambda x: x.score)

    def _current_elite_set(self):
        candidates = list(self._population.population) + list(self._population.next_gen_pop)
        unique_pop = []
        unique_scores = []
        for individual in candidates:
            if individual.score is None or individual.score == float('-inf'):
                continue
            if individual.score not in unique_scores:
                unique_pop.append(individual)
                unique_scores.append(individual.score)

        pop_size = getattr(self, '_pop_size', None)
        if pop_size is None:
            pop_size = getattr(self._population, '_pop_size', len(unique_pop))
        return heapq.nlargest(pop_size, unique_pop, key=lambda x: x.score)

    def _should_progressively_widen(self, mcts: MCTS, node: MCTSNode) -> bool:
        return int(node.visits ** mcts.alpha) > len(node.children)

    def population_management_s1(self, pop_input, size):
        pop = []
        for entry in pop_input:
            individual = self._individual_from_entry(entry)
            if individual is not None and individual.score is not None:
                pop.append(individual)
        if size > len(pop):
            size = len(pop)

        unique_pop = []
        unique_algorithms = []
        for individual in pop:
            algorithm = getattr(individual, 'algorithm', str(individual))
            if algorithm not in unique_algorithms:
                unique_pop.append(individual)
                unique_algorithms.append(algorithm)

        # The reference code orders s1 paths with nlargest(objective). LLM4AD
        # scores are sign-flipped objectives, so the equivalent order is
        # nsmallest(score).
        return heapq.nsmallest(size, unique_pop, key=lambda x: x.score)

    def _select_e2_reference(self, node_set, father):
        candidates = []
        for entry in node_set:
            individual = self._individual_from_entry(entry)
            if individual is not None and individual.score is not None and individual.score != float('-inf'):
                candidates.append(individual)
        if not candidates:
            candidates = self._current_elite_set()

        other = [individual for individual in candidates if individual != father]
        if len(other) == 0:
            return None

        other = sorted(other, key=lambda f: f.score, reverse=True)
        probs = [1 / (rank + 1 + len(other)) for rank in range(len(other))]
        return random.choices(other, weights=probs, k=1)[0]

    def _add_root_child(self, mcts: MCTS, func: Function):
        now_node = MCTSNode(func.algorithm, str(func), -1 * func.score, individual=func,
                            parent=mcts.root, depth=1, visit=1, Q=func.score, raw_info=func)
        mcts.root.add_child(now_node)
        mcts.backpropagate(now_node)
        now_node.subtree.append(now_node)
        return now_node

    def expand(self, mcts: MCTS, node_set, cur_node: MCTSNode, option: str):
        if getattr(self, '_search_aborted', False):
            return node_set
        is_valid_func = True
        if option == 's1':
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.algorithm != "Root":
                path_set.append(now.individual)
                now = copy.deepcopy(now.parent)
            path_set = self.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return node_set

            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_s1(self._task_description_str, path_set, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(path_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'e1':
            indivs = self._sample_e1_references_from_root(mcts, allow_single=True)
            if len(indivs) == 0:
                return node_set
            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'e2':
            i = 0
            while i < 3:
                now_indiv = self._select_e2_reference(node_set, cur_node.individual)
                if now_indiv is None:
                    return node_set
                prompt = MAPrompt.get_prompt_e2(self._task_description_str, [now_indiv, cur_node.individual],
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm1':
            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_m1(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        elif option == 'm2':
            i = 0
            while i < 3:
                prompt = MAPrompt.get_prompt_m2(self._task_description_str, cur_node.individual,
                                                self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator=option)
                if func is False:
                    is_valid_func = False
                    i += 1
                    continue
                is_valid_func = (func.score is not None) and not self.check_duplicate(node_set, str(func))
                if is_valid_func is False:
                    i += 1
                    continue
                else:
                    break

        else:
            assert False, 'Invalid option!'

        if not is_valid_func:
            self._log_mcts_event(
                event='expand',
                status='invalid',
                reason='timeout_or_invalid_function',
                operator=option,
                sample_order=self._tot_sample_nums,
                parent_score=self._node_score(cur_node),
                parent_depth=cur_node.depth,
                parent_visits=cur_node.visits,
            )
            return node_set

        if option != 'e1':
            parent_score = self._node_score(cur_node)
        else:
            if self.check_duplicate_obj(mcts.root.children, func.score):
                self._log_mcts_event(
                    event='expand',
                    status='duplicate',
                    reason='duplicate_e1_objective',
                    operator=option,
                    sample_order=self._tot_sample_nums,
                    parent_score=None,
                    child_score=func.score,
                    parent_depth=cur_node.depth,
                    parent_visits=cur_node.visits,
                )
                return node_set
            parent_score = None

        if is_valid_func and func.score != float('-inf'):
            self._population.register_function(func)
            now_node = MCTSNode(func.algorithm, str(func), -1 * func.score, individual=func,
                                parent=cur_node, depth=cur_node.depth + 1, visit=1, Q=func.score, raw_info=func)
            if option == 'e1':
                now_node.subtree.append(now_node)
            cur_node.add_child(now_node)
            mcts.backpropagate(now_node)
            if node_set is not cur_node.children:
                node_set.append(func)
                size_act = min(len(node_set), self._pop_size)
                node_set = self.population_management(node_set, size_act)
            self._log_mcts_event(
                event='expand',
                status='expanded',
                operator=option,
                sample_order=self._tot_sample_nums,
                parent_score=parent_score,
                child_score=func.score,
                parent_depth=cur_node.depth,
                child_depth=now_node.depth,
                parent_visits=cur_node.visits,
                child_visits=now_node.visits,
                parent_children_count=len(cur_node.children),
                root_parent=cur_node.algorithm == 'Root',
            )
        return node_set

    def _initialize_mcts_root(self, mcts: MCTS):
        brothers = []
        target_size = int(self._init_pop_size)

        while len(brothers) == 0 and self._continue_loop():
            try:
                prompt = MAPrompt.get_prompt_i1(self._task_description_str, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator='i1')
                if func is False or func.score is None or func.score == float('-inf'):
                    continue
                brothers.append(func)
                self._population.register_function(func)
                self._add_root_child(mcts, func)
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        while len(brothers) < target_size and self._continue_loop() and not getattr(self, '_search_aborted', False):
            try:
                indivs = self._sample_e1_references_from_population(brothers, allow_single=True)
                if len(indivs) == 0:
                    continue
                prompt = MAPrompt.get_prompt_e1(self._task_description_str, indivs, self._function_to_evolve)
                func = self._sample_evaluate_register(prompt, func_only=True, operator='e1')
                if func is False or func.score is None or func.score == float('-inf'):
                    continue
                if self.check_duplicate_obj(brothers, func.score) or self.check_duplicate(brothers, str(func)):
                    continue
                brothers.append(func)
                self._population.register_function(func)
                self._add_root_child(mcts, func)

                if self._tot_sample_nums >= self._initial_sample_nums_max:
                    self._log_message(
                        f'Note: During initialization, MCTS_AHD gets {len(brothers)} algorithms '
                        f'after {self._initial_sample_nums_max} trails.')
                    break
            except Exception:
                if self._debug_mode:
                    traceback.print_exc()
                    exit()
                continue

        self._population.survival()
        return brothers

    def _multi_threaded_sampling(self, fn: callable, *args, **kwargs):
        """Execute `fn` using multithreading.
        In MCTS_MA, `fn` can be `self._iteratively_use_eoh_operator`.
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
        run_status = 'finished'
        run_error = None
        try:
            mcts = MCTS('Root', self.alpha, self.lambda_0)
            brothers = self._initialize_mcts_root(mcts)
            nodes_set = self.population_management(brothers, min(len(brothers), self._pop_size))

            # terminate searching if
            if len(brothers) < self._selection_num:
                self._log_message(
                    f'The search is terminated since MCTS_AHD unable to obtain {self._selection_num} feasible algorithms during initialization. '
                    f'Please increase the `initial_sample_nums_max` argument (currently {self._initial_sample_nums_max}). '
                    f'Please also check your evaluation implementation and LLM implementation.')
                return

            # evolutionary search
            while self._continue_loop():
                self._log_mcts_state(mcts, phase='iteration_start')
                cur_node = mcts.root
                while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth and self._continue_loop():
                    uct_scores = [mcts.uct(node, self._eval_remain_ratio()) for node in
                                  cur_node.children]
                    selected_pair_idx = uct_scores.index(max(uct_scores))
                    if self._should_progressively_widen(mcts, cur_node):
                        if cur_node == mcts.root:
                            op = 'e1'
                            nodes_set = self.expand(mcts, nodes_set, cur_node, op)
                        else:
                            # i = random.randint(1, n_op - 1)
                            op = 'e2'
                            nodes_set = self.expand(mcts, nodes_set, cur_node, op)
                        if getattr(self, '_search_aborted', False):
                            break
                    cur_node = cur_node.children[selected_pair_idx]
                if getattr(self, '_search_aborted', False):
                    break
                self._log_mcts_state(mcts, phase='selected_leaf', selected_node=cur_node)
                for i in range(len(self._operators)):
                    if not self._continue_loop():
                        break
                    op = self._operators[i]
                    self._log_mcts_event(
                        event='operator_start',
                        status='scheduled',
                        operator=op,
                        sample_order=self._tot_sample_nums,
                        parent_score=self._node_score(cur_node),
                        parent_depth=cur_node.depth,
                        parent_visits=cur_node.visits,
                    )
                    op_w = self._operator_weights[i]
                    for j in range(op_w):
                        if not self._continue_loop():
                            break
                        nodes_set = self.expand(mcts, nodes_set, cur_node, op)
                self._population.survival()
                nodes_set = self.population_management(nodes_set, min(len(nodes_set), self._pop_size))
        except Exception as exc:
            run_status = 'error'
            run_error = exc
            logger = getattr(self._profiler, 'log_error', None)
            if callable(logger):
                logger('run', exc)
            raise
        finally:
            if getattr(self, '_search_aborted', False):
                run_status = 'aborted'
            summary_payload = {}
            if run_error is not None:
                summary_payload.update(
                    error_type=type(run_error).__name__,
                    error=str(run_error),
                )
            finish_profiler(
                self,
                status=run_status,
                **summary_payload,
            )
            close_sampler_llm(self._sampler)
            shutdown_executor(self._evaluation_executor)

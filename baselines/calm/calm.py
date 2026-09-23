"""CALM search framework without GRPO, adapted to LLM4AD abstractions.

Faithful port of reference_code/CALM verbal operators, pool/collapse,
numeric refine, and accept/reject logic. LLM backend uses ``LLM.draw_sample``;
evaluation uses ``SecureEvaluator``. Injection/simplification detectors match
current prompt text (upstream GitHub still matched obsolete wording).
"""

from __future__ import annotations

import inspect
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np

from baselines.observability import (
    init_observability,
    is_search_aborted,
    log_event,
    log_state,
    record_sample_failure,
    reset_sample_failures,
)
from core import (
    Evaluation,
    Function,
    LLM,
    SecureEvaluator,
    TextFunctionProgramConverter,
)
from baselines.sampling import SampleTrimmer
from .numeric_refine import code_signature, run_numeric_refinement
from .parse import (
    extract_first_double_braced,
    extract_function_from_string,
    extract_idea_description,
    get_code,
)
from .pool import HeuristicRecord
from .profiler import CALMProfiler
from .prompt import Prompt, PromptBuilder
from .sampler import CALMSampler
from .schedule import prepare_round_messages
from .seeds import load_seed
from .task_config import get_task_hyperparams, resolve_task_key


class CALM:
    """CALM (w/o GRPO): evolutionary heuristic search with verbal operators."""

    METHOD_LABEL = 'CALM (w/o GRPO)'

    def __init__(
            self,
            llm: LLM,
            evaluation: Evaluation,
            profiler: CALMProfiler | None = None,
            max_sample_nums: Optional[int] = 1000,
            *,
            seed: int = 0,
            task_key: Optional[str] = None,
            num_samplers: int = 1,
            num_evaluators: int = 1,
            max_consecutive_sample_failures: int = 20,
            debug_mode: bool = False,
            **hyperparam_overrides,
    ):
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode)
        self._llm = llm
        self._profiler = profiler
        self._max_sample_nums = max_sample_nums
        self._debug_mode = debug_mode
        self._tot_sample_nums = 0
        self._num_samplers = max(1, int(num_samplers))
        self._num_evaluators = max(1, int(num_evaluators))
        self._evaluation_executor = ThreadPoolExecutor(max_workers=self._num_evaluators)

        self._task_key = task_key or resolve_task_key(evaluation)
        self._hp = get_task_hyperparams(self._task_key, hyperparam_overrides)
        self._rs = np.random.RandomState(seed)

        template_func = extract_function_from_string(self._template_program_str)
        self._expected_signature = (
            inspect.signature(template_func) if template_func is not None else None
        )

        self._sampler = CALMSampler(llm, profiler=profiler)
        self._prompts = PromptBuilder(
            problem_name=self._hp.problem_name,
            problem_description=self._task_description_str,
            problem_unit=self._hp.problem_unit,
            algorithm_template=self._template_program_str,
            rs=self._rs,
            injected_components=[],
            train_epoch_getter=lambda: self._train_epoch,
        )

        self._algos: List[HeuristicRecord] = []
        self._seed_algos: List[HeuristicRecord] = []
        self._used_prompts: List[Prompt] = []
        self._last_revisited_prompts: List[Prompt] = []
        self._injected_components: List[str] = self._prompts.injected_components
        self._seen_code_signatures = set()
        self._seen_performance_profiles: List[np.ndarray] = []
        self._numeric_refine_attempted = set()
        self._age_stuck = 0
        self._train_epoch = 0
        self._log_step = 0
        self._messages: List[List[dict]] = []
        self._best_perf = -float('inf')

        init_observability(self, max_consecutive_sample_failures=max_consecutive_sample_failures)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def _log_info(self, msg: str) -> None:
        text = f'<Epoch {self._train_epoch} / Step {self._log_step}> {msg}'
        if self._profiler is not None:
            self._profiler.append_log(text)
        if self._debug_mode:
            print(text)

    def _has_budget(self) -> bool:
        if is_search_aborted(self):
            return False
        if self._max_sample_nums is None:
            return True
        return self._tot_sample_nums < self._max_sample_nums

    def _remaining_budget(self) -> int:
        if self._max_sample_nums is None:
            return 10 ** 9
        return max(0, self._max_sample_nums - self._tot_sample_nums)

    @property
    def best_perf(self) -> float:
        if len(self._algos) > 0:
            return float(np.max([a.perf for a in self._algos if a.perf is not None]))
        return -float('inf')

    def matches_expected_signature(self, func) -> bool:
        if self._expected_signature is None:
            return True
        try:
            return inspect.signature(func) == self._expected_signature
        except (TypeError, ValueError):
            return False

    def performance_profile_signature(self, perfs):
        profile = np.ravel(perfs).astype(float)
        if profile.size == 0 or not np.all(np.isfinite(profile)):
            return None
        return profile

    def is_seen_performance_profile(self, perfs) -> bool:
        tolerance = float(self._hp.archive_profile_tolerance)
        if tolerance < 0:
            return False
        profile = self.performance_profile_signature(perfs)
        if profile is None:
            return False
        for seen_profile in self._seen_performance_profiles:
            if profile.shape == seen_profile.shape and np.max(np.abs(profile - seen_profile)) <= tolerance:
                return True
        return False

    def record_performance_profile(self, perfs) -> None:
        profile = self.performance_profile_signature(perfs)
        if profile is not None:
            self._seen_performance_profiles.append(profile.copy())

    def profile_reward_stats(self, algo: HeuristicRecord, base_algo: HeuristicRecord):
        if algo.perfs is None or base_algo.perfs is None:
            return None
        algo_perfs = np.ravel(np.asarray(algo.perfs, dtype=float))
        base_perfs = np.ravel(np.asarray(base_algo.perfs, dtype=float))
        if algo_perfs.shape != base_perfs.shape or algo_perfs.size == 0:
            return None
        deltas = algo_perfs - base_perfs
        if not np.all(np.isfinite(deltas)):
            return None
        scale = max(np.max(np.abs(base_perfs)), np.max(np.abs(algo_perfs)), 1e-10)
        return {
            'mean_delta': float(np.mean(deltas)),
            'worst_delta': float(np.min(deltas)),
            'non_worse_fraction': float(np.mean(deltas >= -1e-12)),
            'scale': float(scale),
        }

    def passes_profile_reward_gate(self, stats) -> bool:
        if stats is None:
            return True
        required_fraction = float(np.clip(
            self._hp.profile_reward_gate_non_worse_fraction, 0.0, 1.0,
        ))
        worst_tolerance = max(0.0, float(self._hp.profile_reward_gate_worst_tolerance))
        return (
            stats['mean_delta'] > 0.0
            and stats['non_worse_fraction'] >= required_fraction
            and stats['worst_delta'] >= -worst_tolerance * stats['scale']
        )

    def profile_reward_bonus(self, algo, base_algo, stats=None) -> float:
        max_bonus = max(0.0, float(self._hp.profile_reward_max_bonus))
        if max_bonus <= 0.0:
            return 0.0
        if stats is None:
            stats = self.profile_reward_stats(algo, base_algo)
        if stats is None:
            return 0.0
        required_fraction = float(np.clip(self._hp.profile_reward_non_worse_fraction, 0.0, 1.0))
        bonus = 0.0
        if stats['worst_delta'] > 0:
            weight = max(0.0, float(self._hp.profile_reward_worst_weight))
            bonus += weight * min(1.0, stats['worst_delta'] / stats['scale'])
        if stats['mean_delta'] > 0 and stats['non_worse_fraction'] >= required_fraction:
            weight = max(0.0, float(self._hp.profile_reward_consistency_weight))
            bonus += weight * stats['non_worse_fraction'] * min(1.0, stats['mean_delta'] / stats['scale'])
        return min(max_bonus, bonus)

    # ------------------------------------------------------------------ #
    # Evaluation adapter
    # ------------------------------------------------------------------ #

    def _code_to_program(self, code: str):
        trimmed = SampleTrimmer.trim_preface_of_function(code)
        if not trimmed:
            trimmed = code
        func = SampleTrimmer.sample_to_function(trimmed, self._template_program_str)
        if func is None:
            parsed = TextFunctionProgramConverter.text_to_program(code)
            if parsed is None or len(parsed.functions) != 1:
                return None, None
            func = parsed.functions[0]
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program_str)
        return func, program

    def _evaluate_code(self, code: str) -> tuple:
        """Return (score, perfs_array, status). status in {ok, code_bug}."""
        func, program = self._code_to_program(code)
        if func is None or program is None:
            return None, None, 'code_bug'
        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program,
            ).result()
        except Exception:
            return None, None, 'code_bug'
        if score is None:
            return None, None, 'code_bug'
        score = float(score)
        perfs = np.asarray([score], dtype=float)
        func.score = score
        func.evaluate_time = eval_time
        return score, perfs, 'ok'

    def _register_function(
            self,
            algo: HeuristicRecord,
            *,
            sample_time: float | None = None,
            counts_budget: bool = True,
    ) -> None:
        func, program = self._code_to_program(algo.code)
        if func is None:
            return
        func.score = algo.perf
        func.algorithm = algo.idea
        func.operator = algo.parent_prompt_type
        func.sample_time = sample_time
        algo.function = func
        if counts_budget:
            self._tot_sample_nums += 1
        if self._profiler is not None:
            self._profiler.register_function(func, program=str(program) if program else '')

    def _save_best_if_needed(self, algo: HeuristicRecord) -> None:
        if self._profiler is not None:
            self._profiler.save_best_algo(step=self._log_step, sid=algo.sid, code=algo.code)

    # ------------------------------------------------------------------ #
    # Init / schedule
    # ------------------------------------------------------------------ #

    def _initialize_seeds(self) -> None:
        seed_code, seed_idea = load_seed(self._task_key)
        score, perfs, status = self._evaluate_code(seed_code)
        if status != 'ok' or score is None:
            raise RuntimeError(f'CALM seed evaluation failed for task {self._task_key}')
        algo = HeuristicRecord(
            code=seed_code,
            idea=seed_idea if seed_idea.startswith('The idea of the algorithm is to') else (
                f'The idea of the algorithm is to solve the {self._task_key} in some way'
            ),
            name='seed',
            parent_prompt_type='seed',
            perf=float(score),
            perfs=perfs.copy(),
            birth=0,
            code_key=code_signature(seed_code),
        )
        self._seen_code_signatures.add(algo.code_key)
        self.record_performance_profile(perfs)
        self._algos = [algo]
        self._seed_algos = [algo]
        self._best_perf = algo.perf
        # Seed does not consume evaluation_budget in original CALM; still register for profiler.
        self._register_function(algo, counts_budget=False)
        if self._profiler is not None:
            # Count seed as sample 0 visibility without advancing fair budget.
            pass

        old_best = self._best_perf
        refined, self._tot_sample_nums, self._best_perf = run_numeric_refinement(
            candidate_parents=self._seed_algos,
            algos=self._algos,
            hp=self._hp,
            rs=self._rs,
            evaluations_used=self._tot_sample_nums,
            evaluation_budget=self._max_sample_nums or 10 ** 9,
            log_step=self._log_step,
            seen_code_signatures=self._seen_code_signatures,
            numeric_refine_attempted=self._numeric_refine_attempted,
            matches_expected_signature=self.matches_expected_signature,
            evaluate_code=self._evaluate_code_for_refine,
            is_seen_performance_profile=self.is_seen_performance_profile,
            record_performance_profile=self.record_performance_profile,
            best_perf=self._best_perf,
            log_info=self._log_info,
            register_accepted=lambda a: self._register_function(a, counts_budget=False),
            on_new_best=self._save_best_if_needed,
            variants_per_parent=self._hp.numeric_refine_initial_variants,
            top_k=0,
            parent_limit=len(self._seed_algos),
            source_label='INITIAL_NUMERIC_REFINE',
        )
        if self._best_perf > old_best:
            self._age_stuck = 0
        del refined

    def _evaluate_code_for_refine(self, code: str) -> tuple:
        score, perfs, status = self._evaluate_code(code)
        return score, perfs, status

    def _prepare_dataset(self) -> None:
        if not self._algos:
            self._initialize_seeds()

        messages, used_prompts, last_revisited, algos, age_stuck = prepare_round_messages(
            algos=self._algos,
            seed_algos=self._seed_algos,
            used_prompts=self._used_prompts,
            last_revisited_prompts=self._last_revisited_prompts,
            hp=self._hp,
            rs=self._rs,
            age_stuck=self._age_stuck,
            train_epoch=self._train_epoch,
            prompts=self._prompts,
            log_info=self._log_info,
        )
        self._messages = messages
        self._used_prompts = used_prompts
        self._last_revisited_prompts = last_revisited
        self._algos = algos
        self._age_stuck = age_stuck
        self._train_epoch += 1
        self._best_perf = self.best_perf

    # ------------------------------------------------------------------ #
    # Reward / accept (search path; rewards logged, no GRPO)
    # ------------------------------------------------------------------ #

    def _process_completions(self, prompts_batch: List[list], completions: List[str]) -> List[float]:
        self._log_step += 1
        if len(self._algos) >= self._hp.population_size:
            self._age_stuck += 1
            self._log_info(f'Stuck counter += 1, arrives at {self._age_stuck}')
        self._log_info(f'=== Step {self._log_step} (Epoch {self._train_epoch}) ===')

        res: List[Optional[float]] = []
        curr_algos: List[HeuristicRecord] = []
        curr_prompts: List[Prompt] = []
        curr_responses: List[str] = []
        curr_algos_reward_idx: List[int] = []
        numeric_refine_parents: List[HeuristicRecord] = []
        sample_times: List[float] = []

        for curr_algo_i, (prompt_msgs, response) in enumerate(zip(prompts_batch, completions)):
            prompt_text = prompt_msgs[-1]['content']
            prompt_idx = self._used_prompts.index(prompt_text)
            prompt = self._used_prompts[prompt_idx]
            curr_prompts.append(prompt)
            is_revisit = prompt in self._last_revisited_prompts
            algo_name = f"{prompt.op}{'_rev' if is_revisit else ''}({self._log_step})"
            curr_responses.append(response)

            idea = extract_first_double_braced(response)
            if idea is None:
                idea = extract_idea_description(response)
            idea_start = 'The idea of the algorithm is to'
            if idea is None or not idea.startswith(idea_start) or len(idea) - len(idea_start) <= 10:
                res.append(self._hp.reward_idea_not_exist)
                prompt.record_trial({
                    'performance': 'idea_not_exist',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue

            code = get_code(response)
            if code is None:
                res.append(self._hp.reward_code_not_exist)
                prompt.record_trial({
                    'performance': 'code_not_exist',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue
            if isinstance(code, list):
                code = code[0]

            step_func = extract_function_from_string(code)
            if step_func is None:
                res.append(self._hp.reward_function_not_exist)
                prompt.record_trial({
                    'performance': 'function_not_exist',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue
            if not self.matches_expected_signature(step_func):
                res.append(self._hp.reward_function_not_exist)
                prompt.record_trial({
                    'performance': 'signature_mismatch',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue

            code_key = code_signature(code)
            if code_key in self._seen_code_signatures:
                res.append(self._hp.reward_random_algorithm)
                prompt.record_trial({
                    'performance': 'duplicate_code',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                self._log_info(
                    f"{prompt.op.upper()}{'_REV' if is_revisit else ''} | "
                    'Rejected exact duplicate implementation before evaluation'
                )
                continue

            algo = HeuristicRecord(
                code=code,
                idea=idea,
                name=algo_name,
                parent_prompt_type=prompt.op,
                birth=self._log_step,
                response=response,
                code_key=code_key,
                born_from_revisit=is_revisit,
            )
            res.append(None)
            curr_algos.append(algo)
            curr_algos_reward_idx.append(curr_algo_i)
            sample_times.append(0.0)

        remaining = self._remaining_budget()
        if remaining < len(curr_algos):
            overflow_algos = curr_algos[max(remaining, 0):]
            overflow_indices = curr_algos_reward_idx[max(remaining, 0):]
            for algo, curr_algo_i in zip(overflow_algos, overflow_indices):
                prompt = curr_prompts[curr_algo_i]
                prompt.record_trial({
                    'performance': 'budget_exhausted',
                    'n_epoch': self._train_epoch,
                    'is_revisit': getattr(algo, 'born_from_revisit', False),
                })
                res[curr_algo_i] = self._hp.reward_random_algorithm
            curr_algos = curr_algos[:max(remaining, 0)]
            curr_algos_reward_idx = curr_algos_reward_idx[:max(remaining, 0)]

        # Count all evaluated candidates toward budget (CALM evaluations_used).
        self._tot_sample_nums += len(curr_algos)

        for algo, curr_algo_i in zip(curr_algos, curr_algos_reward_idx):
            prompt = curr_prompts[curr_algo_i]
            is_revisit = getattr(algo, 'born_from_revisit', False)
            score, perfs, status = self._evaluate_code(algo.code)
            if status != 'ok' or score is None:
                res[curr_algo_i] = self._hp.reward_bug_in_function
                prompt.record_trial({
                    'performance': 'code_bug',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue

            if 'random' in algo.code or 'np.random' in algo.code:
                res[curr_algo_i] = self._hp.reward_random_algorithm
                prompt.record_trial({
                    'performance': 'random_algorithm',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                continue

            algo.perf = float(score)
            algo.perfs = perfs.copy()
            self._seen_code_signatures.add(algo.code_key)
            if self.is_seen_performance_profile(perfs):
                res[curr_algo_i] = self._hp.reward_random_algorithm
                prompt.record_trial({
                    'performance': 'duplicate_performance_profile',
                    'n_epoch': self._train_epoch,
                    'is_revisit': is_revisit,
                })
                self._log_info(
                    f"{prompt.op.upper()}{'_REV' if is_revisit else ''} | "
                    f'Rejected duplicate performance profile after evaluation | '
                    f'Perf: {algo.perf:.6f} | Evals: {self._tot_sample_nums}/{self._max_sample_nums}'
                )
                continue
            self.record_performance_profile(perfs)
            numeric_refine_parents.append(algo)

            base_algos = []
            for base_code in prompt.base_codes:
                if base_code is None:
                    continue
                for a in self._algos:
                    if a.code.strip() == base_code.strip():
                        base_algos.append(a)
                        break

            is_new = algo not in self._algos
            is_new_best = algo.perf > self.best_perf
            if is_new:
                self._algos.append(algo)
                self._register_function(algo, counts_budget=False)

            if len(base_algos) == 0:
                base_algos = self._seed_algos[:]
            best_base_perf = float(np.max([a.perf for a in base_algos]))
            is_better = algo.perf > best_base_perf
            prompt.record_trial({
                'performance': float(algo.perf),
                'n_epoch': self._train_epoch,
                'is_revisit': is_revisit,
            })
            self._log_info(
                f"{prompt.op.upper()}{'_REV' if is_revisit else ''} | "
                f"Based on: [{', '.join([a.sid for a in base_algos])}] | "
                f'Perf: {algo.perf:.6f} | New Best: {is_new_best} | '
                f'Better than parents: {is_better} | '
                f'Evals: {self._tot_sample_nums}/{self._max_sample_nums}'
            )
            if is_new_best:
                self._age_stuck = 0
                self._best_perf = algo.perf
                self._log_info(
                    f'New best performance: {algo.perf} at step {self._log_step} by {algo.name}'
                )
                self._log_info(f'Idea: {algo.idea}')
                self._save_best_if_needed(algo)

            if prompt.op == 'initialization':
                reward = 0.0
            else:
                delta_perf = np.clip(
                    abs(algo.perf - best_base_perf)
                    / max(min(abs(algo.perf), abs(best_base_perf)), 1e-10),
                    1e-10,
                    1.0,
                )
                if is_better:
                    best_base_algo = max(base_algos, key=lambda a: a.perf)
                    profile_stats = self.profile_reward_stats(algo, best_base_algo)
                    if self.passes_profile_reward_gate(profile_stats):
                        reward = (
                            1.0 + delta_perf
                            + self.profile_reward_bonus(algo, best_base_algo, stats=profile_stats)
                        )
                    else:
                        reward = 0.0
                        if profile_stats is not None:
                            self._log_info(
                                'PROFILE_GATE_SKIP | '
                                f'Based on: [{best_base_algo.sid}] | Perf: {algo.perf:.6f} | '
                                f'Mean delta: {profile_stats["mean_delta"]:.6f} | '
                                f'Worst delta: {profile_stats["worst_delta"]:.6f} | '
                                f'Non-worse frac: {profile_stats["non_worse_fraction"]:.2f}'
                            )
                else:
                    if algo.perf >= best_base_perf:
                        reward = 0.0
                    else:
                        reward = self._hp.reward_random_algorithm / 2 * (
                            delta_perf if algo not in base_algos else (2 * 0.8)
                        )
                if prompt.is_injection:
                    match = re.search(
                        r"The new component ([A-Za-z()'\- ]+?) has been introduced",
                        curr_responses[curr_algo_i],
                    )
                    if match:
                        new_component = match.group(1).strip()
                        if new_component not in self._injected_components and is_new:
                            self._injected_components.append(new_component)
                            self._log_info(f'New component {new_component} has been introduced')

            res[curr_algo_i] = reward

        old_best = self._best_perf
        _, self._tot_sample_nums, self._best_perf = run_numeric_refinement(
            candidate_parents=numeric_refine_parents,
            algos=self._algos,
            hp=self._hp,
            rs=self._rs,
            evaluations_used=self._tot_sample_nums,
            evaluation_budget=self._max_sample_nums or 10 ** 9,
            log_step=self._log_step,
            seen_code_signatures=self._seen_code_signatures,
            numeric_refine_attempted=self._numeric_refine_attempted,
            matches_expected_signature=self.matches_expected_signature,
            evaluate_code=self._evaluate_code_for_refine,
            is_seen_performance_profile=self.is_seen_performance_profile,
            record_performance_profile=self.record_performance_profile,
            best_perf=self.best_perf,
            log_info=self._log_info,
            register_accepted=lambda a: self._register_function(a, counts_budget=False),
            on_new_best=self._save_best_if_needed,
        )
        if self._best_perf > old_best:
            self._age_stuck = 0

        perfs = map(str, sorted([a.perf for a in self._algos if a.perf is not None])[::-1])
        self._log_info(f"Number of algos: {len(self._algos)}, Perfs: {','.join(perfs)}")
        return [0.0 if r is None else float(r) for r in res]

    def _run_round(self) -> None:
        if not self._messages:
            return
        expanded_prompts: List[list] = []
        expanded_completions: List[str] = []
        n_gen = max(1, int(self._hp.n_generations))
        for messages in self._messages:
            if not self._has_budget():
                break
            try:
                responses = self._sampler.sample(
                    messages,
                    n=n_gen,
                    operator='calm_search',
                    sample_order=self._tot_sample_nums + 1,
                )
                reset_sample_failures(self)
            except Exception as exc:
                if self._debug_mode:
                    raise
                record_sample_failure(self, exc)
                continue
            for response in responses:
                expanded_prompts.append(messages)
                expanded_completions.append(response)
        if expanded_completions:
            self._process_completions(expanded_prompts, expanded_completions)

    def _save_trace(self) -> None:
        if self._profiler is None:
            return
        prompt_trace = [
            {
                'prompt': prompt.prompt,
                'op': prompt.op,
                'n_calls': prompt.n_calls,
                'best_generated_algo_perf': prompt.best_generated_algo_perf,
                'statuses': prompt.statuses,
            }
            for prompt in self._used_prompts
        ]
        algo_trace = []
        for algo in self._algos:
            algo_trace.append({
                'name': algo.name,
                'perf': None if algo.perf is None else float(algo.perf),
                'perfs': [] if algo.perfs is None else [float(x) for x in np.ravel(algo.perfs)],
                'idea': algo.idea,
                'birth': algo.birth,
                'parent_prompt_type': algo.parent_prompt_type,
                'code': algo.code,
            })
        self._profiler.save_trace({
            'method': self.METHOD_LABEL,
            'log_step': self._log_step,
            'train_epoch': self._train_epoch,
            'evaluations_used': self._tot_sample_nums,
            'prompts': prompt_trace,
            'algos': algo_trace,
        })

    def run(self) -> None:
        log_state(self, phase='start', method='calm', sample_count=self._tot_sample_nums)
        self._prepare_dataset()
        while self._has_budget():
            self._run_round()
            self._save_trace()
            if not self._has_budget():
                break
            self._prepare_dataset()
            log_event(
                self,
                event='epoch',
                method='calm',
                train_epoch=self._train_epoch,
                sample_count=self._tot_sample_nums,
                archive_size=len(self._algos),
                best_perf=self.best_perf,
            )
        self._save_trace()
        if self._profiler is not None:
            self._profiler.finish()
        log_state(
            self,
            phase='final',
            method='calm',
            sample_count=self._tot_sample_nums,
            archive_size=len(self._algos),
            best_perf=self.best_perf,
        )
        self._evaluation_executor.shutdown(wait=False)

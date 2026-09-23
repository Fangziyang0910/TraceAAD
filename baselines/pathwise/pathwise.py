from __future__ import annotations

import concurrent.futures
import copy
import math
import time
import traceback
from dataclasses import dataclass
from typing import Literal, Optional

from .graph import ParentInfo, PathWiseAction, PathWiseEdge, PathWiseGraph, PathWiseNode
from .population import Population
from .profiler import PathWiseProfiler
from .prompt import PathWisePrompt
from .sampler import PathWiseSampler
from llm4ad.base import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter
from llm4ad.tools.profiler import ProfilerBase
from .observability import (
    close_llm,
    finish_profiler,
    init_observability,
    is_search_aborted,
    log_error,
    log_event,
    log_llm_call,
    log_state,
    record_sample_failure,
    reset_sample_failures,
    shutdown_executor,
)


@dataclass
class _Rollout:
    node: PathWiseNode | None
    score: float
    description: str
    action: PathWiseAction
    sample_time: float


class PathWise:
    def __init__(
            self,
            llm: LLM,
            evaluation: Evaluation,
            profiler: ProfilerBase = None,
            max_sample_nums: Optional[int] = 500,
            pop_size: int = 6,
            init_pop_size: Optional[int] = None,
            num_actions: int = 2,
            num_rollouts: int = 2,
            max_inner_steps: int = 3,
            num_samplers: int = 1,
            num_evaluators: int = 1,
            *,
            policy_llm: LLM | None = None,
            world_model_llm: LLM | None = None,
            policy_critic_llm: LLM | None = None,
            world_model_critic_llm: LLM | None = None,
            external_knowledge: str = "",
            use_policy_critic: bool = True,
            use_world_model_critic: bool = True,
            use_prompt_perturbation: bool = True,
            preserve_best_nodes_innersteps: bool = True,
            policy_perturbation_prob: float = 0.5,
            policy_perturbation_final_prob: float = 0.25,
            world_model_perturbation_prob: float = 0.5,
            world_model_perturbation_final_prob: float = 0.25,
            resume_mode: bool = False,
            debug_mode: bool = False,
            multi_thread_or_process_eval: Literal["thread", "process"] = "thread",
            max_consecutive_sample_failures: int = 20,
            **kwargs,
    ):
        """PathWise integrated with the LLM4AD method/evaluation interfaces.

        PathWise keeps the original mechanism's two-timescale search: a population
        anchors each outer iteration, and an entailment graph is expanded by policy
        actions, world-model rollouts, and critic reflections inside the iteration.
        Scores follow LLM4AD's convention: higher is better.
        """
        max_fe = kwargs.pop("max_fe", None)
        alias_num_actions = kwargs.pop("N_a", None)
        alias_num_rollouts = kwargs.pop("N_w", None)
        alias_pop_size = kwargs.pop("N_p", None)
        if max_fe is not None and max_sample_nums == 500:
            max_sample_nums = max_fe
        if alias_num_actions is not None and num_actions == 2:
            num_actions = alias_num_actions
        if alias_num_rollouts is not None and num_rollouts == 2:
            num_rollouts = alias_num_rollouts
        if alias_pop_size is not None and pop_size == 6:
            pop_size = alias_pop_size

        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._pop_size = pop_size
        self._init_pop_size = init_pop_size if init_pop_size is not None else 5 * pop_size
        self._num_actions = num_actions
        self._num_rollouts = num_rollouts
        self._max_inner_steps = max_inner_steps
        self._num_samplers = num_samplers
        self._num_evaluators = num_evaluators
        self._external_knowledge = external_knowledge or ""

        self._use_policy_critic = use_policy_critic
        self._use_world_model_critic = use_world_model_critic
        self._use_prompt_perturbation = use_prompt_perturbation
        self._preserve_best_nodes_innersteps = preserve_best_nodes_innersteps
        self._policy_perturbation_prob = policy_perturbation_prob
        self._policy_perturbation_final_prob = policy_perturbation_final_prob
        self._world_model_perturbation_prob = world_model_perturbation_prob
        self._world_model_perturbation_final_prob = world_model_perturbation_final_prob

        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        self._multi_thread_or_process_eval = multi_thread_or_process_eval

        self._policy_llm = policy_llm or llm
        self._world_model_llm = world_model_llm or llm
        self._policy_critic_llm = policy_critic_llm or self._policy_llm
        self._world_model_critic_llm = world_model_critic_llm or self._world_model_llm
        for agent_llm in self._llms():
            agent_llm.debug_mode = debug_mode

        max_world_model_retries = kwargs.pop("max_world_model_retries", 3)
        max_policy_retries = kwargs.pop("max_policy_retries", 1)
        self._max_world_model_retries = max(0, int(max_world_model_retries))
        self._max_policy_retries = max(0, int(max_policy_retries))

        raw_template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        if raw_template_program is None or len(raw_template_program.functions) != 1:
            raise ValueError("PathWise requires an evaluation template with exactly one evolvable function.")
        self._template_program: Program = PathWiseSampler.with_global_imports(raw_template_program)
        self._function_to_evolve: Function = copy.deepcopy(self._template_program.functions[0])
        self._function_to_evolve_name = self._function_to_evolve.name

        self._sampler = PathWiseSampler(self._world_model_llm, self._template_program)
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **kwargs)
        self._profiler = profiler
        self._prompt = PathWisePrompt(self._task_description_str, self._function_to_evolve)
        self._population = Population(pop_size=self._pop_size)

        self._tot_sample_nums = 0
        init_observability(self, max_consecutive_sample_failures)
        self._outer_iteration = 0
        self._inner_step = 0
        self._node_counter = 0
        self._best_node: PathWiseNode | None = None
        self._all_nodes_archive: dict[str, PathWiseNode] = {}
        self._root_node_ids: set[str] = set()
        self._discarded_nodes: list[PathWiseNode] = []
        self._policy_reflection = ""
        self._world_model_reflection = ""
        self._policy_reflection_history: list[str] = []
        self._world_model_reflection_history: list[str] = []

        assert multi_thread_or_process_eval in ["thread", "process"]
        if multi_thread_or_process_eval == "thread":
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_evaluators)
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_evaluators)

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    def _llms(self) -> list[LLM]:
        unique = []
        seen = set()
        for agent_llm in [
            self._policy_llm,
            self._world_model_llm,
            self._policy_critic_llm,
            self._world_model_critic_llm,
        ]:
            if id(agent_llm) not in seen:
                seen.add(id(agent_llm))
                unique.append(agent_llm)
        return unique

    def _has_budget(self) -> bool:
        return not is_search_aborted(self) and (self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums)

    def _debug_print(self, title: str, content: str):
        if self._debug_mode:
            print("--------------------------------------------------------------------")
            print(f"{title}: \n{content}")
            print("--------------------------------------------------------------------\n")

    def _perturbation_probability(self, initial: float, final: float) -> float:
        if self._max_sample_nums is None or self._max_sample_nums <= 0:
            return initial
        progress = min(1.0, self._tot_sample_nums / self._max_sample_nums)
        return initial + (final - initial) * progress

    def _policy_perturbation(self) -> str:
        if not self._use_prompt_perturbation:
            return ""
        prob = self._perturbation_probability(
            self._policy_perturbation_prob,
            self._policy_perturbation_final_prob,
        )
        return PathWisePrompt.perturbation_phrase(PathWisePrompt.POLICY_PHRASES, prob)

    def _world_model_perturbation(self) -> str:
        if not self._use_prompt_perturbation:
            return ""
        prob = self._perturbation_probability(
            self._world_model_perturbation_prob,
            self._world_model_perturbation_final_prob,
        )
        return PathWisePrompt.perturbation_phrase(PathWisePrompt.WORLD_MODEL_PHRASES, prob)

    @staticmethod
    def _valid_score(score) -> bool:
        return Population.is_valid_score(score)

    def _update_best(self, node: PathWiseNode):
        if not self._valid_score(node.score):
            return
        if self._best_node is None or node.score > self._best_node.score:
            self._best_node = copy.deepcopy(node)

    def _register_node(self, node: PathWiseNode, program: Program):
        if self._profiler is not None:
            self._profiler.register_function(node.function, program=str(program))

    def _register_population(self):
        if isinstance(self._profiler, PathWiseProfiler):
            self._profiler.register_population(self._population)

    def _parent_info(self, parent_ids: list[str], graph: PathWiseGraph | None = None) -> list[ParentInfo]:
        parents = []
        for parent_id in parent_ids:
            node = None
            if graph is not None:
                node = graph.nodes.get(parent_id)
            if node is None:
                node = self._all_nodes_archive.get(parent_id)
            if node is not None:
                parents.append(ParentInfo(parent_id, node.description, node.score))
        return parents

    def _evaluate_function(
            self,
            func: Function,
            operator: str,
            description: str,
            rationale: str,
            *,
            node_id: str,
            parent_ids: list[str] | None = None,
            graph: PathWiseGraph | None = None,
            sample_time: float = 0.0,
            program: Program | None = None,
    ) -> PathWiseNode | None:
        if not self._has_budget():
            return None

        func = copy.deepcopy(func)
        func.operator = operator
        func.algorithm = description
        if program is None:
            program = TextFunctionProgramConverter.function_to_program(func, self._template_program)
        if program is None:
            self._tot_sample_nums += 1
            log_event(self, event="sample_rejected", method="pathwise", status="program_parse_failed",
                      operator=operator, node_id=node_id, sample_order=self._tot_sample_nums,
                      counts_budget=True)
            return None

        try:
            score, eval_time = self._evaluation_executor.submit(
                self._evaluator.evaluate_program_record_time,
                program,
            ).result()
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="evaluate",
                operator=operator,
                sample_order=self._tot_sample_nums + 1,
                node_id=node_id,
                parent_ids=parent_ids or [],
                counts_budget=False,
            )
            return None

        func.score = score
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        self._tot_sample_nums += 1
        reset_sample_failures(self)

        node = PathWiseNode(
            function=func,
            rationale=rationale,
            description=description,
            score=score,
            node_id=node_id,
            parents=self._parent_info(parent_ids or [], graph),
        )
        self._register_node(node, program)
        log_event(self, event="sample_registered", method="pathwise",
                  status="evaluated" if self._valid_score(score) else "invalid",
                  operator=operator, node_id=node_id, parent_ids=parent_ids or [],
                  score=score, sample_order=self._tot_sample_nums, counts_budget=True)
        log_state(self, phase="archive", method="pathwise", sample_count=self._tot_sample_nums,
                  archive_size=len(self._all_nodes_archive), best_score=self._best_node.score if self._best_node else None)

        if not self._valid_score(score):
            return None
        self._all_nodes_archive[node.node_id] = copy.deepcopy(node)
        self._update_best(node)
        return node

    def _count_invalid_sample(self):
        if self._has_budget():
            self._tot_sample_nums += 1

    def _init_sampling_kwargs(self) -> dict:
        temperature = getattr(self._policy_llm, "temperature", None)
        if isinstance(temperature, (int, float)):
            return {"temperature": temperature + 0.3}
        return {}

    def _initialize_population(self):
        candidates: list[PathWiseNode] = []
        for i in range(self._init_pop_size):
            if not self._has_budget():
                break
            prompt = self._prompt.initialization_prompt(i, self._external_knowledge)
            self._debug_print("PathWise Initialization Prompt", prompt)
            sample_start = time.time()
            try:
                response = self._policy_llm.draw_sample(prompt, **self._init_sampling_kwargs())
            except Exception as exc:
                record_sample_failure(
                    self,
                    exc,
                    stage="initialization",
                    operator="init",
                    sample_order=self._tot_sample_nums + 1,
                    prompt=prompt,
                    role="policy",
                    counts_budget=False,
                )
                continue
            sample_time = time.time() - sample_start
            parsed = PathWiseSampler.parse_initialization_response(response, self._template_program)
            log_llm_call(
                self,
                method="pathwise",
                stage="initialization",
                role="policy",
                operator="init",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                response=response,
                parse_success=parsed is not None,
                sample_time=sample_time,
            )
            if parsed is None:
                self._count_invalid_sample()
                log_event(self, event="sample_rejected", method="pathwise", status="parse_failed",
                          operator="init", sample_order=self._tot_sample_nums, counts_budget=True)
                continue
            func, description, rationale = parsed
            node = self._evaluate_function(
                func,
                "init",
                description,
                rationale,
                node_id=f"init_{i}",
                sample_time=sample_time,
            )
            if node is not None:
                candidates.append(node)

        selected = self._rank_unique(candidates)
        if selected:
            while len(selected) < self._pop_size:
                duplicate = copy.deepcopy(selected[0])
                duplicate.node_id = f"{selected[0].node_id}_dup{len(selected)}"
                duplicate.description = f"{duplicate.description} (duplicated)"
                duplicate.rationale = f"{duplicate.rationale} (duplicated for population size)"
                self._all_nodes_archive[duplicate.node_id] = copy.deepcopy(duplicate)
                selected.append(duplicate)

        self._population.set_nodes(selected[:self._pop_size], increment_generation=True)
        self._register_population()
        log_state(self, phase="population", method="pathwise", generation=self._population.generation,
                  sample_count=self._tot_sample_nums, population_size=len(self._population),
                  node_ids=[node.node_id for node in self._population.nodes])

    def _fallback_action(self, state: list[PathWiseNode]) -> PathWiseAction:
        best = max(state, key=lambda node: node.score)
        return PathWiseAction(
            parents=[best.node_id],
            rationale="Refine the best available heuristic by changing its core decision rule.",
        )

    def _policy_actions(self, state: list[PathWiseNode], graph: PathWiseGraph) -> list[PathWiseAction]:
        actions = []
        for _ in range(self._num_actions):
            prompt = self._prompt.policy_prompt(
                state,
                self._policy_reflection,
                perturbation=self._policy_perturbation(),
            )
            self._debug_print("PathWise Policy Prompt", prompt)
            action = None
            for attempt in range(self._max_policy_retries + 1):
                try:
                    response = self._policy_llm.draw_sample(prompt)
                except Exception as exc:
                    record_sample_failure(
                        self,
                        exc,
                        stage="policy",
                        operator="policy",
                        sample_order=self._tot_sample_nums + 1,
                        prompt=prompt,
                        role="policy",
                        retry=attempt,
                        counts_budget=False,
                    )
                    action = self._fallback_action(state)
                    actions.append(action)
                    break
                action = PathWiseSampler.parse_policy_response(response, state)
                log_llm_call(
                    self,
                    method="pathwise",
                    stage="policy",
                    role="policy",
                    operator="policy",
                    sample_order=self._tot_sample_nums + 1,
                    prompt=prompt,
                    response=response,
                    parse_success=action is not None,
                    retry=attempt,
                    state_node_ids=[node.node_id for node in state],
                )
                if action is not None:
                    actions.append(action)
                    break
                if attempt < self._max_policy_retries:
                    log_event(self, event="policy_retry", method="pathwise", status="retry",
                              sample_order=self._tot_sample_nums + 1,
                              retry=attempt + 1,
                              state_node_ids=[node.node_id for node in state])
            if action is None:
                log_event(self, event="policy_skipped", method="pathwise", status="invalid_parent_selection",
                          sample_order=self._tot_sample_nums + 1,
                          state_node_ids=[node.node_id for node in state])
        return actions

    def _fallback_world_model_rollout(
            self,
            action: PathWiseAction,
            state: list[PathWiseNode],
            graph: PathWiseGraph,
            action_idx: int,
            rollout_idx: int,
            sample_time: float,
            reason: str,
    ) -> _Rollout:
        parent_nodes = [node for node in state if node.node_id in action.parents]
        fallback_source = parent_nodes[0] if parent_nodes else (self._population.nodes[0] if self._population.nodes else None)
        if fallback_source is None:
            return _Rollout(None, float("-inf"), reason, action, sample_time)
        description = (
            f"Fallback heuristic {rollout_idx} "
            f"(invalid LLM output after {self._max_world_model_retries} retries)."
        )
        node = self._evaluate_function(
            fallback_source.function,
            "world_model",
            description,
            action.rationale,
            node_id=f"rollout_{self._outer_iteration}_{self._inner_step}_{action_idx}_{rollout_idx}",
            parent_ids=action.parents,
            graph=graph,
            sample_time=sample_time,
        )
        score = node.score if node is not None else float("-inf")
        if node is not None:
            log_event(self, event="world_model_fallback", method="pathwise", status="fallback",
                      sample_order=self._tot_sample_nums, action_idx=action_idx,
                      rollout_idx=rollout_idx, source_node_id=fallback_source.node_id,
                      reason=reason, counts_budget=True)
        return _Rollout(node, score, description, action, sample_time)

    def _world_model_rollout(
            self,
            action: PathWiseAction,
            state: list[PathWiseNode],
            graph: PathWiseGraph,
            action_idx: int,
            rollout_idx: int,
    ) -> _Rollout:
        parent_nodes = [node for node in state if node.node_id in action.parents]
        prompt = self._prompt.world_model_prompt(
            action,
            parent_nodes,
            self._world_model_reflection,
            perturbation=self._world_model_perturbation(),
        )
        self._debug_print("PathWise World Model Prompt", prompt)
        total_sample_time = 0.0
        for attempt in range(self._max_world_model_retries + 1):
            sample_start = time.time()
            try:
                response = self._world_model_llm.draw_sample(prompt)
            except Exception as exc:
                total_sample_time += time.time() - sample_start
                record_sample_failure(
                    self,
                    exc,
                    stage="world_model",
                    operator="world_model",
                    sample_order=self._tot_sample_nums + 1,
                    prompt=prompt,
                    role="world_model",
                    action_idx=action_idx,
                    rollout_idx=rollout_idx,
                    retry=attempt,
                    counts_budget=False,
                )
                if attempt >= self._max_world_model_retries:
                    return self._fallback_world_model_rollout(
                        action, state, graph, action_idx, rollout_idx, total_sample_time,
                        "World-model request failed.",
                    )
                continue
            sample_time = time.time() - sample_start
            total_sample_time += sample_time
            parsed = PathWiseSampler.parse_world_model_response(response, self._template_program)
            log_llm_call(
                self,
                method="pathwise",
                stage="world_model",
                role="world_model",
                operator="world_model",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                response=response,
                parse_success=parsed is not None,
                sample_time=sample_time,
                retry=attempt,
                action_idx=action_idx,
                rollout_idx=rollout_idx,
                parent_ids=action.parents,
            )
            if parsed is None:
                if attempt < self._max_world_model_retries:
                    log_event(self, event="world_model_retry", method="pathwise", status="retry",
                              operator="world_model", sample_order=self._tot_sample_nums + 1,
                              action_idx=action_idx, rollout_idx=rollout_idx,
                              retry=attempt + 1, counts_budget=False)
                    continue
                return self._fallback_world_model_rollout(
                    action, state, graph, action_idx, rollout_idx, total_sample_time,
                    "Invalid world-model rollout.",
                )

            func, description = parsed
            node = self._evaluate_function(
                func,
                "world_model",
                description,
                action.rationale,
                node_id=f"rollout_{self._outer_iteration}_{self._inner_step}_{action_idx}_{rollout_idx}",
                parent_ids=action.parents,
                graph=graph,
                sample_time=sample_time,
            )
            score = node.score if node is not None else float("-inf")
            return _Rollout(node, score, description, action, sample_time)
        return _Rollout(None, float("-inf"), "Invalid world-model rollout.", action, total_sample_time)

    def _run_world_model_rollouts(
            self,
            actions: list[PathWiseAction],
            state: list[PathWiseNode],
            graph: PathWiseGraph,
    ) -> list[list[_Rollout]]:
        all_rollouts: list[list[_Rollout]] = []
        for action_idx, action in enumerate(actions):
            action_rollouts = []
            for rollout_idx in range(self._num_rollouts):
                if not self._has_budget():
                    break
                action_rollouts.append(
                    self._world_model_rollout(action, state, graph, action_idx, rollout_idx)
                )
            all_rollouts.append(action_rollouts)
        return all_rollouts

    @staticmethod
    def _finite_rollouts(rollouts_per_action: list[list[_Rollout]]) -> list[_Rollout]:
        return [
            rollout
            for rollouts in rollouts_per_action
            for rollout in rollouts
            if rollout.node is not None and math.isfinite(rollout.score)
        ]

    def _policy_critic_reflection(
            self,
            state: list[PathWiseNode],
            actions: list[PathWiseAction],
            rollouts_per_action: list[list[_Rollout]],
    ) -> str:
        if not self._use_policy_critic:
            return self._policy_reflection
        finite_scores = {rollout.score for rollout in self._finite_rollouts(rollouts_per_action)}
        if len(finite_scores) <= 1:
            return self._policy_reflection

        summaries = []
        for rollouts in rollouts_per_action:
            valid = [rollout for rollout in rollouts if rollout.node is not None]
            if valid:
                avg = sum(rollout.score for rollout in valid) / len(valid)
                detail = "\n".join(
                    f"- {rollout.description} (score: {rollout.score:.6f})"
                    for rollout in valid
                )
                summaries.append(f"Average score: {avg:.6f}\nRollouts:\n{detail}")
            else:
                summaries.append("No valid rollouts.")
        prompt = self._prompt.policy_critic_prompt(state, actions, summaries)
        self._debug_print("PathWise Policy Critic Prompt", prompt)
        try:
            reflection = self._policy_critic_llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="policy_critic",
                operator="policy_critic",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                role="policy_critic",
                counts_budget=False,
            )
            return self._policy_reflection
        log_llm_call(
            self,
            method="pathwise",
            stage="policy_critic",
            role="policy_critic",
            operator="policy_critic",
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=reflection,
            parse_success=bool(reflection and reflection.strip()),
        )
        if reflection and reflection.strip():
            self._policy_reflection_history.append(reflection.strip())
            return reflection.strip()
        return self._policy_reflection

    def _world_model_critic_reflection(self, rollouts_per_action: list[list[_Rollout]]) -> str:
        if not self._use_world_model_critic:
            return self._world_model_reflection
        valid = self._finite_rollouts(rollouts_per_action)
        if len(valid) < 2:
            return self._world_model_reflection
        best = max(valid, key=lambda rollout: rollout.score).node
        worst = min(valid, key=lambda rollout: rollout.score).node
        if best.score == worst.score:
            return self._world_model_reflection
        prompt = self._prompt.world_model_critic_prompt(best, worst)
        self._debug_print("PathWise World Model Critic Prompt", prompt)
        try:
            reflection = self._world_model_critic_llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="world_model_critic",
                operator="world_model_critic",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                role="world_model_critic",
                counts_budget=False,
            )
            return self._world_model_reflection
        log_llm_call(
            self,
            method="pathwise",
            stage="world_model_critic",
            role="world_model_critic",
            operator="world_model_critic",
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=reflection,
            parse_success=bool(reflection and reflection.strip()),
        )
        if reflection and reflection.strip():
            self._world_model_reflection_history.append(reflection.strip())
            return reflection.strip()
        return self._world_model_reflection

    def _add_selected_node_to_graph(
            self,
            graph: PathWiseGraph,
            selected: _Rollout,
    ) -> PathWiseEdge:
        selected_node = copy.deepcopy(selected.node)
        selected_node.node_id = f"entail_{self._outer_iteration}_{self._inner_step}"
        selected_node.parents = self._parent_info(selected.action.parents, graph)
        self._all_nodes_archive[selected_node.node_id] = copy.deepcopy(selected_node)
        self._update_best(selected_node)

        graph.add_node(selected_node)
        edge = PathWiseEdge(
            parents=list(selected.action.parents),
            rationale=selected.action.rationale,
            child=selected_node.node_id,
        )
        graph.add_edge(edge)

        remove_ids = list(selected.action.parents)
        if self._preserve_best_nodes_innersteps and self._best_node is not None:
            remove_ids = [node_id for node_id in remove_ids if node_id != self._best_node.node_id]
        graph.remove_nodes(remove_ids)
        selected.node = selected_node
        return edge

    def _inner_entailment_step(self, graph: PathWiseGraph) -> PathWiseNode | None:
        state = graph.get_state()
        if not state:
            return None
        actions = self._policy_actions(state, graph)
        rollouts_per_action = self._run_world_model_rollouts(actions, state, graph)
        valid_rollouts = self._finite_rollouts(rollouts_per_action)
        if not valid_rollouts:
            return None

        selected = max(valid_rollouts, key=lambda rollout: rollout.score)
        self._discarded_nodes.extend(
            copy.deepcopy(rollout.node)
            for rollout in valid_rollouts
            if rollout is not selected and rollout.node is not None
        )
        edge = self._add_selected_node_to_graph(graph, selected)

        self._policy_reflection = self._policy_critic_reflection(state, actions, rollouts_per_action)
        self._world_model_reflection = self._world_model_critic_reflection(rollouts_per_action)
        if isinstance(self._profiler, PathWiseProfiler):
            self._profiler.register_entailment_step(
                outer_iteration=self._outer_iteration,
                inner_step=self._inner_step,
                actions=actions,
                selected_node=selected.node,
                edge=edge,
                policy_reflection=self._policy_reflection,
                world_model_reflection=self._world_model_reflection,
            )
        return selected.node

    def _create_initial_graph(self) -> PathWiseGraph:
        graph = PathWiseGraph()
        self._root_node_ids = set()
        for node in self._population.nodes:
            node = copy.deepcopy(node)
            self._root_node_ids.add(node.node_id)
            self._all_nodes_archive[node.node_id] = copy.deepcopy(node)
            graph.add_node(node)
        return graph

    def _construct_entailment_graph(self) -> tuple[PathWiseGraph, PathWiseNode | None]:
        graph = self._create_initial_graph()
        self._discarded_nodes = []
        self._policy_reflection = self._policy_reflection_history[-1] if self._policy_reflection_history else ""
        self._world_model_reflection = (
            self._world_model_reflection_history[-1] if self._world_model_reflection_history else ""
        )

        final_node = None
        for step in range(self._max_inner_steps):
            if not self._has_budget() or len(graph.get_state()) <= 1:
                break
            self._inner_step = step
            final_node = self._inner_entailment_step(graph)
            if final_node is None:
                break
        if final_node is None:
            final_node = self._fallback_final_node(graph)
        return graph, final_node

    def _fallback_final_node(self, graph: PathWiseGraph) -> PathWiseNode | None:
        if not self._population.nodes or not self._has_budget():
            return None
        best = max(self._population.nodes, key=lambda node: node.score)
        node = self._evaluate_function(
            best.function,
            "fallback",
            "Best from current population.",
            "Fallback from population.",
            node_id=f"fallback_{self._outer_iteration}",
            parent_ids=[],
            graph=graph,
        )
        if node is None:
            return None
        graph.add_node(node)
        log_event(self, event="entailment_fallback", method="pathwise", status="fallback",
                  sample_order=self._tot_sample_nums, node_id=node.node_id,
                  source_node_id=best.node_id, counts_budget=True)
        return node

    def _leaf_nodes(self, graph: PathWiseGraph) -> list[PathWiseNode]:
        parent_ids = graph.parent_ids()
        leaves = [
            node
            for node_id, node in graph.nodes.items()
            if node_id not in parent_ids and node_id not in self._root_node_ids
        ]
        return sorted(leaves, key=lambda node: node.score, reverse=True)

    def _rank_unique(self, nodes: list[PathWiseNode]) -> list[PathWiseNode]:
        unique = []
        seen_code = set()
        seen_score = set()
        for node in sorted(nodes, key=lambda item: item.score, reverse=True):
            if not self._valid_score(node.score):
                continue
            code_key = str(node.function)
            score_key = float(node.score)
            if code_key in seen_code or score_key in seen_score:
                continue
            seen_code.add(code_key)
            seen_score.add(score_key)
            unique.append(copy.deepcopy(node))
        return unique

    def _update_population(self, graph: PathWiseGraph):
        leaves = self._leaf_nodes(graph)
        if len(leaves) >= self._pop_size:
            next_nodes = leaves[:self._pop_size]
        else:
            leaf_code = {str(node.function) for node in leaves}
            remainder_candidates = [
                node
                for node in [
                    *self._discarded_nodes,
                    *graph.nodes.values(),
                    *self._population.nodes,
                ]
                if str(node.function) not in leaf_code
            ]
            next_nodes = leaves + self._rank_unique(remainder_candidates)[:self._pop_size - len(leaves)]

        self._population.set_nodes(next_nodes, increment_generation=True)
        for node in self._population.nodes:
            self._all_nodes_archive[node.node_id] = copy.deepcopy(node)
            self._update_best(node)
        self._register_population()
        log_state(self, phase="population", method="pathwise", generation=self._population.generation,
                  sample_count=self._tot_sample_nums, population_size=len(self._population),
                  node_ids=[node.node_id for node in self._population.nodes],
                  best_score=self._best_node.score if self._best_node else None)

    def run(self):
        run_status = "finished"
        run_error = None
        try:
            if not self._resume_mode:
                self._initialize_population()
                if len(self._population) == 0:
                    print(
                        "The search is terminated since PathWise was unable to obtain any feasible "
                        "algorithm during initialization. Please increase `init_pop_size` and check "
                        "the evaluator and LLM implementation."
                    )
                    return

            while self._has_budget() and len(self._population) > 1:
                log_event(self, event="outer_iteration_start", method="pathwise",
                          outer_iteration=self._outer_iteration, sample_count=self._tot_sample_nums,
                          population_size=len(self._population))
                graph, final_node = self._construct_entailment_graph()
                if final_node is None:
                    break
                self._update_population(graph)
                self._outer_iteration += 1
        except KeyboardInterrupt as exc:
            run_status = "interrupted"
            run_error = exc
        except Exception as exc:
            run_status = "error"
            run_error = exc
            log_error(self, "run", exc, method="pathwise", counts_budget=False)
            if self._debug_mode:
                traceback.print_exc()
                raise
        finally:
            if is_search_aborted(self):
                run_status = "aborted"
            summary_payload = {}
            if run_error is not None:
                summary_payload.update(
                    error_type=type(run_error).__name__,
                    error=str(run_error),
                )
            shutdown_executor(self._evaluation_executor)
            log_state(self, phase="final", method="pathwise", sample_count=self._tot_sample_nums,
                      outer_iteration=self._outer_iteration,
                      population_size=len(self._population),
                      archive_size=len(self._all_nodes_archive),
                      best_score=self._best_node.score if self._best_node else None)
            finish_profiler(self, status=run_status, **summary_payload)
            for agent_llm in self._llms():
                close_llm(agent_llm)

    @property
    def best_node(self) -> PathWiseNode | None:
        return copy.deepcopy(self._best_node)

    @property
    def best_function(self) -> Function | None:
        if self._best_node is None:
            return None
        return copy.deepcopy(self._best_node.function)


Pathwise = PathWise

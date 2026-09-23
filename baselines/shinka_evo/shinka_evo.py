from __future__ import annotations

import copy
import random
import re
from typing import Any, Callable, Optional, Sequence

from core import Evaluation, Function, LLM, Program, SecureEvaluator, TextFunctionProgramConverter
from baselines.profiler import ProfilerBase
from baselines.observability import (
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
)
from .bandit import ShinkaLLMBandit
from .population import ShinkaArchive, ShinkaProgram, is_valid_score
from .profiler import ShinkaEvoProfiler
from .sampler import PatchResult, ShinkaSampler


class ShinkaEvo:
    def __init__(
            self,
            llm: LLM,
            evaluation: Evaluation,
            profiler: ProfilerBase = None,
            max_sample_nums: Optional[int] = 50,
            *,
            num_generations: Optional[int] = 50,
            num_islands: int = 2,
            archive_size: int = 40,
            patch_types: Sequence[str] = ("diff", "full", "cross"),
            patch_type_probs: Sequence[float] = (0.6, 0.3, 0.1),
            parent_selection_strategy: str = "weighted",
            llms: list[LLM] | None = None,
            novelty_llm: LLM | None = None,
            meta_llm: LLM | None = None,
            embedding_fn: Callable[[str], Sequence[float]] | None = None,
            resume_mode: bool = False,
            debug_mode: bool = False,
            # ShinkaEvolve mechanism parameters.
            max_patch_resamples: int = 3,
            max_novelty_attempts: int = 3,
            code_embed_sim_threshold: float = 0.99,
            elite_selection_ratio: float = 0.3,
            num_archive_inspirations: int = 1,
            num_top_k_inspirations: int = 1,
            exploitation_alpha: float = 1.0,
            parent_selection_lambda: float = 10.0,
            num_beams: int = 5,
            archive_selection_strategy: str = "fitness",
            archive_criteria: dict[str, float] | None = None,
            enforce_island_separation: bool = True,
            island_selection_strategy: str = "uniform",
            migration_interval: int = 10,
            migration_rate: float = 0.0,
            island_elitism: bool = True,
            meta_rec_interval: int | None = 10,
            sample_single_meta_rec: bool = True,
            use_text_feedback: bool = False,
            inspiration_sort_order: str = "ascending",
            task_sys_msg: str | None = None,
            random_seed: int | None = None,
            llm_ucb_exploration_coef: float = 1.0,
            llm_ucb_epsilon: float = 0.2,
            llm_ucb_auto_decay: float | None = 0.95,
            max_consecutive_sample_failures: int = 20,
            **secure_evaluator_kwargs,
    ):
        self._template_program_str = evaluation.template_program
        self._task_description_str = evaluation.task_description
        self._max_sample_nums = max_sample_nums
        self._num_generations = num_generations
        self._max_patch_resamples = max_patch_resamples
        self._max_novelty_attempts = max_novelty_attempts
        self._code_embed_sim_threshold = code_embed_sim_threshold
        self._embedding_fn = embedding_fn
        self._novelty_llm = novelty_llm
        self._meta_llm = meta_llm
        self._meta_rec_interval = meta_rec_interval
        self._sample_single_meta_rec = sample_single_meta_rec
        self._resume_mode = resume_mode
        self._debug_mode = debug_mode
        self._rng = random.Random(random_seed)

        primary_llms = list(llms) if llms is not None else [llm]
        if llm not in primary_llms:
            primary_llms.insert(0, llm)
        self._llms = primary_llms
        for agent_llm in self._all_llms():
            agent_llm.debug_mode = debug_mode

        raw_template_program = TextFunctionProgramConverter.text_to_program(self._template_program_str)
        if raw_template_program is None or len(raw_template_program.functions) != 1:
            raise ValueError("ShinkaEvo requires an evaluation template with exactly one evolvable function.")
        self._template_program: Program = copy.deepcopy(raw_template_program)
        self._function_to_evolve: Function = copy.deepcopy(self._template_program.functions[0])

        self._archive = ShinkaArchive(
            num_islands=num_islands,
            archive_size=archive_size,
            elite_selection_ratio=elite_selection_ratio,
            num_archive_inspirations=num_archive_inspirations,
            num_top_k_inspirations=num_top_k_inspirations,
            parent_selection_strategy=parent_selection_strategy,
            exploitation_alpha=exploitation_alpha,
            parent_selection_lambda=parent_selection_lambda,
            num_beams=num_beams,
            archive_selection_strategy=archive_selection_strategy,
            archive_criteria=archive_criteria,
            enforce_island_separation=enforce_island_separation,
            island_selection_strategy=island_selection_strategy,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
            island_elitism=island_elitism,
            rng=self._rng,
        )
        self._bandit = ShinkaLLMBandit(
            self._llms,
            exploration_coef=llm_ucb_exploration_coef,
            epsilon=llm_ucb_epsilon,
            auto_decay=llm_ucb_auto_decay,
            seed=random_seed,
        )
        self._sampler = ShinkaSampler(
            llm=self._llms[0],
            template_program=self._template_program,
            task_description=self._task_description_str,
            patch_types=patch_types,
            patch_type_probs=patch_type_probs,
            use_text_feedback=use_text_feedback,
            inspiration_sort_order=inspiration_sort_order,
            task_sys_msg=task_sys_msg,
        )
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **secure_evaluator_kwargs)
        self._profiler = profiler
        self._tot_sample_nums = 0
        init_observability(self, max_consecutive_sample_failures)
        self._generation = 0
        self._evaluated_since_meta: list[ShinkaProgram] = []
        self._meta_summary = ""
        self._meta_scratch_pad = ""
        self._meta_recommendations = ""
        self._meta_recommendations_history: list[str] = []
        self._best_function_found: Function | None = None

        if profiler is not None:
            self._profiler.record_parameters(llm, evaluation, self)

    @property
    def best_function(self) -> Function | None:
        if self._archive.best_program is None:
            return None
        return copy.deepcopy(self._archive.best_program.function)

    @property
    def best_program(self) -> ShinkaProgram | None:
        return self._archive.best_program

    def _all_llms(self) -> list[LLM]:
        llms = list(self._llms)
        if self._novelty_llm is not None:
            llms.append(self._novelty_llm)
        if self._meta_llm is not None:
            llms.append(self._meta_llm)
        seen = set()
        unique = []
        for llm in llms:
            if id(llm) in seen:
                continue
            seen.add(id(llm))
            unique.append(llm)
        return unique

    def _has_budget(self) -> bool:
        return not is_search_aborted(self) and (self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums)

    def run(self):
        try:
            if not self._resume_mode and self._has_budget() and self._archive.initial_program is None:
                self._evaluate_seed()
            generation = 1
            while self._has_budget():
                if self._num_generations is not None and generation > self._num_generations:
                    break
                self._generation = generation
                self._run_generation(generation)
                for event in self._archive.maybe_migrate(generation):
                    self._register_event("island_migration", {"generation": generation, **event})
                generation += 1
        except Exception as exc:
            log_error(self, "run", exc, method="shinka_evo", counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            self._log_archive_state("final")
            finish_profiler(self, status="aborted" if is_search_aborted(self) else "finished")
            for llm in self._all_llms():
                close_llm(llm)

    def _evaluate_seed(self) -> ShinkaProgram:
        score, eval_time = self._evaluator.evaluate_program_record_time(program=self._template_program)
        self._tot_sample_nums += 1
        result = self._coerce_evaluation_result(score)
        func = copy.deepcopy(self._function_to_evolve)
        func.score = result["combined_score"] if result["correct"] else None
        func.evaluate_time = eval_time
        func.sample_time = 0.0
        func.operator = "seed"
        func.algorithm = "seed"
        embedding = self._compute_embedding(str(self._template_program)) if self._embedding_fn else []
        seed = ShinkaProgram.create(
            func,
            str(self._template_program),
            island_idx=0,
            generation=0,
            patch_type="seed",
            embedding=embedding,
            metadata={"evaluation_result": score},
            **result,
        )
        self._archive.seed_islands(seed)
        reset_sample_failures(self)
        self._best_function_found = self.best_function
        self._register_function(seed)
        self._register_event("archive_update", {
            "generation": 0,
            "program_id": seed.id,
            "archive_ids": list(self._archive.archive_ids),
        })
        self._log_archive_state("seed")
        return seed

    def _run_generation(self, generation: int) -> ShinkaProgram | None:
        arm, selected_llm, bandit_meta = self._bandit.select()
        self._sampler.llm = selected_llm
        self._register_event("bandit_select", {"generation": generation, "arm": arm, **bandit_meta})
        last_error = None
        for novelty_attempt in range(1, self._max_novelty_attempts + 1):
            for resample_attempt in range(1, self._max_patch_resamples + 1):
                if not self._has_budget():
                    return None
                try:
                    island_idx, parent, archive_inspirations, top_k_inspirations, fix_mode = self._select_parent_context()
                except ValueError as err:
                    self._register_event("parent_selection", {"generation": generation, "error": str(err)})
                    return None
                patch_type = "fix" if fix_mode else self._sampler.sample_patch_type(
                    archive_inspirations,
                    top_k_inspirations,
                )
                self._register_event("parent_selection", {
                    "generation": generation,
                    "novelty_attempt": novelty_attempt,
                    "resample_attempt": resample_attempt,
                    "island_idx": island_idx,
                    "parent_id": parent.id,
                    "fix_mode": fix_mode,
                    "patch_type": patch_type,
                    "archive_inspirations": [program.id for program in archive_inspirations],
                    "top_k_inspirations": [program.id for program in top_k_inspirations],
                })
                previous_error = last_error
                prompt = self._sampler.build_prompt(
                    parent,
                    archive_inspirations,
                    top_k_inspirations,
                    patch_type,
                    meta_recommendations=self._current_meta_recommendation(),
                    previous_error=previous_error,
                    fix_mode=fix_mode,
                )
                try:
                    response, sample_time = self._sampler.draw_sample(prompt)
                except Exception as exc:
                    aborted = record_sample_failure(
                        self,
                        exc,
                        stage="patch",
                        operator=patch_type,
                        sample_order=self._tot_sample_nums + 1,
                        prompt=prompt,
                        generation=generation,
                        arm=arm,
                        parent_id=parent.id,
                        island_idx=island_idx,
                        counts_budget=False,
                    )
                    if aborted:
                        return None
                    last_error = str(exc)
                    continue
                log_llm_call(
                    self,
                    method="shinka_evo",
                    stage="patch",
                    operator=patch_type,
                    sample_order=self._tot_sample_nums + 1,
                    prompt=prompt,
                    response=response,
                    sample_time=sample_time,
                    generation=generation,
                    arm=arm,
                    parent_id=parent.id,
                    island_idx=island_idx,
                    fix_mode=fix_mode,
                )
                patch = self._sampler.apply_response(response, parent, patch_type, fix_mode=fix_mode)
                patch.metadata.update({
                    "novelty_attempt": novelty_attempt,
                    "resample_attempt": resample_attempt,
                    "sample_time": sample_time,
                    "arm": arm,
                })
                self._register_patch_attempt(generation, parent, patch_type, patch)
                if patch.success:
                    reset_sample_failures(self)
                    novelty_ok, embedding, novelty_meta = self._assess_novelty(
                        patch,
                        parent,
                        generation,
                    )
                    self._register_event("novelty_decision", {
                        "generation": generation,
                        "parent_id": parent.id,
                        "accepted": novelty_ok,
                        **novelty_meta,
                    })
                    if not novelty_ok:
                        last_error = novelty_meta.get("novelty_explanation", "Proposal rejected by novelty check.")
                        break
                    return self._evaluate_candidate(
                        patch,
                        parent=parent,
                        archive_inspirations=archive_inspirations,
                        top_k_inspirations=top_k_inspirations,
                        generation=generation,
                        patch_type=patch_type,
                        sample_time=sample_time,
                        embedding=embedding,
                        arm=arm,
                    )
                previous_error = patch.error
                last_error = patch.error
                log_error(self, "patch_apply", None, method="shinka_evo",
                          operator=patch_type, generation=generation, parent_id=parent.id,
                          error=patch.error, counts_budget=False)
        self._bandit.update(arm, None, None)
        return None

    def _select_parent_context(self) -> tuple[int | None, ShinkaProgram, list[ShinkaProgram], list[ShinkaProgram], bool]:
        island_idx = self._archive.select_island()
        parent, fix_mode = self._archive.sample_parent_with_fix_mode(island_idx)
        if fix_mode:
            return island_idx, parent, [], [], True
        archive_inspirations, top_k_inspirations = self._archive.sample_inspirations(parent)
        return island_idx, parent, archive_inspirations, top_k_inspirations, False

    def _evaluate_candidate(
            self,
            patch: PatchResult,
            *,
            parent: ShinkaProgram,
            archive_inspirations: list[ShinkaProgram],
            top_k_inspirations: list[ShinkaProgram],
            generation: int,
            patch_type: str,
            sample_time: float,
            embedding: list[float],
            arm: int,
    ) -> ShinkaProgram | None:
        if not self._has_budget() or patch.function is None or patch.program is None:
            return None
        try:
            eval_result, eval_time = self._evaluator.evaluate_program_record_time(program=patch.program)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="evaluate",
                operator=patch_type,
                sample_order=self._tot_sample_nums + 1,
                generation=generation,
                parent_id=parent.id,
                counts_budget=False,
            )
            return None
        self._tot_sample_nums += 1
        reset_sample_failures(self)
        result = self._coerce_evaluation_result(eval_result)
        func = copy.deepcopy(patch.function)
        func.score = result["combined_score"] if result["correct"] else None
        func.evaluate_time = eval_time
        func.sample_time = sample_time
        func.operator = patch_type
        func.algorithm = patch_type
        metadata = dict(patch.metadata)
        metadata["evaluation_result"] = eval_result
        program = ShinkaProgram.create(
            func,
            str(patch.program),
            parent_id=parent.id,
            archive_inspiration_ids=[program.id for program in archive_inspirations],
            top_k_inspiration_ids=[program.id for program in top_k_inspirations],
            island_idx=parent.island_idx,
            generation=generation,
            patch_type=patch_type,
            code_diff=patch.code_diff,
            embedding=embedding,
            metadata=metadata,
            **result,
        )
        self._archive.add_program(program, update_archive=False)
        replaced = self._archive.update_archive(program)
        if replaced is not None:
            self._register_event("archive_update", {
                "generation": generation,
                "program_id": program.id,
                "replaced_program_id": replaced,
                "archive_ids": list(self._archive.archive_ids),
            })
        else:
            self._register_event("archive_update", {
                "generation": generation,
                "program_id": program.id,
                "archive_ids": list(self._archive.archive_ids),
            })
        baseline_program = self._archive.initial_program
        baseline = max(parent.combined_score, baseline_program.combined_score if baseline_program else 0.0)
        reward = program.combined_score if program.correct else None
        update = self._bandit.update(arm, reward, baseline)
        self._register_event("bandit_update", {
            "generation": generation,
            "arm": arm,
            "reward": update.reward,
            "baseline": update.baseline,
            "shifted_reward": update.shifted_reward,
        })
        self._evaluated_since_meta.append(program)
        self._maybe_update_meta()
        self._register_function(program)
        self._best_function_found = self.best_function
        self._log_archive_state("candidate_evaluated")
        return program

    def _assess_novelty(
            self,
            patch: PatchResult,
            parent: ShinkaProgram,
            generation: int,
    ) -> tuple[bool, list[float], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "novelty_checks_performed": 0,
            "novelty_cost": 0.0,
            "novelty_explanation": "",
            "max_similarity": 0.0,
            "similarity_scores": [],
        }
        if self._embedding_fn is None or patch.program is None or generation == 0:
            return True, [], metadata
        embedding = self._compute_embedding(str(patch.program))
        if not embedding:
            metadata["novelty_explanation"] = "No embedding returned; novelty check skipped."
            return True, [], metadata
        similarities = self._archive.compute_similarities(embedding, parent.island_idx)
        metadata["similarity_scores"] = similarities
        if not similarities:
            return True, embedding, metadata
        max_similarity = max(similarities)
        metadata["max_similarity"] = max_similarity
        if max_similarity <= self._code_embed_sim_threshold:
            return True, embedding, metadata
        metadata["novelty_checks_performed"] = 1
        if self._novelty_llm is None:
            metadata["novelty_explanation"] = (
                f"Rejected: max similarity {max_similarity:.3f} exceeds "
                f"threshold {self._code_embed_sim_threshold:.3f}."
            )
            return False, embedding, metadata
        similar = self._archive.most_similar_program(embedding, parent.island_idx)
        prompt = self._novelty_prompt(str(patch.program), similar)
        try:
            response = self._novelty_llm.draw_sample(prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="novelty",
                operator="novelty",
                sample_order=self._tot_sample_nums + 1,
                prompt=prompt,
                generation=generation,
                parent_id=parent.id,
                counts_budget=False,
            )
            metadata["novelty_explanation"] = f"Novelty request failed: {exc}"
            return False, embedding, metadata
        log_llm_call(
            self,
            method="shinka_evo",
            stage="novelty",
            operator="novelty",
            sample_order=self._tot_sample_nums + 1,
            prompt=prompt,
            response=response,
            generation=generation,
            parent_id=parent.id,
            accepted=self._is_novelty_response_accept(response),
        )
        metadata["novelty_explanation"] = response
        return self._is_novelty_response_accept(response), embedding, metadata

    def _compute_embedding(self, program: str) -> list[float]:
        if self._embedding_fn is None:
            return []
        embedding = self._embedding_fn(program)
        return [float(value) for value in embedding] if embedding is not None else []

    @staticmethod
    def _is_novelty_response_accept(response: str) -> bool:
        content = (response or "").strip().upper()
        return content.startswith("NOVEL") or content.startswith("**NOVEL**")

    @staticmethod
    def _novelty_prompt(proposed_code: str, most_similar: ShinkaProgram | None) -> str:
        existing = most_similar.program if most_similar is not None else ""
        return (
            "You are judging whether a proposed program is meaningfully novel.\n"
            "Respond with NOVEL or NOT NOVEL first, then a brief explanation.\n\n"
            "# Existing similar program\n"
            f"```python\n{existing}\n```\n\n"
            "# Proposed program\n"
            f"```python\n{proposed_code}\n```"
        )

    @staticmethod
    def _coerce_evaluation_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            raw_score = result.get("combined_score", result.get("score"))
            correct = bool(result.get("correct", is_valid_score(raw_score)))
            if not is_valid_score(raw_score):
                correct = False
                score = 0.0
            else:
                score = float(raw_score)
            return {
                "combined_score": score,
                "correct": correct,
                "public_metrics": dict(result.get("public_metrics", {})),
                "private_metrics": dict(result.get("private_metrics", {})),
                "text_feedback": result.get("text_feedback", ""),
            }
        correct = is_valid_score(result)
        return {
            "combined_score": float(result) if correct else 0.0,
            "correct": correct,
            "public_metrics": {},
            "private_metrics": {},
            "text_feedback": "",
        }

    def _maybe_update_meta(self) -> None:
        if self._meta_llm is None or self._meta_rec_interval is None:
            return
        if len(self._evaluated_since_meta) < self._meta_rec_interval:
            return
        recent = list(self._evaluated_since_meta)
        self._evaluated_since_meta = []
        summaries_prompt = self._meta_prompt_individual(recent)
        try:
            summaries = self._meta_llm.draw_sample(summaries_prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="meta_summary",
                operator="meta_summary",
                sample_order=self._tot_sample_nums + 1,
                prompt=summaries_prompt,
                generation=self._generation,
                counts_budget=False,
            )
            return
        log_llm_call(self, method="shinka_evo", stage="meta_summary", operator="meta_summary",
                     sample_order=self._tot_sample_nums + 1, prompt=summaries_prompt,
                     response=summaries, generation=self._generation)
        insights_prompt = self._meta_prompt_insights(summaries)
        try:
            insights = self._meta_llm.draw_sample(insights_prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="meta_insights",
                operator="meta_insights",
                sample_order=self._tot_sample_nums + 1,
                prompt=insights_prompt,
                generation=self._generation,
                counts_budget=False,
            )
            return
        log_llm_call(self, method="shinka_evo", stage="meta_insights", operator="meta_insights",
                     sample_order=self._tot_sample_nums + 1, prompt=insights_prompt,
                     response=insights, generation=self._generation)
        recommendations_prompt = self._meta_prompt_recommendations(insights)
        try:
            recommendations = self._meta_llm.draw_sample(recommendations_prompt)
        except Exception as exc:
            record_sample_failure(
                self,
                exc,
                stage="meta_recommendations",
                operator="meta_recommendations",
                sample_order=self._tot_sample_nums + 1,
                prompt=recommendations_prompt,
                generation=self._generation,
                counts_budget=False,
            )
            return
        log_llm_call(self, method="shinka_evo", stage="meta_recommendations", operator="meta_recommendations",
                     sample_order=self._tot_sample_nums + 1, prompt=recommendations_prompt,
                     response=recommendations, generation=self._generation)
        self._meta_summary = (self._meta_summary + "\n\n" + summaries).strip() if self._meta_summary else summaries
        self._meta_scratch_pad = insights
        self._meta_recommendations = recommendations
        if recommendations:
            self._meta_recommendations_history.append(recommendations)
        self._register_event("meta_update", {
            "generation": self._generation,
            "num_programs": len(recent),
            "meta_summary": summaries,
            "meta_scratch_pad": insights,
            "meta_recommendations": recommendations,
        })

    def _current_meta_recommendation(self) -> str | None:
        if not self._meta_recommendations:
            return None
        if not self._sample_single_meta_rec:
            return self._meta_recommendations
        recommendations = self._parse_numbered_recommendations(self._meta_recommendations)
        if not recommendations:
            return self._meta_recommendations
        return self._rng.choice(recommendations)

    @staticmethod
    def _parse_numbered_recommendations(text: str) -> list[str]:
        lines = (text or "").strip().splitlines()
        recommendations = []
        current = []
        for line in lines:
            if re.match(r"^\d+\.\s+", line):
                if current:
                    recommendations.append("\n".join(current))
                current = [re.sub(r"^\d+\.\s+", "", line)]
            elif current:
                current.append(line)
        if current:
            recommendations.append("\n".join(current))
        return recommendations

    @staticmethod
    def _meta_prompt_individual(programs: list[ShinkaProgram]) -> str:
        blocks = []
        for program in programs:
            blocks.append(
                f"Program {program.id}\nscore={program.combined_score}, correct={program.correct}\n"
                f"```python\n{program.program}\n```\nfeedback={program.text_feedback}"
            )
        return "Summarize each evaluated program and its useful algorithmic ideas.\n\n" + "\n\n".join(blocks)

    def _meta_prompt_insights(self, summaries: str) -> str:
        previous = self._meta_scratch_pad or "*No previous memory state.*"
        return (
            "Synthesize global insights from recent program summaries and previous memory.\n\n"
            f"# Previous Memory\n{previous}\n\n# Recent Summaries\n{summaries}"
        )

    def _meta_prompt_recommendations(self, insights: str) -> str:
        best = self._archive.best_program
        best_block = best.program if best else ""
        return (
            "Generate numbered actionable recommendations for the next program mutation.\n\n"
            f"# Insights\n{insights}\n\n# Current Best Program\n```python\n{best_block}\n```"
        )

    def _register_function(self, program: ShinkaProgram) -> None:
        if self._profiler is not None:
            self._profiler.register_function(program.function, program=program.program)

    def _register_patch_attempt(
            self,
            generation: int,
            parent: ShinkaProgram,
            patch_type: str,
            patch: PatchResult,
    ) -> None:
        self._register_event("patch_attempt", {
            "generation": generation,
            "parent_id": parent.id,
            "patch_type": patch_type,
            "success": patch.success,
            "error": patch.error,
            **patch.metadata,
        })

    def _register_event(self, event_type: str, content: dict[str, Any]) -> None:
        if isinstance(self._profiler, ShinkaEvoProfiler):
            self._profiler.register_event(event_type, content)
        else:
            log_event(self, event=event_type, method="shinka_evo", **content)

    def _log_archive_state(self, phase: str) -> None:
        best = self._archive.best_program
        log_state(
            self,
            phase=phase,
            method="shinka_evo",
            generation=self._generation,
            sample_count=self._tot_sample_nums,
            archive_size=len(self._archive.archive_ids),
            num_islands=len(self._archive.islands),
            island_sizes=[len(self._archive.islands[idx]) for idx in sorted(self._archive.islands)],
            best_program_id=best.id if best is not None else None,
            best_score=best.combined_score if best is not None else None,
            meta_recommendations_count=len(self._meta_recommendations_history),
        )


ShinkaEvolve = ShinkaEvo

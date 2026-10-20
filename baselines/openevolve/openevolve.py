"""OpenEvolve's iteration loop, run serially over the vendored database and prompt sampler.

Local adaptations are documented in README.md.
"""

import copy
import math
import random
import re
import time

from core import SecureEvaluator
from core.code import function_from_source, parse_program
from core.llm import generate
from baselines.shared.checkpoint import save_checkpoint
from baselines.shared.observability import (
    close_llm, finish_profiler, init_observability, is_search_aborted, log_error,
    log_event, log_llm_call, log_state, record_sample_failure, reset_sample_failures,
)
from .upstream.config import DatabaseConfig, PromptConfig
from .upstream.database import Program, ProgramDatabase
from .upstream.prompt.sampler import PromptSampler
from .upstream.prompt.templates import TemplateManager
from .upstream.utils.code_utils import apply_diff_blocks, extract_diffs, format_diff_summary
from .upstream.utils.metrics_utils import safe_numeric_average

# openevolve.config.Config defaults.
DIFF_PATTERN = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
MAX_CODE_LENGTH = 10000
# Default ensemble weights in configs/default_config.yaml.
ENSEMBLE_WEIGHTS = (0.8, 0.2)
# The protocol keeps instance counts and indices out of generation prompts.
INSTANCE_DETAILS = ((re.compile(r" across \d+ instances"), ""),
                    (re.compile(r"\binstance \d+ "), "an instance "))
OBJECTIVE_NOTE = ("The evaluator minimizes the task objective. combined_score is its negative, "
                  "so a higher FITNESS SCORE is better. A failed evaluation scores -inf.")


class Database(ProgramDatabase):
    """Upstream database; diversity cache keys are str hashes, so keep their code for restarts."""

    def __init__(self, config):
        self.cached_codes = {}
        super().__init__(config)

    def _get_cached_diversity(self, program):
        self.cached_codes[hash(program.code)] = program.code
        return super()._get_cached_diversity(program)

    @property
    def diversity_by_code(self):
        return {self.cached_codes[key]: entry for key, entry in self.diversity_cache.items()}

    @diversity_by_code.setter
    def diversity_by_code(self, entries):
        self.diversity_cache = {hash(code): entry for code, entry in entries.items()}
        self.cached_codes = {hash(code): code for code in entries}


def fitness(program):
    """Raw minimized objective of a stored program, or None for a failed evaluation."""
    score = program.metrics.get("combined_score") if program else None
    return -score if score is not None and math.isfinite(score) else None


class OpenEvolve:
    def __init__(self, llm, evaluation, profiler=None, max_sample_nums=1000, *,
                 secondary_llm=None, num_islands=5, population_size=1000, archive_size=100,
                 migration_interval=50, migration_rate=0.1, num_top_programs=3,
                 num_diverse_programs=2, random_seed=0, debug_mode=False, **evaluator_options):
        if max_sample_nums < 1:
            raise ValueError("Budget must be positive")
        template = parse_program(str(evaluation.template_program))
        if template is None or len(template.functions) != 1:
            raise ValueError("OpenEvolve requires exactly one task entrypoint")
        self._template_code = str(evaluation.template_program)
        self._template_function = template.functions[0]
        self._llm, self._secondary_llm = llm, secondary_llm
        self._rng = random.Random(random_seed)
        # Large artifacts would otherwise be written under the working directory.
        self._database = Database(DatabaseConfig(
            population_size=population_size, archive_size=archive_size, num_islands=num_islands,
            migration_interval=migration_interval, migration_rate=migration_rate,
            random_seed=random_seed, artifact_size_threshold=1 << 30))
        self._sampler = PromptSampler(PromptConfig(
            system_message=self._system_message(evaluation.task_description),
            num_top_programs=num_top_programs, num_diverse_programs=num_diverse_programs))
        self._profiler = profiler
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_options)
        self._max_sample_nums = max_sample_nums
        self._debug_mode = debug_mode
        self._tot_sample_nums = 0
        self._evaluation_count = 0
        self._seeded = False
        self._pending_evaluation = None
        init_observability(self)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    @staticmethod
    def _system_message(task):
        """A user config replaces the system message; keep the default and add the task."""
        default = TemplateManager().get_template("system_message")
        return f"{default}\n\n{task.strip()}\n\n{OBJECTIVE_NOTE}"

    def _all_llms(self):
        return [self._llm] + ([self._secondary_llm] if self._secondary_llm is not None else [])

    @property
    def best_function(self):
        best = self._database.programs.get(self._database.best_program_id)
        if fitness(best) is None:
            return None
        function = function_from_source(best.code, self._template_function.name)
        if function is not None:
            function.score = fitness(best)
        return function

    def _has_budget(self):
        return not is_search_aborted(self) and self._tot_sample_nums < self._max_sample_nums

    def run(self):
        status = "finished"
        try:
            save_checkpoint(self, "openevolve", "ready")
            if not self._seeded and self._has_budget():
                self._tot_sample_nums += 1
                self._seeded = True
                self._queue(dict(code=self._template_code, operator="seed", changes="",
                                 iteration=0, island=None, parent=None, sample_time=0.0))
            while not is_search_aborted(self):
                if self._pending_evaluation is not None:
                    self._evaluate(self._pending_evaluation)
                    self._pending_evaluation = None
                    save_checkpoint(self, "openevolve", "generation")
                if not self._has_budget():
                    break
                self._step()
                if self._pending_evaluation is None:
                    save_checkpoint(self, "openevolve", "generation")
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "aborted"
            self._search_aborted = True
            log_error(self, "run", exc, method="openevolve", counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            self._log_state("final")
            finish_profiler(self, status=status, search_evaluations=self._evaluation_count)
            for llm in self._all_llms():
                close_llm(llm)

    def _step(self):
        database = self._database
        # Default controller: iterations are dealt to islands in round-robin order.
        iteration = self._tot_sample_nums
        island = (iteration - 1) % len(database.islands)
        parent, inspirations = database.sample_from_island(
            island, num_inspirations=self._sampler.config.num_diverse_programs)
        prompt = self._prompt(parent, inspirations, iteration, island)
        llm = self._llm
        if self._secondary_llm is not None:
            llm = self._rng.choices(self._all_llms(), weights=ENSEMBLE_WEIGHTS)[0]
        log_event(self, event="parent_selection", method="openevolve", iteration=iteration,
                  island=island, parent_id=self._candidate(parent),
                  inspiration_ids=[self._candidate(p) for p in inspirations])
        started = time.monotonic()
        try:
            result = generate(llm, prompt)
        except Exception as exc:
            record_sample_failure(self, exc, stage="patch", operator="diff", prompt=prompt,
                                  iteration=iteration, counts_budget=False)
            return
        sample_time = time.monotonic() - started
        reset_sample_failures(self)
        self._tot_sample_nums += 1
        log_llm_call(self, llm=llm, transport_calls=result["calls"], method="openevolve",
                     stage="patch", operator="diff", sample_order=self._tot_sample_nums,
                     iteration=iteration, island=island, parent_id=self._candidate(parent),
                     counts_budget=True)
        try:
            code, changes = self._apply(result["content"] or "", parent.code)
        except ValueError as exc:
            function = copy.deepcopy(self._template_function)
            function.score, function.operator = None, "rejected_diff"
            function.sample_time, function.parent_id = sample_time, self._candidate(parent)
            if self._profiler:
                self._profiler.register_function(function, budget_used=self._tot_sample_nums)
            log_event(self, event="diff_rejected", error=str(exc), iteration=iteration,
                      parent_id=self._candidate(parent), sample_order=self._tot_sample_nums,
                      counts_budget=True)
            return
        self._queue(dict(code=code, operator="diff", changes=changes, iteration=iteration,
                         island=island, sample_time=sample_time, parent=dict(
                             candidate=self._candidate(parent), generation=parent.generation,
                             island=parent.metadata.get("island", database.current_island),
                             metrics=copy.deepcopy(parent.metrics), id=parent.id)))

    def _prompt(self, parent, inspirations, iteration, island):
        """process_parallel._run_iteration_worker, reading the live database as its snapshot."""
        database, config = self._database, self._sampler.config
        parent_island = parent.metadata.get("island", database.current_island)
        programs = [database.programs[i] for i in database.islands[parent_island] if i in database.programs]
        programs.sort(key=lambda p: p.metrics.get("combined_score", safe_numeric_average(p.metrics)),
                      reverse=True)
        # The controller snapshot carries artifacts for the first max_snapshot_artifacts programs only.
        visible = list(database.programs)[:database.config.max_snapshot_artifacts]
        prompt = self._sampler.build_prompt(
            current_program=parent.code, parent_program=parent.code,
            program_metrics=parent.metrics,
            previous_programs=[p.to_dict() for p in programs[:config.num_top_programs]],
            top_programs=[p.to_dict() for p in
                          programs[:config.num_top_programs + config.num_diverse_programs]],
            inspirations=[p.to_dict() for p in inspirations], language="python",
            evolution_round=iteration, diff_based_evolution=True,
            program_artifacts=database.get_artifacts(parent.id) if parent.id in visible else None,
            feature_dimensions=database.config.feature_dimensions, island_id=island)
        return [{"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}]

    def _apply(self, response, parent_code):
        """Diff branch of _run_iteration_worker; every rejection there ends the iteration."""
        blocks = extract_diffs(response, DIFF_PATTERN)
        if not blocks:
            raise ValueError("No valid diffs found in response")
        code, applied = apply_diff_blocks(parent_code, blocks)
        if applied == 0:
            raise ValueError(f"None of the {len(blocks)} SEARCH block(s) matched the parent program")
        if code == parent_code:
            raise ValueError("Diff did not change the parent program")
        if len(code) > MAX_CODE_LENGTH:
            raise ValueError(f"Generated code exceeds maximum length ({len(code)} > {MAX_CODE_LENGTH})")
        config = self._sampler.config
        return code, format_diff_summary(blocks, max_line_len=config.diff_summary_max_line_len,
                                         max_lines=config.diff_summary_max_lines)

    def _queue(self, pending):
        self._pending_evaluation = pending
        save_checkpoint(self, "openevolve", "generated")

    @staticmethod
    def _metrics(outcome):
        """A user evaluator's metrics and artifacts; failures rank below every valid program."""
        score = outcome.result
        if outcome.failure_kind is None and score is not None and math.isfinite(float(score)):
            return {"combined_score": -float(score)}, {}
        metrics = {"error": 0.0, "combined_score": -math.inf}
        error = outcome.error or outcome.failure_kind or "evaluation failed"
        for pattern, replacement in INSTANCE_DETAILS:
            error = pattern.sub(replacement, error)
        artifacts = {"stderr": error, "failure_stage": "evaluation"}
        if outcome.failure_kind == "timeout":
            metrics["timeout"] = True
            artifacts.update(timeout=True, error_type="timeout")
        return metrics, artifacts

    def _evaluate(self, pending):
        database, parent = self._database, pending["parent"]
        outcome, seconds = self._evaluator.evaluate_program_record_time_with_details(pending["code"])
        self._evaluation_count += 1
        metrics, artifacts = self._metrics(outcome)
        valid = math.isfinite(metrics["combined_score"])
        function = (function_from_source(pending["code"], self._template_function.name)
                    or copy.deepcopy(self._template_function))
        function.score = -metrics["combined_score"] if valid else None
        function.operator, function.algorithm = pending["operator"], pending["changes"]
        function.sample_time, function.evaluate_time = pending["sample_time"], seconds
        function.parent_id = parent["candidate"] if parent else None
        function.depth = parent["generation"] + 1 if parent else 0
        if self._profiler:
            self._profiler.register_function(function, program=pending["code"],
                                             budget_used=self._tot_sample_nums)
        identity = self._profiler._num_samples if self._profiler else self._tot_sample_nums
        metadata = {"candidate_id": identity}
        if parent:
            metadata.update(changes=pending["changes"], parent_metrics=parent["metrics"],
                            island=parent["island"])
        program = Program(id=str(identity), code=pending["code"], language="python",
                          parent_id=parent["id"] if parent else None,
                          generation=function.depth, metrics=metrics,
                          iteration_found=pending["iteration"], metadata=metadata)
        if parent:
            database.add(program, iteration=pending["iteration"], target_island=pending["island"])
        else:
            database.add(program)
        database.store_artifacts(program.id, artifacts)
        island = program.metadata["island"]
        log_event(self, event="program_evaluated", node_id=identity, parent_id=function.parent_id,
                  iteration=pending["iteration"], island=island, fitness=function.score,
                  failure_kind=outcome.failure_kind, error=outcome.error,
                  instance_scores=outcome.instance_scores, sample_order=self._tot_sample_nums,
                  cell=next((k for k, v in database.island_feature_maps[island].items()
                             if v == program.id), None), counts_budget=True)
        if parent:
            database.increment_island_generation(island_idx=island)
            if database.should_migrate():
                before = set(database.programs)
                database.migrate_programs()
                log_event(self, event="island_migration", iteration=pending["iteration"],
                          island_generations=list(database.island_generations),
                          migrants=[(self._candidate(database.programs[i]),
                                     database.programs[i].metadata["island"])
                                    for i in database.programs if i not in before])
        self._log_state("program_evaluated")

    def _candidate(self, program):
        """Profiler candidate behind a stored program; copies keep or point to their source."""
        while program is not None and "candidate_id" not in program.metadata:
            program = self._database.programs.get(program.parent_id)
        return program.metadata["candidate_id"] if program is not None else None

    def _log_state(self, phase):
        database = self._database
        best = database.programs.get(database.best_program_id)
        log_state(self, phase=phase, method="openevolve", sample_count=self._tot_sample_nums,
                  best_program_id=self._candidate(best), best_fitness=fitness(best),
                  island_sizes=[len(island) for island in database.islands],
                  island_generations=list(database.island_generations),
                  archive_size=len(database.archive), program_count=len(database.programs))

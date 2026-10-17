"""Paper-based DeltaEvolve loop; unspecified policies are documented in README.md."""

import copy
import math
import random
import time

from core import SecureEvaluator
from core.code import parse_program
from core.llm import generate
from baselines.shared.checkpoint import save_checkpoint
from baselines.shared.observability import (
    close_llm, finish_profiler, init_observability, is_search_aborted, log_error,
    log_event, log_llm_call, log_state, record_sample_failure, reset_sample_failures,
)
from .population import Population
from .prompt import build_prompt, parse_response, program_from_code


class DeltaEvolve:
    def __init__(self, llm, evaluation, profiler=None, max_sample_nums=1000, *,
                 embedding_fn, secondary_llm=None, recent_window=10,
                 population_size=40, archive_size=20, num_islands=3,
                 top_k=3, diverse_m=2, migration_interval=10, migration_rate=0.1,
                 random_seed=0, debug_mode=False, **evaluator_options):
        if max_sample_nums < 1 or recent_window < 1 or min(top_k, diverse_m) < 0:
            raise ValueError("Budget/window must be positive; history counts must be nonnegative")
        self._template_program = parse_program(str(evaluation.template_program))
        if self._template_program is None or len(self._template_program.functions) != 1:
            raise ValueError("DeltaEvolve requires exactly one task entrypoint")
        self._template_program = program_from_code(str(evaluation.template_program), self._template_program)
        self._task = evaluation.task_description + (
            "\nThe evaluator minimizes the raw objective. FITNESS SCORE is its negative; "
            "maximize FITNESS SCORE while preserving the task interface and constraints.")
        self._llm, self._secondary_llm = llm, secondary_llm
        self._embedding_fn = embedding_fn
        self._rng = random.Random(random_seed)
        self._population = Population(self._rng, num_islands=num_islands,
            population_size=population_size, archive_size=archive_size,
            migration_interval=migration_interval, migration_rate=migration_rate)
        self._profiler = profiler
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_options)
        self._max_sample_nums = max_sample_nums
        self._recent_window, self._top_k, self._diverse_m = recent_window, top_k, diverse_m
        self._debug_mode = debug_mode
        self._tot_sample_nums = self._iteration = 0
        self._evaluation_count = 0
        self._seeded = False
        self._pending_evaluation = None
        self._pending_node = None
        init_observability(self)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    def _all_llms(self):
        return [self._llm] + ([self._secondary_llm] if self._secondary_llm is not None else [])

    @property
    def best_function(self):
        if not self._population.archive_ids:
            return None
        best = self._population.nodes[self._population.archive_ids[0]]
        function = program_from_code(best["code"], self._template_program).functions[0]
        function.score = best["fitness"]
        return function

    def _has_budget(self):
        return not is_search_aborted(self) and self._tot_sample_nums < self._max_sample_nums

    def run(self):
        status = "finished"
        try:
            save_checkpoint(self, "deltaevolve", "ready")
            if not self._seeded and self._has_budget():
                self._tot_sample_nums += 1
                self._seeded = True
                self._queue_evaluation(self._template_program, "", "", None, 0, 0.0)
            while not is_search_aborted(self):
                if self._pending_evaluation is not None:
                    pending = self._pending_evaluation
                    parent = self._population.nodes.get(pending["parent_id"])
                    self._evaluate(program_from_code(pending["code"], self._template_program),
                        pending["summary"], pending["plan"], parent,
                        pending["island"], pending["sample_time"])
                    self._pending_evaluation = None
                    save_checkpoint(self, "deltaevolve", "evaluated")
                if self._pending_node is not None:
                    self._commit_pending()
                    save_checkpoint(self, "deltaevolve", "generation")
                if not self._population.archive_ids:
                    raise ValueError("Initial task program has no valid evaluation; cannot select a parent")
                if not self._has_budget():
                    break
                self._step()
                if self._pending_evaluation is None:
                    save_checkpoint(self, "deltaevolve", "generation")
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "aborted"
            self._search_aborted = True
            log_error(self, "run", exc, method="deltaevolve", counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            log_state(self, phase="final", method="deltaevolve", sample_count=self._tot_sample_nums,
                      iteration=self._iteration, archive_ids=self._population.archive_ids,
                      island_sizes=[len(cells) for cells in self._population.islands])
            finish_profiler(self, status=status, search_evaluations=self._evaluation_count)
            close_llm(self._embedding_fn)
            for llm in self._all_llms():
                close_llm(llm)

    def _step(self):
        island, parent, top, diverse = self._population.select(self._top_k, self._diverse_m)
        prompt, rendered = build_prompt(self._task, parent, top, diverse,
                                       self._iteration + 1, self._recent_window)
        llm = self._llm
        if self._secondary_llm is not None and self._rng.random() >= 0.8:
            llm = self._secondary_llm
        log_event(self, event="parent_selection", method="deltaevolve", island=island,
                  iteration=self._iteration + 1, parent_id=parent["id"], history=rendered)
        started = time.monotonic()
        try:
            result = generate(llm, prompt)
        except Exception as exc:
            record_sample_failure(self, exc, stage="patch", prompt=prompt, counts_budget=False)
            return
        sample_time = time.monotonic() - started
        reset_sample_failures(self)
        self._tot_sample_nums += 1
        self._iteration += 1
        log_llm_call(self, llm=llm, transport_calls=result["calls"], method="deltaevolve",
                     stage="patch", operator="delta_mutation", sample_order=self._tot_sample_nums,
                     parent_id=parent["id"], counts_budget=True)
        try:
            program, summary, plan = parse_response(result["content"], parent["code"], self._template_program)
        except (ValueError, SyntaxError, TypeError) as exc:
            function = copy.deepcopy(self._template_program.functions[0])
            function.score, function.operator = None, "invalid_delta_patch"
            function.sample_time = sample_time
            function.parent_id = parent["id"]
            if self._profiler:
                self._profiler.register_function(function, budget_used=self._tot_sample_nums)
            log_event(self, event="patch_rejected", error=str(exc), parent_id=parent["id"],
                      sample_order=self._tot_sample_nums, counts_budget=True)
            for transfer in self._population.migrate(self._iteration):
                log_event(self, event="island_migration", iteration=self._iteration, **transfer)
            return
        self._queue_evaluation(program, summary, plan, parent, island, sample_time)

    def _queue_evaluation(self, program, summary, plan, parent, island, sample_time):
        self._pending_evaluation = dict(code=str(program), summary=summary, plan=plan,
            parent_id=parent["id"] if parent else None, island=island, sample_time=sample_time)
        save_checkpoint(self, "deltaevolve", "generated")

    def _evaluate(self, program, summary, plan, parent, island, sample_time):
        outcome, seconds = self._evaluator.evaluate_program_record_time_with_details(program)
        self._evaluation_count += 1
        score = outcome.result
        valid = outcome.failure_kind is None and score is not None and math.isfinite(float(score))
        function = copy.deepcopy(program.functions[0])
        function.score = float(score) if valid else None
        function.operator = "delta_mutation" if parent else "seed"
        function.algorithm, function.sample_time, function.evaluate_time = summary, sample_time, seconds
        function.parent_id = parent["id"] if parent else None
        function.depth = parent["depth"] + 1 if parent else 0
        if self._profiler:
            self._profiler.register_function(function, program=str(program), budget_used=self._tot_sample_nums)
        change = "seed" if parent is None else (
            "failed" if not valid else "improved" if score < parent["fitness"] else
            "degraded" if score > parent["fitness"] else "unchanged")
        identity = self._profiler._num_samples if self._profiler else self._tot_sample_nums
        node = {"id": identity, "parent_id": function.parent_id, "depth": function.depth,
                "iteration": self._iteration, "island": island, "code": str(program),
                "fitness": function.score, "summary": summary, "plan": plan, "change": change}
        log_event(self, event="delta_evaluated", **node, instance_scores=outcome.instance_scores,
                  failure_kind=outcome.failure_kind, error=outcome.error,
                  sample_order=self._tot_sample_nums, counts_budget=True)
        if valid:
            self._pending_node = node
        else:
            for transfer in self._population.migrate(self._iteration):
                log_event(self, event="island_migration", iteration=self._iteration, **transfer)

    def _commit_pending(self):
        node = self._pending_node
        started = time.monotonic()
        try:
            embedding = self._embedding_fn(node["code"])
            node["cell"] = self._population.cell(embedding)
        except Exception as exc:
            log_event(self, event="embedding_call", node_id=node["id"], error=str(exc),
                      seconds=time.monotonic() - started, counts_budget=False,
                      usage=getattr(self._embedding_fn, "last_usage", {}))
            raise
        log_event(self, event="embedding_call", node_id=node["id"], counts_budget=False,
                  seconds=time.monotonic() - started,
                  usage=getattr(self._embedding_fn, "last_usage", {}))
        self._population.add(node, node["island"])
        if node["parent_id"] is None:
            for island in range(1, len(self._population.islands)):
                self._population._insert(node["id"], island)
        else:
            for transfer in self._population.migrate(self._iteration):
                log_event(self, event="island_migration", iteration=self._iteration, **transfer)
        self._pending_node = None
        log_event(self, event="database_update", node_id=node["id"],
                  cell=node["cell"], archive_ids=self._population.archive_ids,
                  island_sizes=[len(cells) for cells in self._population.islands])

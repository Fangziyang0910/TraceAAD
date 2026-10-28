"""TraceAAD: ordinary Refine, Explore and Crossover over program individuals."""

from datetime import datetime
from pathlib import Path
from core.llm import generate, ModelCallError
from core.scheduling import SchedulerError
from .storage import write_json
from .config import Config
from .delivery import JSON_RESPONSE_FORMAT, DeliveryError, parse_json_response
from .instance_evaluation import InstanceProgramEvaluator
from .selection import choose_reference, sample_parent
from .state import Facts, Progress
from .program import Program
from .prompts import PromptBuilder


class TraceAAD:
    METHOD = "traceaad"

    def __init__(self, *, evaluation, llm, run_dir, config=None, task=None):
        self.config = config or Config()
        self.llm = llm
        if (
            not callable(getattr(llm, "count_prompt_tokens", None))
            or not callable(getattr(llm, "count_tokens", None))
        ):
            raise ValueError("search requires serving token counts")
        self.run_dir, self.template = (Path(run_dir), str(evaluation.template_program))
        self.facts = Facts(run_dir)
        self.programs = self.facts.programs
        self.prompts = PromptBuilder(llm, task, evaluation, self.programs, self.config)
        evaluation_options = {
            "seeds": self.config.evaluation_seeds,
            "timeout_seconds": self.config.eval_timeout_seconds,
            "n_workers": self.config.eval_workers,
            "scheduler_socket": self.config.scheduler_socket,
        }
        self.training = InstanceProgramEvaluator(evaluation, role="search", **evaluation_options)
        saved = self.facts.state or {}
        self.progress = Progress.restore(saved, self.config.seed)

    @property
    def phase(self):
        return self.progress.phase

    @property
    def archive(self):
        return self.facts.valid

    @property
    def budget_used(self):
        return self.progress.generations

    @property
    def evaluation_calls(self):
        return len(self.facts.evaluations)

    def _save(self, **record):
        self.facts.commit(self.progress.snapshot(), **record)

    def _generate(self, request):
        """Generate one program reply and record every transport attempt."""
        try:
            details = generate(
                self.llm,
                request["prompt"],
                max_tokens=self.config.output_tokens,
                response_format=JSON_RESPONSE_FORMAT,
            )
        except ModelCallError as exc:
            self.progress.record_calls(exc.calls, request, self.config.output_tokens)
            self._save(calls=exc.calls)
            raise
        self.progress.record_calls(details["calls"], request, self.config.output_tokens)
        return details

    def _create_program(self, request, *, parent=None, reference=None, repair_of=None):
        details = self._generate(request)
        calls = details["calls"]
        self.progress.generations += 1
        if repair_of is not None:
            self.progress.repair_id = None
        if self.phase == "roots":
            self.progress.init_attempts += 1
        try:
            code, idea, analysis = parse_json_response(
                details.get("content", ""),
                details.get("finish_reason"),
                repair=repair_of is not None,
            )
        except DeliveryError as exc:
            self._save(
                calls=calls,
                delivery_error={
                    "error": str(exc),
                    "action": request["action"],
                    "parent_id": parent.id if parent else None,
                },
            )
            return None
        program = Program(
            id=len(self.programs) + 1,
            generation=self.progress.generations,
            code=code,
            idea=parent.idea if repair_of is not None else idea,
            analysis=analysis,
            action=request["action"],
            parent_id=parent.id if parent else None,
            reference_id=reference.id if reference else None,
            repair_of=repair_of,
            depth=parent.depth + 1 if parent else 0,
            request_id=calls[-1]["request_id"],
            call_ids=[c["request_id"] for c in calls],
            request={k: v for k, v in request.items() if k != "prompt"},
        )
        self.programs[program.id] = program
        return self._evaluate_program(program, calls, new=True)

    def _repair(self):
        failed = self.programs[self.progress.repair_id]
        request = self.prompts.repair(failed)
        return self._create_program(request, parent=failed, repair_of=failed.id)

    def _is_root(self, node):
        """An initial algorithm: no valid program precedes it on its formation path
        (a repaired initial program starts from its failed first version)."""
        parent = self.programs.get(node.parent_id)
        while parent is not None:
            if parent.valid:
                return False
            parent = self.programs.get(parent.parent_id)
        return True

    def _roots(self):
        roots = [node for node in self.archive.values() if self._is_root(node)]
        if (
            len(roots) >= self.config.roots
            or self.progress.init_attempts >= self.config.init_attempt_limit
        ):
            self.progress.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        request = self.prompts.initial(roots)
        self._create_program(request)

    def _search(self, sampled=None):
        eligible = list(self.archive.values())
        if self.budget_used >= self.config.budget or not eligible:
            self.progress.phase = "freeze"
            self._save()
            return
        self._step(eligible, sampled)

    def _step(self, eligible, sampled=None, **record):
        """One ordinary attempt from a parent sampled among ``eligible``; ``record`` joins the request."""
        parent, selection = sample_parent(eligible, self.programs, self.progress.rng["parent"])
        if sampled is None:
            sampled = self.progress.rng["action"].choices(
                list(self.config.operators), list(self.config.operators.values())
            )[0]
        action = sampled
        reference, reference_selection = (None, None)
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(
                parent, self._references(parent), self.progress.rng["reference"]
            )
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        request = self.prompts.build(action, parent, reference=reference)
        request.update(
            sampled_action=sampled,
            fallbacks=flags,
            parent_id=parent.id,
            reference_id=reference.id if reference else None,
            selection=selection,
            reference_selection=reference_selection,
            **record,
        )
        self._create_program(request, parent=parent, reference=reference)

    def _references(self, parent):
        """The programs a Crossover reference for ``parent`` is chosen from."""
        return self.archive

    def _freeze(self):
        """Freeze the best recorded training individual without another evaluation."""
        best = min(self.archive.values(), key=lambda p: (p.fitness, p.id), default=None)
        if best is None:
            self.progress.phase = "no_valid_root"
            self._save()
        else:
            self._finish(best)

    def _finish(self, node):
        """Freeze one winner before any held-out evaluation."""
        self.progress.selected_id = node.id
        (self.run_dir / "best_program.py").write_text(node.code, encoding="utf-8")
        write_json(
            self.run_dir / "selection.json",
            {"criterion": "training", "results": [], "finalists": [], "selected_node": node.id},
        )
        self.progress.phase = "finished"
        self._save()

    def _summary(self, status, error=None):
        archive = self.archive
        best = (
            archive.get(self.progress.selected_id)
            if self.phase == "finished"
            else min(archive.values(), key=lambda n: (n.fitness, n.id), default=None)
        )
        if best is not None:
            best = {k: v for k, v in best.metadata().items() if k not in {"failure", "repaired"}}
        summary = {
            "status": status,
            "phase": self.phase,
            "method": self.METHOD,
            "final_selection": "training",
            "budget": self.config.budget,
            "budget_used": self.progress.generations,
            "budget_axis": "候选尝试",
            "init_attempts": self.progress.init_attempts,
            "num_nodes": len(archive),
            "num_failed_programs": len(self.programs) - len(archive),
            "num_roots": sum(self._is_root(n) for n in archive.values()),
            "model_calls": self.progress.model_calls,
            "evaluation_calls": self.evaluation_calls,
            "search_evaluations": sum((e["role"] == "search" for e in self.facts.evaluations)),
            "selection_evaluations": 0,
            "input_tokens": self.progress.input_tokens,
            "output_tokens": self.progress.output_tokens,
            "service_failures": self.progress.service_failures,
            "started_at": self.progress.started_at,
            "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "seconds": self.progress.seconds,
            "best": best,
            "finalists": [],
            "error": error,
        }
        self.facts.save_summary(summary)
        return summary

    def run(self):
        if self.progress.phase in {"finished", "no_valid_root"}:
            previous = self.facts.load_summary()
            if previous:
                return previous
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.progress.phase in {"roots", "search", "freeze"}:
                if self.progress.pending_id is not None:
                    self._evaluate_program(self.programs[self.progress.pending_id])
                    continue
                if self.progress.repair_id is not None:
                    self._repair()
                elif self.progress.phase == "roots":
                    self._roots()
                elif self.progress.phase == "search":
                    self._search()
                else:
                    self._freeze()
        except SchedulerError as exc:
            return self._summary("service_unavailable", str(exc))
        except ModelCallError as exc:
            if not exc.transient:
                raise
            return self._summary(
                "service_unavailable",
                f"model service unavailable after {len(exc.calls)} calls: {exc}",
            )
        return self._summary(self.progress.phase)

    def _evaluate_program(self, program, calls=(), *, new=False):
        try:
            evaluations = program.evaluate(self.training, self.template)
        except SchedulerError:
            self.progress.pending_id = program.id
            self._save(program=program, new=new, calls=calls)
            raise
        self.progress.repair_id = (
            program.id
            if not program.valid
            and program.repair_of is None
            and (
                self.progress.init_attempts < self.config.init_attempt_limit
                if self.phase == "roots"
                else self.progress.generations < self.config.budget
            )
            else None
        )
        self.progress.pending_id = None
        self._save(program=program, new=new, calls=calls, evaluations=evaluations)
        if self.progress.repair_id is not None:
            return self._repair()
        return program if program.valid else None

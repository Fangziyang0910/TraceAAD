"""V12.1 search over program individuals, including failed and repaired versions."""

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import time
from core.llm import generate, ModelCallError
from core.scheduling import SchedulerError
from .storage import write_json
from .config import REVISION, INDIVIDUAL_FORMAT, Config
from .delivery import (
    JSON_REPLY,
    DeliveryError,
    parse_json_response,
    read_json,
    validate_source,
)
from .evaluation import fingerprint
from .history import final_program
from .instance_evaluation import InstanceProgramEvaluator
from .selection import better, choose_reference, sample_parent
from .state import Facts, Progress
from .program import Program
from .linking import choose_linked, normal, parse_match, rank_candidates
from .prompts import PromptBuilder
from .verification import parse_dropped, parse_verdict


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _tuple_tree(value):
    return tuple((_tuple_tree(v) for v in value)) if isinstance(value, list) else value


class TraceAADV121:
    METHOD = "v12_1"
    Evaluator = InstanceProgramEvaluator
    Config = Config
    PromptBuilder = PromptBuilder
    MEASURE_CALLS = True
    REPLACE_FAILED_FINALISTS = True

    def __init__(
        self,
        *,
        evaluation,
        llm,
        run_dir,
        config=None,
        task=None,
        selection_evaluation=None,
        embed=None,
    ):
        self.config, self.llm, self.task = (config or self.Config(), llm, task)
        if (
            getattr(llm, "chars_per_token", None) is not None
            or not callable(getattr(llm, "count_prompt_tokens", None))
            or (not callable(getattr(llm, "count_tokens", None)))
        ):
            raise ValueError("search requires serving token counts")
        self.run_dir, self.template = (Path(run_dir), str(evaluation.template_program))
        self.facts = Facts(run_dir)
        self.programs = self.facts.programs
        self.prompts = self.PromptBuilder(
            llm, task, evaluation, self.programs, self.config
        )
        self.training = self.Evaluator(
            evaluation,
            self.config.evaluation_seeds,
            "search",
            measure_calls=self.MEASURE_CALLS,
            **self._evaluation_options(),
        )
        self.selection = (
            self.Evaluator(
                selection_evaluation,
                self.config.evaluation_seeds,
                "selection",
                measure_calls=self.MEASURE_CALLS,
                **self._evaluation_options(),
            )
            if selection_evaluation is not None
            else None
        )
        if self.selection:
            if str(selection_evaluation.template_program) != self.template:
                raise ValueError("selection has a different task interface")
            search_data = getattr(evaluation, "_datasets", None)
            selection_data = getattr(selection_evaluation, "_datasets", None)
            if self.training.environment == self.selection.environment or (
                search_data is not None
                and selection_data is not None
                and (fingerprint(search_data) == fingerprint(selection_data))
            ):
                raise ValueError("selection must use distinct fixed data")
        self.identity = json.loads(
            json.dumps(
                {
                    "revision": REVISION,
                    "config": asdict(self.config),
                    "task": task,
                    "protocol": self.training.protocol,
                    "selection_protocol": self.selection.protocol
                    if self.selection
                    else None,
                }
            )
        )
        saved = self.facts.state or {}
        if (
            saved
            and getattr(self.training, "function_budget_scope", None) == "mean_instance"
            and (saved.get("identity", {}).get("revision") != REVISION)
        ):
            raise ValueError(
                "mean CPU budget requires a new batch; saved evaluation revision differs"
            )
        if (
            getattr(evaluation, "interface", None)
            and saved
            and (saved.get("identity", {}).get("protocol") != self.training.protocol)
        ):
            raise ValueError(
                "complete-solver resume requires the same task interface and evaluation protocol"
            )
        self.progress = Progress(
            **{
                name: saved[name]
                for name in Progress.__dataclass_fields__
                if name in saved
            }
        )
        self.parent_rng = random.Random(f"{self.METHOD}:{self.config.seed}:parent")
        self.action_rng = random.Random(f"{self.METHOD}:{self.config.seed}:action")
        self.reference_rng = random.Random(
            f"{self.METHOD}:{self.config.seed}:reference"
        )
        for name, state in saved.get("rng", {}).items():
            getattr(self, name + "_rng").setstate(_tuple_tree(state))
        self._clock = time.monotonic()
        self.training.scheduler_socket = self.config.scheduler_socket
        if self.selection:
            self.selection.scheduler_socket = self.config.scheduler_socket
        if self.config.linking and embed is None:
            raise ValueError("re-proposal linking requires an embedding function")
        self.embed, self._vectors = (embed, {})
        if self.config.linking:
            self.identity["embedding"] = getattr(embed, "settings", None)
        saved = self.facts.state or {}
        saved_identity = {
            k: v for k, v in saved.get("identity", {}).items() if k != "mechanism"
        }
        if saved and saved_identity != self.identity:
            raise ValueError(
                "V12.1 resume requires the same configuration, task, protocol and embedding"
            )

    @property
    def phase(self):
        return self.progress.phase

    def _evaluation_options(self):
        return {
            "timeout_seconds": self.config.eval_timeout_seconds,
            "n_workers": self.config.eval_workers,
        }

    @property
    def archive(self):
        return self.facts.valid

    @property
    def budget_used(self):
        return self.progress.generations

    @property
    def evaluation_calls(self):
        return len(self.facts.evaluations)

    def _state(self):
        return {
            **asdict(self.progress),
            "identity": self.identity,
            "elapsed": self.progress.elapsed + time.monotonic() - self._clock,
            "individual_format": INDIVIDUAL_FORMAT,
            "rng": {
                name: getattr(self, name + "_rng").getstate()
                for name in ("parent", "action", "reference")
            },
        }

    def _save(self, **record):
        self.facts.commit(self._state(), **record)

    def _call(self, calls, prompt, input_tokens, stage=None, **kwargs):
        """Generate and account for a request, preserving earlier stages if it fails."""
        try:
            details, failed = (generate(self.llm, prompt, **kwargs), None)
            made = details["calls"]
        except ModelCallError as exc:
            details, made, failed = ({}, exc.calls, exc)
        for call in made:
            self.progress.model_calls += 1
            call["request_id"] = self.progress.model_calls
            if stage is not None:
                call["stage"] = stage
            usage = call["usage"]
            self.progress.input_tokens += (
                usage.get("prompt_tokens", input_tokens) or input_tokens
            )
            self.progress.output_tokens += (
                usage.get(
                    "completion_tokens",
                    kwargs["max_tokens"] if not call["error"] else 0,
                )
                or 0
            )
            self.progress.service_failures += bool(call["error"])
        calls.extend(made)
        if failed:
            self._save(calls=calls)
            if failed.transient:
                raise RuntimeError(
                    f"model service unavailable after {len(calls)} calls: {failed}"
                ) from failed
            raise failed
        return details

    def _generate(self, request):
        if request["action"] == "Complete":
            calls = []
            reply = self._call(
                calls,
                request["prompt"],
                request["input_tokens"],
                "complete",
                max_tokens=self.config.output_tokens,
                response_format=JSON_REPLY,
            )
            return (
                calls[-1]["request_id"],
                {**reply, "design": request["design"]},
                calls,
            )
        if not (self.config.linking and request["action"] == "Explore"):
            return self._generate_code(request)
        parent, calls = (self.programs[request["parent_id"]], [])
        first = self._call(
            calls,
            request["prompt"],
            request["input_tokens"],
            "propose",
            max_tokens=self.config.output_tokens,
            response_format=JSON_REPLY,
        )
        try:
            reply = read_json(
                first.get("content", ""),
                first.get("finish_reason"),
                ("analysis", "design"),
            )
            analysis, design = (reply["analysis"].strip(), normal(reply["design"]))
            if not design:
                raise DeliveryError("proposal needs a nonempty design")
        except DeliveryError as exc:
            return (
                calls[-1]["request_id"],
                {**first, "delivery_error": str(exc)},
                calls,
            )
        proposal = f"Analysis: {analysis}\nDesign: {design}"
        link, earlier = self._link(calls, design, parent)
        link["proposal"] = proposal
        implement = self.prompts.implement(request["prompt"], proposal, earlier)
        request["link"] = link
        second = self._call(
            calls,
            implement["prompt"],
            implement["input_tokens"],
            "implement",
            max_tokens=self.config.output_tokens,
            response_format=JSON_REPLY,
        )
        if self.config.completion == "resample":
            link["implement_prompt"] = calls[-1]["prompt"]
        return (
            calls[-1]["request_id"],
            {**second, "design": design, "analysis": analysis},
            calls,
        )

    def _create_program(self, request, *, parent=None, reference=None, repair_of=None):
        rid, details, calls = self._generate(request)
        self.progress.generations += 1
        if repair_of is not None:
            self.progress.repair_id = None
        if self.phase == "roots":
            self.progress.init_attempts += 1
        inherited = repair_of is not None or "design" in details
        try:
            if details.get("delivery_error"):
                raise DeliveryError(details["delivery_error"])
            code, idea, delivery = parse_json_response(
                details.get("content", ""),
                details.get("finish_reason"),
                ("code",) if inherited else ("design", "code"),
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
            idea=parent.idea if repair_of is not None else details.get("design", idea),
            analysis=details.get("analysis", delivery["analysis"]),
            action=request["action"],
            parent_id=parent.id if parent else None,
            reference_id=reference.id if reference else None,
            repair_of=repair_of,
            depth=parent.depth + 1 if parent else 0,
            request_id=rid,
            call_ids=[c["request_id"] for c in calls],
            request={
                k: v
                for k, v in request.items()
                if k not in {"prompt", "link", "exploration"}
            },
            link=request.get("link"),
            exploration=request.get("exploration"),
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
        if self.progress.repair_id is not None:
            self._repair()
            return
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

    def _choose_parent(self, eligible):
        return sample_parent(eligible, self.programs, self.parent_rng)

    def _ordinary_search(self, sampled=None):
        eligible = list(self.archive.values())
        if not eligible:
            self.progress.phase = "freeze"
            self._save()
            return
        parent, selection = self._choose_parent(eligible)
        if sampled is None:
            sampled = self.action_rng.choices(
                list(self.config.operators), list(self.config.operators.values())
            )[0]
        action = sampled
        reference, reference_selection = (None, None)
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(
                parent, self.archive, self.reference_rng
            )
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        request = self.prompts.build(action, parent, reference=reference)
        if action == "Explore":
            request["exploration"] = {"id": len(self.facts.explorations) + 1, "step": 0}
        request.update(
            sampled_action=sampled,
            fallbacks=flags,
            parent_id=parent.id,
            reference_id=reference.id if reference else None,
            selection=selection,
            reference_selection=reference_selection,
        )
        self._create_program(request, parent=parent, reference=reference)

    def _open_exploration(self):
        tagged = [
            p for p in self.programs.values() if p.exploration and p.repair_of is None
        ]
        if not tagged:
            return None
        eid = max((p.exploration["id"] for p in tagged))
        if eid in self.facts.explorations:
            return None
        versions = sorted(
            (p for p in tagged if p.exploration["id"] == eid), key=lambda p: p.id
        )
        proposal, development = (versions[0], versions[1:])
        first = final_program(proposal, self.programs)
        proposed = first if first.valid else None
        valid = [final_program(p, self.programs) for p in versions]
        best = min(
            (p for p in valid if p.valid), key=lambda p: (p.fitness, p.id), default=None
        )
        return {
            "id": eid,
            "proposal": proposal,
            "proposed": proposed,
            "development": development,
            "best": best,
        }

    def _ranking(self):
        return [
            n.id for n in sorted(self.archive.values(), key=lambda n: (n.fitness, n.id))
        ]

    def _freeze(self):
        """Recheck the training top programs and freeze the best valid result."""
        if self.selection:
            return self._freeze_selection()
        ranking = self._ranking()
        results = []
        for node_id in ranking[: self.config.final_candidates]:
            node = self.archive[node_id]
            outcome = self.training.evaluate(
                node.code, node.key, **self._evaluation_options()
            )
            results.append(
                {
                    "node_id": node_id,
                    **{k: v for k, v in outcome.items() if k != "evaluations"},
                }
            )
            self._save(evaluations=outcome["evaluations"])
        self.progress.finalists = [r["node_id"] for r in results]
        self.progress.selection_results = results
        scored = [r for r in results if r["fitness"] is not None]
        if not scored:
            self.progress.phase = "selection_failed" if ranking else "no_valid_root"
            self._save()
            return
        chosen = min(scored, key=lambda r: (r["fitness"], ranking.index(r["node_id"])))[
            "node_id"
        ]
        self._finish(self.archive[chosen], "training")

    def _finish(self, node, criterion):
        """Freeze one winner before any held-out evaluation."""
        self.progress.selected_id = node.id
        (self.run_dir / "best_program.py").write_text(node.code, encoding="utf-8")
        write_json(
            self.run_dir / "selection.json",
            {
                "criterion": criterion,
                "selection_protocol": self.selection.protocol
                if self.selection
                else None,
                "results": self.progress.selection_results,
                "finalists": self.progress.finalists,
                "selected_node": node.id,
                "selected_key": node.key,
            },
        )
        self.progress.phase = "finished"
        self._save()

    def _select(self):
        """Evaluate finalists on the selection set; a finalist that fails there is
        replaced by the next program in the training ranking, up to as many
        replacements as there are finalists."""
        if len(self.progress.selection_results) < len(self.progress.finalists):
            node = self.archive[
                self.progress.finalists[len(self.progress.selection_results)]
            ]
            outcome = self.selection.evaluate(
                node.code, node.key, **self._evaluation_options()
            )
            failure = outcome["failure"]
            self.progress.selection_results.append(
                {
                    "node_id": node.id,
                    **{k: v for k, v in outcome.items() if k != "evaluations"},
                }
            )
            if (
                failure
                and self.REPLACE_FAILED_FINALISTS
                and (len(self.progress.finalists) < 2 * self.config.final_candidates)
            ):
                remaining = [
                    i for i in self._ranking() if i not in self.progress.finalists
                ]
                if remaining:
                    self.progress.finalists.append(remaining[0])
            self._save(evaluations=outcome["evaluations"])
            return
        valid = [r for r in self.progress.selection_results if r["fitness"] is not None]
        if not valid:
            self.progress.phase = "selection_failed"
        else:
            selected = min(
                valid,
                key=lambda r: (
                    r["fitness"],
                    self.progress.finalists.index(r["node_id"]),
                ),
            )
            self._finish(self.archive[selected["node_id"]], "validation")
            return
        self._save()

    def _summary(self, status, error=None):
        best = (
            self.archive.get(self.progress.selected_id)
            if self.phase == "finished"
            else min(
                self.archive.values(), key=lambda n: (n.fitness, n.id), default=None
            )
        )
        if best is not None:
            best = {
                **{
                    k: v
                    for k, v in asdict(best).items()
                    if k not in {"code", "failure"}
                },
                "key": best.key,
                "score": best.score,
                "valid": best.valid,
                "selection_fitness": next(
                    (
                        r["fitness"]
                        for r in self.progress.selection_results
                        if r["node_id"] == best.id
                    ),
                    None,
                ),
            }
            if (
                self.phase == "finished"
                and self.selection is None
                and (best["selection_fitness"] is not None)
            ):
                best.update(
                    fitness=best["selection_fitness"], score=best["selection_fitness"]
                )
        summary = {
            "status": status,
            "phase": self.phase,
            "method": self.METHOD,
            "revision": REVISION,
            "final_selection": "validation" if self.selection else "training",
            "budget": self.config.budget,
            "budget_used": self.progress.generations,
            "budget_axis": "候选尝试",
            "init_attempts": self.progress.init_attempts,
            "num_nodes": len(self.archive),
            "num_failed_programs": len(self.programs) - len(self.archive),
            "num_roots": sum((self._is_root(n) for n in self.archive.values())),
            "model_calls": self.progress.model_calls,
            "evaluation_calls": self.evaluation_calls,
            "search_evaluations": sum(
                (e["role"] == "search" for e in self.facts.evaluations)
            ),
            "selection_evaluations": sum(
                (e["role"] == "selection" for e in self.facts.evaluations)
            ),
            "input_tokens": self.progress.input_tokens,
            "output_tokens": self.progress.output_tokens,
            "service_failures": self.progress.service_failures,
            "started_at": self.progress.started_at,
            "finished_at": now(),
            "seconds": self.progress.elapsed + time.monotonic() - self._clock,
            "best": best,
            "finalists": self.progress.finalists,
            "error": error,
        }
        self.facts.save_summary(summary)
        return summary

    def run(self):
        if self.progress.phase in {
            "finished",
            "search_complete",
            "selection_failed",
            "no_valid_root",
        }:
            previous = self.facts.load_summary()
            if previous:
                return previous
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            while self.progress.phase in {"roots", "search", "freeze", "selection"}:
                if self.progress.pending_id is not None:
                    self._evaluate_program(self.programs[self.progress.pending_id])
                    continue
                if self.progress.phase == "roots":
                    self._roots()
                elif self.progress.phase == "search":
                    self._search()
                elif self.progress.phase == "freeze":
                    self._freeze()
                else:
                    self._select()
        except SchedulerError as exc:
            return self._summary("service_unavailable", str(exc))
        except RuntimeError as exc:
            if "model service unavailable" in str(exc):
                return self._summary("service_unavailable", str(exc))
            raise
        return self._summary(self.progress.phase)

    def _vector(self, text):
        if text not in self._vectors:
            self._vectors[text] = self.embed(text)
        return self._vectors[text]

    def _implementations(self):
        return {
            pid: p
            for pid, p in self.archive.items()
            if not p.verdict or p.verdict.get("implemented") is not False
        }

    def _link(self, calls, design, parent):
        """The matched earlier programs and the one to show, recorded whether or not it is shown."""
        link = {
            "design": design,
            "candidates": [],
            "similarities": [],
            "matched": [],
            "linked_id": None,
            "display_draw": None,
            "shown": False,
            "error": None,
        }
        if not design:
            return (link, None)
        try:
            ranked = rank_candidates(
                design,
                self._implementations(),
                parent.id,
                self._vector,
                self.config.match_candidates,
            )
        except Exception as exc:
            link["error"] = f"{type(exc).__name__}: {exc}"
            return (link, None)
        link["candidates"] = [program.id for program, _ in ranked]
        link["similarities"] = [round(similarity, 4) for _, similarity in ranked]
        if not ranked:
            return (link, None)
        prompt, tokens = self.prompts.match(design, ranked)
        reply = self._call(
            calls,
            prompt,
            tokens,
            "match",
            max_tokens=self.config.output_tokens,
            temperature=0.0,
            response_format=JSON_REPLY,
        )
        link["matched"] = parse_match(reply.get("content", ""), link["candidates"])
        if not link["matched"]:
            return (link, None)
        linked = choose_linked([self.archive[i] for i in link["matched"]])
        draw = random.Random(
            f"v12.1:{self.config.seed}:display:{self.progress.generations + 1}"
        ).random()
        link.update(
            linked_id=linked.id,
            display_draw=draw,
            shown=draw < self.config.display_probability,
        )
        return (link, linked if link["shown"] else None)

    def _search(self):
        if self.progress.repair_id is not None:
            self._repair()
            return
        opened = self._open_exploration()
        if self.progress.generations >= self.config.budget:
            if opened is not None:
                self._close_exploration(opened)
            self.progress.phase = "freeze"
            self._save()
            return
        if opened is not None:
            self._develop(opened)
            return
        self._ordinary_search()

    def _develop(self, exploration):
        """Complete the latest version while it is not a test of the Design and does not beat the start."""
        proposal, latest = (exploration["proposal"], self._latest(exploration))
        design = proposal.idea
        if latest is None or not design:
            self._close_exploration(exploration)
            return
        start = self.programs[proposal.parent_id]
        verdict = self._verdict(latest, design, start)
        if (
            self._tests(verdict)
            or better(latest.fitness, start.fitness)
            or len(exploration["development"]) >= self.config.completion_steps
            or (
                self._completion_draw(exploration["id"])
                >= self.config.completion_probability
            )
        ):
            self._close_exploration(exploration)
            return
        link = proposal.link or {}
        earlier = self.archive.get(link.get("linked_id")) if link.get("shown") else None
        base = (
            latest
            if verdict["kept"] is not False and self.config.completion == "read"
            else start
        )
        if self.config.completion == "resample":
            request = self.prompts._result([link["implement_prompt"]], "Complete")
        else:
            request = self.prompts.complete(
                base,
                link.get("proposal") or f"Design: {design}",
                verdict,
                earlier,
                version=None if base is latest else latest,
            )
        request.update(
            sampled_action="Complete",
            fallbacks=[],
            parent_id=base.id,
            reference_id=None,
            selection=None,
            reference_selection=None,
            design=design,
            completion=self.config.completion,
            exploration={
                "id": exploration["id"],
                "step": len(exploration["development"]) + 1,
            },
        )
        self._create_program(request, parent=base)

    def _close_exploration(self, exploration):
        self._save(exploration=self._exploration_record(exploration))

    def _exploration_record(self, exploration):
        proposal, proposed, best = (
            exploration["proposal"],
            exploration["proposed"],
            exploration["best"],
        )
        search_best = min(self.archive.values(), key=lambda p: (p.fitness, p.id))
        versions = []
        for program in [proposal, *exploration["development"]]:
            final = final_program(program, self.programs)
            if final.valid:
                verdict = final.verdict
                versions.append(
                    {
                        "id": final.id,
                        "implemented": verdict["implemented"] if verdict else None,
                        "kept": verdict["kept"] if verdict else None,
                        "tests": bool(verdict) and self._tests(verdict),
                    }
                )
        tests = [v["id"] for v in versions if v["tests"]]
        return {
            "id": exploration["id"],
            "start_id": proposal.parent_id,
            "proposal_id": proposal.id,
            "idea": proposal.idea,
            "first_status": final_program(proposal, self.programs).status,
            "proposed_id": proposed.id if proposed else None,
            "first_score": proposed.score if proposed else None,
            "developed": bool(exploration["development"]),
            "development_ids": [p.id for p in exploration["development"]],
            "best_id": best.id if best else None,
            "best_score": best.score if best else None,
            "search_best_id": search_best.id,
            "search_best_score": search_best.score,
            "closed_after": self.progress.generations,
            "completion_draw": self._completion_draw(exploration["id"]),
            "versions": versions,
            "test_id": tests[0] if tests else None,
            "test_improved": bool(tests)
            and better(
                self.programs[tests[0]].fitness,
                self.programs[proposal.parent_id].fitness,
            ),
        }

    def _latest(self, exploration):
        versions = [
            final_program(p, self.programs)
            for p in [exploration["proposal"], *exploration["development"]]
        ]
        return next((p for p in reversed(versions) if p.valid), None)

    def _verdict(self, program, design, start):
        """Whether ``program`` carries out ``design``, then whether it keeps the rest of ``start``:
        read once, saved, and reused on resume."""
        if program.verdict is not None:
            return program.verdict
        calls, record = (
            [],
            {
                "program_id": program.id,
                "start_id": start.id,
                "design": design,
                "implemented": None,
                "missing": None,
                "kept": None,
                "dropped": None,
            },
        )
        prompt, tokens = self.prompts.verify(design, program)
        reply = self._call(
            calls,
            prompt,
            tokens,
            "verify",
            max_tokens=self.config.output_tokens,
            temperature=0.0,
            response_format=JSON_REPLY,
        )
        record["implemented"], record["missing"] = parse_verdict(self._judged(reply))
        if record["implemented"] is not False:
            prompt, tokens = self.prompts.contain(design, program, start)
            reply = self._call(
                calls,
                prompt,
                tokens,
                "contain",
                max_tokens=self.config.output_tokens,
                temperature=0.0,
                response_format=JSON_REPLY,
            )
            record["kept"], record["dropped"] = parse_dropped(self._judged(reply))
        record["call_ids"] = [c["request_id"] for c in calls]
        program.verdict = record
        self._save(calls=calls, updated_program=program)
        return record

    @staticmethod
    def _judged(reply):
        return reply.get("content", "") if reply.get("finish_reason") == "stop" else ""

    @staticmethod
    def _tests(verdict):
        """A version tests its Design unless it was read as missing the Design or dropping part of the start."""
        return verdict["implemented"] is not False and verdict["kept"] is not False

    def _completion_draw(self, exploration_id):
        return random.Random(
            f"v12.1:{self.config.seed}:complete:{exploration_id}"
        ).random()

    def _generate_code(self, request):
        calls = []
        details = self._call(
            calls,
            request["prompt"],
            request["input_tokens"],
            max_tokens=self.config.output_tokens,
            response_format=JSON_REPLY,
        )
        return (calls[-1]["request_id"], details, calls)

    def _freeze_selection(self):
        self.progress.finalists = self._ranking()[
            : self.config.final_candidates if self.selection else 1
        ]
        if self.selection:
            self.progress.phase = "selection"
            self._save()
        elif self.progress.finalists:
            self._finish(self.archive[self.progress.finalists[0]], "training")
        else:
            self.progress.phase = "no_valid_root"
            self._save()

    def _evaluate_program(self, program, calls=(), *, new=False):
        try:
            validate_source(program.code, self.template)
        except (SyntaxError, ValueError) as exc:
            program.failure = {
                "kind": "invalid_source",
                "error": str(exc),
                "line": None,
                "seconds": None,
                "calls": None,
                "function_seconds": None,
                "call_running": False,
            }
        evaluations = []
        if program.failure is None:
            try:
                outcome = self.training.evaluate(
                    program.code, program.key, **self._evaluation_options()
                )
            except SchedulerError:
                self.progress.pending_id = program.id
                self._save(
                    program=program if new else None,
                    updated_program=None if new else program,
                    calls=calls,
                )
                raise
            evaluations = outcome["evaluations"]
            program.fitness, program.failure = (outcome["fitness"], outcome["failure"])
            program.eval_seconds = outcome.get("instance_seconds", outcome["seconds"])
            program.eval_wall_seconds = outcome["seconds"]
            program.calls, program.function_seconds = (
                outcome["calls"],
                outcome["function_seconds"],
            )
        program.status = program.failure["kind"] if program.failure else "valid"
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
        self._save(
            program=program if new else None,
            updated_program=None if new else program,
            calls=calls,
            evaluations=evaluations,
        )
        if self.progress.repair_id is not None:
            return self._repair()
        return program if program.valid else None

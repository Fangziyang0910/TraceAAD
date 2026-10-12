"""Historical V10.17 generation and search, with the current evaluation protocol."""

from benchmarks.tasks import FUNCTION_SECONDS, FUNCTION_LIMITS
from traceaad.common.delivery import DeliveryError, SourceError
from traceaad.common.prompts import ContextTooLong
from traceaad.common.selection import choose_reference
from traceaad.v10_17.prompts import PromptBuilder as HistoricalPrompts
from traceaad.v10_17.traceaad import TraceAADV1017
from traceaad.v10_21.config import Config
from traceaad.v10_21.traceaad import TraceAADV1021
from . import historical_delivery


class PromptBuilder(HistoricalPrompts):
    def evaluation_text(self, meaning, higher):
        return ("[Evaluation]\nEach program is evaluated on a fixed set of training instances.\n"
                f"Score: {meaning}. {'Higher' if higher else 'Lower'} is better.\n"
                f"All calls to the function together may use at most {FUNCTION_LIMITS.get(self.task, FUNCTION_SECONDS):g} "
                "seconds of CPU time per instance, averaged over all instances. "
                f"Each whole instance has a safety limit of {self.config.eval_timeout_seconds:g} seconds of wall time.")

    def error_text(self, failed):
        if failed["failure"]["kind"] == "runtime_error":
            return "\n".join((failed["failure"]["error"] or "").splitlines()[-15:])[-1500:]
        return super().error_text(failed)

    def failure(self, program):
        failure = program["failure"]
        if failure["kind"] == "timeout":
            # Instance limits cannot be described as an old whole-evaluation deadline.
            calls, inside = failure.get("calls"), failure.get("function_seconds") or 0.0
            text = failure["error"]
            if calls is not None:
                text += f"; after {self._calls(calls, inside)}"
                if failure.get("call_running"):
                    text += f", stopped inside call {calls + 1}"
            return text
        return super().failure(program)


class CurrentProtocolV1017(TraceAADV1017):
    Config = Config
    PromptBuilder = PromptBuilder
    # Only evaluation execution and final selection come from the current adapter.
    Evaluator = TraceAADV1021.Evaluator
    _evaluation_options = TraceAADV1021._evaluation_options
    _freeze = TraceAADV1021._freeze

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.selection is not None:
            raise ValueError("Current V10.17 experiments freeze on training, without a selection split")
        self.training.scheduler_socket = self.config.scheduler_socket
        self.identity["mechanism"] = "v1017-original-current-protocol-1"
        saved = self.facts.state or {}
        if saved and saved.get("identity") != self.identity:
            raise ValueError("V10.17 resume requires the same mechanism, configuration, task and protocol")

    def _parse_candidate(self, request, details, parent):
        try:
            return historical_delivery.parse_response(details.get("content", ""), details.get("finish_reason"), self.template)
        except historical_delivery.DeliveryError as exc:
            raise DeliveryError(str(exc)) from exc
        except historical_delivery.SourceError as exc:
            raise SourceError(str(exc), exc.code) from exc


    def _roots(self):
        if self.progress.repair_id is not None:
            return self._repair()
        roots = [node for node in self.archive.values() if self._is_root(node)]
        if (len(roots) >= self.config.roots or self.progress.init_attempts >= self.config.init_attempt_limit
                or self.attempts >= self.config.budget):
            self.progress.phase = "search" if roots else "no_valid_root"
            self._save()
            return
        self._attempt(self.prompts.initial(roots))

    def _ordinary_search(self, sampled=None):
        eligible = [node for node in self.archive.values() if node["id"] not in self.progress.too_long]
        if not eligible:
            self.progress.phase = "freeze"
            self._save()
            return
        parent, selection = self._choose_parent(eligible)
        if sampled is None:
            sampled = self.action_rng.choices(list(self.config.operators), list(self.config.operators.values()))[0]
        action = sampled
        reference, reference_selection = None, None
        flags = []
        if action == "Crossover":
            reference, reference_selection = choose_reference(parent, self.archive, self.reference_rng)
            if reference is None:
                action = "Refine"
                flags.append("crossover_fallback")
        try:
            request = self.prompts.build(action, parent, reference=reference)
        except ContextTooLong:
            self.progress.too_long.append(parent["id"])
            self._save()
            return
        if request["action"] == "Refine" and action == "Crossover":
            reference = None
            flags.append("crossover_context_fallback")
        if self.RECORD_EXPLORATIONS:
            request['exploration'] = ({'id': len(self.facts.explorations) + 1, 'step': 0}
                                      if request['action'] in self.EXPLORING else None)
        request.update(sampled_action=sampled, fallbacks=flags, parent_id=parent["id"],
                       reference_id=reference["id"] if reference else None, selection=selection,
                       reference_selection=reference_selection)
        self._attempt(request, parent=parent, reference=reference)

    def _develop(self, exploration):
        if (exploration["proposed"] is None or not self._developed(exploration["id"])
                or len(exploration["development"]) >= self.config.development_steps):
            self._close_exploration(exploration)
            return
        best = exploration["best"]
        try:
            request = self.prompts.build("Refine", best)
        except ContextTooLong:
            self.progress.too_long.append(best["id"])
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Refine", fallbacks=[], parent_id=best["id"], reference_id=None,
                       selection=None, reference_selection=None,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=best)

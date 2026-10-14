"""V12.1: V12.0 with design verification. Each Explore version is read as a test of its Design in the start
algorithm: it must carry out the Design and change nothing else. A version that is not a test is completed,
not judged on its score; one that does not carry out its Design is not inherited."""

import random

from traceaad.common.delivery import parse_response
from traceaad.common.history import final_attempt
from traceaad.common.prompts import ContextTooLong
from traceaad.common.selection import better
from traceaad.v12_0 import TraceAADV120
from .config import Config
from .prompts import PromptBuilder
from .verification import parse_dropped, parse_verdict


REPEATED = ("duplicate", "known_failure")


class TraceAADV121(TraceAADV120):
    METHOD = "v12_1"
    Config = Config
    PromptBuilder = PromptBuilder

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts.verdicts = self.facts.verdicts

    def _implementations(self):
        """A program whose code does not carry out its Design is not matched to its Design."""
        return {pid: p for pid, p in self.archive.items()
                if self.facts.verdicts.get(pid, {}).get("implemented") is not False}

    def _generate(self, request):
        if request["action"] == "Complete":
            calls = []
            reply = self._call(calls, request["prompt"], request["input_tokens"], "complete",
                               max_tokens=self.config.output_tokens)
            # The Design is the proposal's; the reply only writes code.
            return calls[-1]["request_id"], {**reply, "content": "Design: " + request["design"],
                                            "implementation": reply.get("content") or ""}, calls
        rid, details, calls = super()._generate(request)
        if "implementation" in details:
            request["link"]["proposal"] = details["content"]
            if self.config.completion == "resample":
                request["link"]["implement_prompt"] = calls[-1]["prompt"]
        return rid, details, calls

    def _parse_candidate(self, request, details, parent):
        if request["action"] != "Complete":
            return super()._parse_candidate(request, details, parent)
        code, _, delivery = parse_response(details["implementation"], details.get("finish_reason"), self.template)
        return code, request["design"], delivery

    def _latest(self, exploration):
        """The exploration's most recent valid version."""
        latest = None
        for attempt in [exploration["proposal"], *exploration["development"]]:
            final = final_attempt(attempt, self.attempts_table)
            if final["status"] == "valid":
                latest = self.archive[final["program_id"]]
        return latest

    def _verdict(self, program, design, start):
        """Whether ``program`` carries out ``design``, then whether it keeps the rest of ``start``:
        read once, saved, and reused on resume."""
        if program["id"] in self.facts.verdicts:
            return self.facts.verdicts[program["id"]]
        calls, record = [], {"program_id": program["id"], "start_id": start["id"], "design": design,
                             "implemented": None, "missing": None, "kept": None, "dropped": None, "error": None}
        prompt, tokens = self.prompts.verify(design, program)
        if tokens > self.config.max_input_tokens:
            record["error"] = "verify prompt exceeds context"
        else:
            reply = self._call(calls, prompt, tokens, "verify",
                               max_tokens=self.config.verify_output_tokens, temperature=0.0)
            record["implemented"], record["missing"] = parse_verdict(self._judged(reply))
        # What else changed matters only for a program that carries out its Design.
        if record["implemented"] is not False:
            prompt, tokens = self.prompts.contain(design, program, start)
            if tokens > self.config.max_input_tokens:
                record["error"] = "contain prompt exceeds context"
            else:
                reply = self._call(calls, prompt, tokens, "contain",
                                   max_tokens=self.config.verify_output_tokens, temperature=0.0)
                record["kept"], record["dropped"] = parse_dropped(self._judged(reply))
        record["call_ids"] = [c["request_id"] for c in calls]
        self._save(calls=calls, verdict=record)
        return record

    @staticmethod
    def _judged(reply):
        # A reply cut at the output limit ends before its judgment; a line quoted in its reasoning is not one.
        return reply.get("content", "") if reply.get("finish_reason") == "stop" else ""

    @staticmethod
    def _tests(verdict):
        """A version tests its Design unless it was read as missing the Design or dropping part of the start."""
        return verdict["implemented"] is not False and verdict["kept"] is not False

    def _completion_draw(self, exploration_id):
        # A fixed draw per exploration, so a resumed run draws the same.
        return random.Random(f"v12.1:{self.config.seed}:complete:{exploration_id}").random()

    def _develop(self, exploration):
        """Complete the latest version while it is not a test of the Design and does not beat the start."""
        proposal, latest = exploration["proposal"], self._latest(exploration)
        design = proposal.get("idea")
        if latest is None or not design:
            self._close_exploration(exploration)
            return
        start = self.programs[proposal["parent_id"]]
        verdict = self._verdict(latest, design, start)
        development = exploration["development"]
        # A completion that returned an existing program had nothing more to change.
        returned = bool(development) and final_attempt(development[-1], self.attempts_table)["status"] in REPEATED
        if (self._tests(verdict) or better(latest["fitness"], start["fitness"]) or returned
                or len(exploration["development"]) >= self.config.completion_steps
                or self._completion_draw(exploration["id"]) >= self.config.completion_probability):
            self._close_exploration(exploration)
            return
        link = proposal.get("link") or {}
        earlier = self.archive.get(link.get("linked_id")) if link.get("shown") else None
        # A version that kept the start is completed in place; otherwise the Design goes into the start itself.
        base = latest if verdict["kept"] is not False and self.config.completion == "read" else start
        try:
            if self.config.completion == "resample":
                # The first implementation request, sent again: an independent implementation from the start.
                request = self.prompts._result([link["implement_prompt"]], "Complete")
            else:
                request = self.prompts.complete(base, link.get("proposal") or f"Design: {design}", verdict, earlier,
                                                version=None if base is latest else latest)
        except ContextTooLong:
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Complete", fallbacks=[], parent_id=base["id"], reference_id=None,
                       selection=None, reference_selection=None, design=design, completion=self.config.completion,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=base)

    def _exploration_record(self, exploration):
        record = super()._exploration_record(exploration)
        versions = []
        for attempt in [exploration["proposal"], *exploration["development"]]:
            final = final_attempt(attempt, self.attempts_table)
            if final["status"] == "valid":
                verdict = self.facts.verdicts.get(final["program_id"])
                versions.append({"id": final["program_id"],
                                 "implemented": verdict["implemented"] if verdict else None,
                                 "kept": verdict["kept"] if verdict else None,
                                 "tests": bool(verdict) and self._tests(verdict)})
        tests = [v["id"] for v in versions if v["tests"]]
        record.update(developed=bool(exploration["development"]),
                      completion_draw=self._completion_draw(exploration["id"]), versions=versions,
                      # The first version read as a test of the Design in its start, and whether it beat the start.
                      test_id=tests[0] if tests else None,
                      test_improved=bool(tests) and better(self.programs[tests[0]]["fitness"],
                                                           self.programs[exploration["proposal"]["parent_id"]]["fitness"]))
        return record

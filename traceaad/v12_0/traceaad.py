"""V12.0: re-proposal linking.

V10.17 search under the current protocol (V10.21's adapter), with one change to
Explore. The search proposes the same computation many times, and each proposal
used to be implemented from scratch and judged by its own first version. Now an
Explore proposal is linked to the best earlier implementation of the computation
it names, so a repeated proposal continues the most developed implementation:

1. propose: V10.21's Explore prompt, stopped before the code (Analysis and Design);
2. match: the model names, among the nearest earlier Designs, those built on the
   same main computation (not a candidate);
3. implement: the same material plus the proposal and the best-scoring matched
   program; this reply is the candidate.

Parent selection, operator shares, Refine, Crossover, Repair, the 1/8 x 3
continuation and the budget are V10.17's.
"""

import random

from core.llm import ModelCallError, generate
from traceaad.common.delivery import extract_idea
from traceaad.v10_21.traceaad import TraceAADV1021
from .config import Config
from .linking import STOPS, choose_linked, cut_proposal, normal, parse_match, rank_candidates
from .prompts import PromptBuilder

MECHANISM = "v12.0-reproposal-linking-1"


class TraceAADV120(TraceAADV1021):
    METHOD = "v12_0"
    Config = Config
    PromptBuilder = PromptBuilder

    def __init__(self, *, embed=None, **kwargs):
        super().__init__(**kwargs)
        if self.config.linking and embed is None:
            raise ValueError("re-proposal linking requires an embedding function")
        self.embed, self._vectors = embed, {}
        self.identity["mechanism"] = MECHANISM
        if self.config.linking:
            self.identity["embedding"] = getattr(embed, "settings", None)
        saved = self.facts.state or {}
        if saved and saved.get("identity") != self.identity:
            raise ValueError("V12.0 resume requires the same configuration, task, protocol, mechanism and embedding")

    def _vector(self, text):
        if text not in self._vectors:
            self._vectors[text] = self.embed(text)
        return self._vectors[text]

    def _call(self, calls, prompt, input_tokens, stage, **kwargs):
        """One model request of a linked Explore, accounted as V10.17 accounts a generation."""
        try:
            details, failed = generate(self.llm, prompt, **kwargs), None
            made = details["calls"]
        except ModelCallError as exc:
            details, made, failed = {}, exc.calls, exc
        for call in made:
            self.progress.model_calls += 1
            call["request_id"], call["stage"] = self.progress.model_calls, stage
            usage = call["usage"]
            self.progress.input_tokens += usage.get("prompt_tokens", input_tokens) or input_tokens
            self.progress.output_tokens += usage.get("completion_tokens",
                kwargs["max_tokens"] if not call["error"] else 0) or 0
            self.progress.service_failures += bool(call["error"])
        calls.extend(made)
        if failed:
            self._save(calls=calls)
            if failed.transient:
                raise RuntimeError(f"model service unavailable after {len(calls)} calls: {failed}") from failed
            raise failed
        return details

    def _link(self, calls, design, parent):
        """The matched earlier programs and the one to show, recorded whether or not it is shown."""
        link = {"design": design, "candidates": [], "similarities": [], "matched": [],
                "linked_id": None, "display_draw": None, "shown": False, "error": None}
        if not design:
            return link, None
        try:
            ranked = rank_candidates(design, self.archive, parent["id"], self._vector, self.config.match_candidates)
        except Exception as exc:  # the embedding service is outside the search: this proposal stays unlinked
            link["error"] = f"{type(exc).__name__}: {exc}"
            return link, None
        link["candidates"] = [program["id"] for program, _ in ranked]
        link["similarities"] = [round(similarity, 4) for _, similarity in ranked]
        if not ranked:
            return link, None
        prompt, tokens = self.prompts.match(design, ranked)
        reply = self._call(calls, prompt, tokens, "match",
                           max_tokens=self.config.match_output_tokens, temperature=0.0)
        link["matched"] = parse_match(reply.get("content", ""), link["candidates"])
        if not link["matched"]:
            return link, None
        linked = choose_linked([self.archive[i] for i in link["matched"]])
        # A fixed draw per candidate id, so a resumed run draws the same.
        draw = random.Random(f"v12.0:{self.config.seed}:display:{self.attempts + 1}").random()
        link.update(linked_id=linked["id"], display_draw=draw, shown=draw < self.config.display_probability)
        return link, linked if link["shown"] else None

    def _generate(self, request):
        if not (self.config.linking and request["action"] == "Explore"):
            return super()._generate(request)
        parent, calls = self.programs[request["parent_id"]], []
        first = self._call(calls, request["prompt"], request["input_tokens"], "propose",
                           max_tokens=self.config.output_tokens, stop=list(STOPS))
        proposal = cut_proposal(first.get("content", ""))
        link, earlier = self._link(calls, normal(extract_idea(proposal)), parent)
        implement = self.prompts.implement(request["prompt"], parent, proposal, earlier)
        if "earlier_program" in implement["trims"]:
            link["shown"] = False
        link["trims"] = implement["trims"]
        request["link"] = link
        second = self._call(calls, implement["prompt"], implement["input_tokens"], "implement",
                            max_tokens=self.config.output_tokens)
        # The Design comes from the proposal, the code from the implementation.
        content = proposal + "\n\n" + (second.get("content") or "")
        return calls[-1]["request_id"], {**second, "content": content}, calls

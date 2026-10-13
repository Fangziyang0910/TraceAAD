"""V12.0: propose, match an earlier computation, then implement with its best program."""

import random

from traceaad.common.delivery import extract_idea, parse_response
from traceaad.common.prompts import ContextTooLong
from traceaad.common.search import Search
from traceaad.common.history import final_attempt
from traceaad.common.instance_evaluation import InstanceProgramEvaluator
from .config import Config
from .linking import STOPS, choose_linked, cut_proposal, normal, parse_match, rank_candidates
from .prompts import PromptBuilder


class TraceAADV120(Search):
    RECORD_EXPLORATIONS = True
    EXPLORING = ("Explore",)
    Evaluator = InstanceProgramEvaluator
    METHOD = "v12_0"
    Config = Config
    PromptBuilder = PromptBuilder

    def __init__(self, *, embed=None, **kwargs):
        super().__init__(**kwargs)
        self.training.scheduler_socket = self.config.scheduler_socket
        if self.selection:
            self.selection.scheduler_socket = self.config.scheduler_socket
        if self.config.linking and embed is None:
            raise ValueError("re-proposal linking requires an embedding function")
        self.embed, self._vectors = embed, {}
        if self.config.linking:
            self.identity["embedding"] = getattr(embed, "settings", None)
        saved = self.facts.state or {}
        saved_identity = {k: v for k, v in saved.get("identity", {}).items() if k != "mechanism"}
        if saved and saved_identity != self.identity:
            raise ValueError("V12.0 resume requires the same configuration, task, protocol and embedding")

    def _vector(self, text):
        if text not in self._vectors:
            self._vectors[text] = self.embed(text)
        return self._vectors[text]

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
        if tokens > self.config.max_input_tokens:
            link["error"] = "match prompt exceeds context"
            return link, None
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
        try:
            implement = self.prompts.implement(request["prompt"], parent, proposal, earlier)
        except ContextTooLong as exc:
            # No implementation was requested: keep its costs, but do not charge a candidate.
            calls[-1]["context_error"] = str(exc)
            self.progress.too_long.append(parent["id"])
            self._save(calls=calls)
            raise
        if "earlier_program" in implement["trims"]:
            link["shown"] = False
        link["trims"] = implement["trims"]
        request["link"] = link
        second = self._call(calls, implement["prompt"], implement["input_tokens"], "implement",
                            max_tokens=self.config.output_tokens)
        # The Design comes from the proposal, the code from the implementation.
        return calls[-1]["request_id"], {**second, "content": proposal,
                                        "implementation": second.get("content") or ""}, calls

    def _parse_candidate(self, request, details, parent):
        if "implementation" not in details:
            return super()._parse_candidate(request, details, parent)
        code, _, delivery = parse_response(details["implementation"], details.get("finish_reason"), self.template)
        return code, request["link"]["design"], delivery

    def _ordinary_search(self, sampled=None):
        try:
            return super()._ordinary_search(sampled)
        except ContextTooLong:
            # The parent was marked unavailable and the completed calls were saved above.
            return

    def _search(self):
        if self.progress.repair_id is not None:
            self._repair()
            return
        opened = self._open_exploration()
        if self.attempts >= self.config.budget:
            if opened is not None:
                self._close_exploration(opened)
            self.progress.phase = "freeze"
            self._save()
            return
        if opened is not None:
            self._develop(opened)
            return
        self._ordinary_search()

    def _developed(self, exploration_id):
        """Whether an exploration's new program is developed: a fixed draw per exploration,
        independent of every other random choice, so a resumed run draws the same."""
        # Keep the original seed namespace so existing V12.0 runs resume identically.
        draw = random.Random(f"v10.17:{self.config.seed}:develop:{exploration_id}").random()
        return draw < self.config.development_probability

    def _develop(self, exploration):
        if (exploration["proposed"] is None or not self._developed(exploration["id"])
                or len(exploration["development"]) >= self.config.development_steps):
            self._close_exploration(exploration)
            return
        best = exploration["best"]
        try:
            request = self.prompts.build("Refine", best)
        except ContextTooLong:
            self._close_exploration(exploration)
            return
        request.update(sampled_action="Refine", fallbacks=[], parent_id=best["id"], reference_id=None,
                       selection=None, reference_selection=None,
                       exploration={"id": exploration["id"], "step": len(exploration["development"]) + 1})
        self._attempt(request, parent=best)

    def _close_exploration(self, exploration):
        proposal, proposed, best = exploration["proposal"], exploration["proposed"], exploration["best"]
        search_best = min(self.archive.values(), key=lambda n: (n["fitness"], n["id"]))
        record = {
            "id": exploration["id"], "start_id": proposal["parent_id"], "proposal_attempt": proposal["id"],
            "idea": proposal["idea"], "first_status": final_attempt(proposal, self.attempts_table)["status"],
            "proposed_id": proposed["id"] if proposed else None,
            "developed": proposed is not None and self._developed(exploration["id"]),
            "first_score": proposed["score"] if proposed else None,
            "development_attempts": [a["id"] for a in exploration["development"]],
            "best_id": best["id"] if best else None, "best_score": best["score"] if best else None,
            "search_best_id": search_best["id"], "search_best_score": search_best["score"],
            "closed_after": self.attempts}
        self._save(exploration=record)

    def _evaluation_options(self):
        return {"timeout_seconds": self.config.eval_timeout_seconds,
                "n_workers": self.config.eval_workers}

    def _freeze(self):
        """Recheck the training top programs and freeze the best valid result."""
        if self.selection:
            return super()._freeze()
        ranking = self._ranking()
        results = []
        for node_id in ranking[:self.config.final_candidates]:
            node = self.archive[node_id]
            outcome = self.training.evaluate(node["code"], node["key"], **self._evaluation_options())
            results.append({"node_id": node_id, **{k: v for k, v in outcome.items() if k != "evaluations"}})
            self._save(evaluations=outcome["evaluations"])
        self.progress.finalists = [r["node_id"] for r in results]
        self.progress.selection_results = results
        scored = [r for r in results if r["fitness"] is not None]
        if not scored:
            self.progress.phase = "selection_failed" if ranking else "no_valid_root"
            self._save()
            return
        chosen = min(scored, key=lambda r: (r["fitness"], ranking.index(r["node_id"])))["node_id"]
        self._finish(self.archive[chosen], "training")

"""V14: the TraceAAD search loop with three counters to fitness-guided conservatism.

Revisit rejection: a reply whose code is identical to an evaluated program is not
evaluated; the same request is asked once more with that fact stated.
Divergence: after ``stagnation_window`` attempts without a new best, one attempt
writes a new algorithm from a map of the designs developed so far, without code.
Development period: the new design then receives ``mature_steps`` Refine-style
attempts before it competes through ordinary parent selection.
"""

from copy import deepcopy

from traceaad.delivery import DeliveryError, optional_field, parse_json_response
from traceaad.program import Program
from traceaad.selection import better
from traceaad.traceaad import TraceAAD
from . import novelty
from .config import Config
from .prompts import PromptBuilderV14


class TraceAADV14(TraceAAD):
    METHOD = "traceaad_v14"

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        self.prompts = PromptBuilderV14(
            self.llm, kwargs.get("task"), kwargs["evaluation"], self.programs, self.config
        )
        saved = self.facts.state or {}
        self.v14 = deepcopy(
            saved.get("v14") or {"episode": None, "episodes": [], "last_episode_end": 0}
        )
        self._shingles = {}
        self._identities = {}
        for program in sorted(self.programs.values(), key=lambda p: p.id):
            if program.status != "revisit":
                self._remember(program)

    def _save(self, **record):
        self.facts.commit({**self.progress.snapshot(), "v14": self.v14}, **record)

    def _remember(self, program):
        key = novelty.identity(program.code)
        if key is not None:
            self._identities.setdefault(key, program.id)

    def _best(self):
        return min(self.archive.values(), key=lambda p: (p.fitness, p.id))

    # Generation and revisit rejection

    def _create_program(self, request, *, parent=None, reference=None, repair_of=None):
        if self.v14["episode"] is not None:
            request.setdefault("episode", self.v14["episode"]["index"])
        details = self._generate(request)
        calls = details["calls"]
        self.progress.generations += 1
        if repair_of is not None:
            self.progress.repair_id = None
        if self.phase == "roots":
            self.progress.init_attempts += 1
        repairing = request["action"] == "Repair"
        try:
            code, idea, analysis = parse_json_response(
                details.get("content", ""), details.get("finish_reason"), repair=repairing
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
            idea=parent.idea if repairing else idea,
            change=parent.change if repairing else optional_field(details["content"], "change"),
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

    def _evaluate_program(self, program, calls=(), *, new=False):
        if new and self.config.revisit:
            seen = self._identities.get(novelty.identity(program.code))
            if seen is not None:
                return self._reject(program, seen, calls)
            self._remember(program)
        return super()._evaluate_program(program, calls, new=new)

    def _reject(self, program, seen, calls):
        program.status = "revisit"
        program.failure = {"kind": "revisit", "of": seen}
        if self.phase == "roots":
            room = self.progress.init_attempts < self.config.init_attempt_limit
        else:
            room = self.progress.generations < self.config.budget
        self.progress.repair_id = program.id if program.repair_of is None and room else None
        self.progress.pending_id = None
        self._save(program=program, new=True, calls=calls)
        if self.progress.repair_id is not None:
            return self._repair()
        return None

    def _repair(self):
        failed = self.programs[self.progress.repair_id]
        if failed.status != "revisit":
            return super()._repair()
        parent = self.programs.get(failed.parent_id)
        reference = self.programs.get(failed.reference_id)
        seen = failed.failure["of"]
        if seen == failed.parent_id:
            what = "the current algorithm"
        elif seen == failed.reference_id:
            what = "the reference algorithm"
        else:
            what = "an algorithm already evaluated in this search"
        request = self.prompts.revisit(self._rebuild(failed, parent, reference), what)
        request.update(
            {k: v for k, v in failed.request.items() if k not in request},
            revisit_of=seen,
            revised=failed.id,
        )
        return self._create_program(request, parent=parent, reference=reference, repair_of=failed.id)

    def _rebuild(self, failed, parent, reference):
        """The request that produced ``failed``, built from the current records."""
        action = failed.action
        if action == "Init":
            return self.prompts.initial([n for n in self.archive.values() if self._is_root(n)])
        if action == "Diverge":
            return self._diverge_request()
        if action == "Mature":
            return self.prompts.mature(parent, self._best().fitness)
        return self.prompts.build(action, parent, reference=reference)

    # Divergence and development period

    def _search(self, sampled=None):
        if self.budget_used >= self.config.budget or not self.archive:
            return super()._search(sampled)
        if self.v14["episode"] is not None:
            return self._develop()
        if self.config.diverge and self._stalled():
            self.v14["episode"] = {
                "index": len(self.v14["episodes"]) + 1,
                "start": self.progress.generations,
                "best_before": self._best().fitness,
                "requests": 0,
            }
            return self._develop()
        return super()._search(sampled)

    def _stalled(self):
        since = max(self._best().generation, self.v14["last_episode_end"])
        return self.progress.generations - since >= self.config.stagnation_window

    def _members(self, index):
        return [p for p in self.programs.values() if p.request.get("episode") == index]

    def _develop(self):
        episode = self.v14["episode"]
        members = self._members(episode["index"])
        valid = [p for p in members if p.valid]
        if not valid:
            # A second new design when the first reply failed or was never saved.
            if episode["requests"] >= 2:
                return self._close(members)
            episode["requests"] += 1
            self._save()
            return self._create_program(self._diverge_request())
        done = sum(p.action == "Mature" and p.repair_of is None for p in members)
        if done >= self.config.mature_steps:
            return self._close(members)
        node = min(valid, key=lambda p: (p.fitness, p.id))
        request = self.prompts.mature(node, self._best().fitness)
        request.update(parent_id=node.id, development_step=done + 1)
        return self._create_program(request, parent=node)

    def _close(self, members):
        episode = self.v14["episode"]
        valid = [p for p in members if p.valid]
        best = min(valid, key=lambda p: (p.fitness, p.id)) if valid else None
        episode.update(
            end=self.progress.generations,
            members=[p.id for p in members],
            best_id=best.id if best else None,
            best_fitness=best.fitness if best else None,
            new_best=bool(best and better(best.fitness, episode["best_before"])),
        )
        self.v14["episodes"].append(episode)
        self.v14["episode"] = None
        self.v14["last_episode_end"] = self.progress.generations
        self._save()

    def _shingle(self, program):
        if program.id not in self._shingles:
            self._shingles[program.id] = novelty.shingles(program.code)
        return self._shingles[program.id]

    def _tried(self):
        """Recent new designs: each first Diverge reply with its best score after development."""
        tried = []
        for episode in self.v14["episodes"][-self.config.diverge_shown :] if self.config.diverge_shown else []:
            first = next(
                (
                    self.programs[i]
                    for i in episode["members"]
                    if self.programs[i].action == "Diverge" and self.programs[i].repair_of is None
                ),
                None,
            )
            if first is not None:
                tried.append((first, episode["best_fitness"]))
        return tried

    def _diverge_request(self):
        tried = self._tried()
        excluded = {p.id for p, _ in tried}
        designs = []
        for program in sorted(self.archive.values(), key=lambda p: (p.fitness, p.id)):
            if program.id in excluded:
                continue
            shingles = self._shingle(program)
            if all(
                novelty.distance(shingles, self._shingle(q)) >= self.config.map_distance
                for q in designs
            ):
                designs.append(program)
                if len(designs) == self.config.map_size:
                    break
        return self.prompts.diverge(designs, tried, self._best().fitness)

    def _summary(self, status, error=None):
        save = self.facts.save_summary

        def with_mechanism(summary):
            programs = self.programs.values()
            summary["v14"] = {
                "revisits": sum(p.status == "revisit" for p in programs),
                "episodes": len(self.v14["episodes"]) + (self.v14["episode"] is not None),
                "episodes_with_new_best": sum(bool(e.get("new_best")) for e in self.v14["episodes"]),
                "diverge_attempts": sum(p.action == "Diverge" for p in programs),
                "development_attempts": sum(p.action == "Mature" for p in programs),
            }
            save(summary)

        self.facts.save_summary = with_mechanism
        try:
            return super()._summary(status, error)
        finally:
            self.facts.save_summary = save

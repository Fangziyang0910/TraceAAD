"""V20 search: Base with trajectory scoring, trajectory context and experience learning.

Selection records live in ``attempt.record`` and are replayed into the selector
on resume; ``experience.json`` holds the experience state after each attempt.
"""

from traceaad_base.search import TraceAAD as Base

from .config import Config
from .context import Context, OPERATORS
from .experience import Experience, REFLECTION_SCHEMA
from .prompts import PromptBuilder
from .representation import identity
from .selection import Selector


class TraceAAD(Base):
    method = "traceaad_v20"
    prompt_type = PromptBuilder

    def __init__(self, task, model, *, config: Config | None = None, run_dir=None, resume=False,
                 embedder=None, reflection_model=None):
        config = config or Config()
        self.selector = Selector()
        self.experience = Experience(task, embedder, config)
        self.context = Context(task, config, self.experience)
        self.experience.formation = lambda archive, parent: self.context.ancestors(archive, parent)[0]
        self.reflection_model = reflection_model or (
            model.for_role(REFLECTION_SCHEMA) if hasattr(model, "for_role") else model)
        super().__init__(task, model, config=config, run_dir=run_dir, resume=resume)
        if resume:
            self.experience.data = self.store.read("experience.json", self.experience.data)

    def restore(self, attempt):
        if "selection" in attempt.record:
            self.selector.observe(attempt.record["selection"], attempt.parent_id,
                                  self.archive.get(attempt.program_id))

    def select(self, attempt_id):
        config = self.config
        if not config.trajectory_scoring:
            return super().select(attempt_id)
        distinct = len({identity(p.code) for p in self.archive.values() if p.valid}) > 1
        parent, operator, reference, decision, candidates = self.selector.choose(
            self.archive, config, self.rng,
            distinct_crossover=distinct if config.reference_selection else True)
        self.store.append("decisions.jsonl", {"attempt_id": attempt_id, "candidates": candidates})
        return parent, operator, reference, {"selection": decision}

    def evolution_prompt(self, attempt, parent, reference):
        config = self.config
        best_before = self.best.evaluation.fitness
        if not (config.trajectory_context or config.reference_selection or config.experience_learning):
            attempt.record["context"] = {"best_before": best_before}
            return super().evolution_prompt(attempt, parent, reference)
        info = {}
        if attempt.operator == "Crossover" and config.reference_selection:
            reference, info = self.context.reference(self.archive, parent, self.rng)
            if reference is None:
                attempt.operator = "Explore"
        attempt.reference_id = reference.id if reference else None
        notes, retrieval = [], {}
        if config.experience_learning:
            events = self.experience.data["events"].values()
            recent = [e for e in events if e["parent_id"] == parent.id][-2:]
            need = OPERATORS[attempt.operator].split(". ", 1)[0] + ".\n" + "\n".join(
                self.experience.card(e["event_id"]) for e in recent)
            if reference:
                need += "\nReference design: " + (reference.design or "")
            notes, retrieval = self.experience.retrieve(parent, need)
        sections, required, view = self.context.render(self.archive, parent, attempt.operator,
                                                        reference, notes)
        attempt.record["context"] = {"reference": info, "experiences": notes, "retrieval": retrieval,
                                     "view": view, "best_before": best_before}
        history = self.context.assemble(sections, view)
        return self.prompts._render(history + ("\n\n" if history else "") + "\n\n".join(required),
                                    self.context.instruction(attempt.operator, sections),
                                    has_scores=True)

    def _reflect(self, attempt, prompt, role):
        record = {"stage": role}
        content = self._request(self.reflection_model, prompt, record)
        attempt.record.setdefault("aux", []).append({"role": role, **record})
        self.store.append("calls.jsonl", {**record, "attempt_id": attempt.id,
                                          "prompt": prompt, "response": content})
        return content

    def finish(self, attempt):
        if "selection" in attempt.record:
            self.selector.observe(attempt.record["selection"], attempt.parent_id,
                                  self.archive.get(attempt.program_id))
        if not self.config.experience_learning:
            return
        if self.stop_reason is None:
            self.experience.reflect(attempt, self.archive,
                                    lambda prompt, role: self._reflect(attempt, prompt, role))
        else:
            # A reflection after the last evaluation cannot change the search.
            self.experience.observe(attempt, self.archive)
        attempt.record.setdefault("aux", []).append(self.experience.drain())
        self.store.write("experience.json", self.experience.data)

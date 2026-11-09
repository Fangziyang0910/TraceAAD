"""V24 search: V21 plus design notes revised every 10 evaluated attempts and shown to Refine and Crossover."""

from traceaad_v21.context import context
from traceaad_v21.search import TraceAAD as V21

from .notes import INTERVAL, REFLECTION_FORMAT, attempt_card, notes_block, parse, reflection_prompt


class TraceAAD(V21):
    method = "traceaad_v24"

    def __init__(self, task, model, *, reflection_model=None, **options):
        # Base replays saved attempts through restore() during its own __init__.
        self.notes, self.notes_from, self.pending = [], None, []
        super().__init__(task, model, **options)
        self.reflection_model = reflection_model or (
            model.for_role(REFLECTION_FORMAT) if hasattr(model, "for_role") else model)

    def evolution_prompt(self, attempt, parent, reference):
        attempt.record["best_before"] = self.best.evaluation.fitness
        text, instruction, view = context(self.archive, parent, attempt.operator, reference, self.config)
        shown = attempt.operator != "Explore" and bool(self.notes)
        attempt.record["context"] = view
        attempt.record["notes_from"] = self.notes_from if shown else None
        if shown:
            text = notes_block(self.notes) + "\n\n" + text
        return self.prompts._render(text, instruction, has_scores=attempt.operator != "Explore")

    def _counts(self, attempt):
        return attempt.operator != "Init" and bool(attempt.evaluations) and attempt.program_id is not None

    def finish(self, attempt):
        super().finish(attempt)
        if not self._counts(attempt):
            return
        self.pending.append(attempt.id)
        if len(self.pending) < INTERVAL:
            return
        if self.stop_reason is None:  # A reflection after the last evaluation cannot change the search.
            self._reflect(attempt)
        self.pending = []

    def _reflect(self, attempt):
        by_id = {a.id: a for a in self.attempts}
        cards = [attempt_card(self.archive, self.archive[by_id[i].program_id], by_id[i].record["best_before"])
                 for i in self.pending]
        prompt = reflection_prompt(self.task, self.notes, cards)
        record = {"stage": "reflect"}
        content = self._request(self.reflection_model, prompt, record)
        attempt.record.setdefault("aux", []).append({"role": "reflect", **record})
        self.store.append("calls.jsonl", {**record, "attempt_id": attempt.id, "prompt": prompt, "response": content})
        parsed = parse(content) if content is not None else None
        reflection = {"attempts": list(self.pending), "status": "success" if parsed else "failed",
                      "previous": list(self.notes)}
        if parsed:
            reflection["analysis"], reflection["notes"] = parsed
            self.notes, self.notes_from = parsed[1], attempt.id
            self.store.write("notes.json", {"from_attempt": attempt.id, "notes": self.notes,
                                            "analysis": parsed[0]})
        attempt.record["reflection"] = reflection

    def restore(self, attempt):
        super().restore(attempt)
        if not self._counts(attempt):
            return
        self.pending.append(attempt.id)
        if len(self.pending) < INTERVAL:
            return
        reflection = attempt.record.get("reflection")
        if reflection and reflection["status"] == "success":
            self.notes, self.notes_from = reflection["notes"], attempt.id
        self.pending = []

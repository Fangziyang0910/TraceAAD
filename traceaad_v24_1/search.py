"""V24.1 lifecycle: independent reflection cadence, unread evidence, and committed experience snapshots."""

from copy import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path

from traceaad_base import Config
from traceaad_v21.search import TraceAAD as V21

from .context import context
from .evidence import build, observe
from .notes import (CAPACITY, CHAT_RESERVE, CONTEXT_TOKENS, INTERVAL, REPRESENTATIVES,
                    Experience, REFLECTION_FORMAT, REFLECTION_TOKENS, reflection_prompt)


class ReflectionRequest:
    """Use Base's request logging with a separate reflection output allowance."""

    def __init__(self, model, max_tokens):
        self.model, self.max_tokens = model, max_tokens

    def request_context(self):
        return getattr(self.model, "request_context", nullcontext)()

    @property
    def last_metadata(self):
        return getattr(self.model, "last_metadata", {})

    def complete(self, prompt, **settings):
        return self.model.complete(prompt, **{**settings, "max_tokens": self.max_tokens})


class TraceAAD(V21):
    method = "traceaad_v24_1"

    def __init__(self, task, model, *, config=None, reflection_model=None,
                 reflection_context_tokens=CONTEXT_TOKENS, reflection_max_tokens=REFLECTION_TOKENS,
                 reflection_counter=None, **options):
        config = config or Config()
        if config.initial_size < 1:
            raise ValueError("V24.1 needs at least one initial program")
        if reflection_model is None:
            reflection_model = model.for_role(REFLECTION_FORMAT) if hasattr(model, "for_role") else model
        if reflection_model is model and hasattr(model, "request_options"):
            reflection_model = copy(model)
        if hasattr(reflection_model, "request_options"):
            reflection_model.request_options = {**reflection_model.request_options,
                                                "response_format": REFLECTION_FORMAT}
        self.reflection_model = reflection_model
        self.reflection_max_tokens = reflection_max_tokens
        self.input_limit = reflection_context_tokens - reflection_max_tokens - CHAT_RESERVE
        if self.input_limit < 1:
            raise ValueError("Reflection context must leave room for input, output and chat framing")
        counter = reflection_counter or getattr(self.reflection_model, "count_prompt_tokens", None)
        if not callable(counter):
            raise ValueError("V24.1 requires count_prompt_tokens or an explicit reflection_counter")
        self.measure = counter
        self.reflection_settings = {
            "interval": INTERVAL, "capacity": CAPACITY, "representatives": REPRESENTATIVES,
            "context_tokens": reflection_context_tokens, "output_reserve": reflection_max_tokens,
            "chat_reserve": CHAT_RESERVE, "input_limit": self.input_limit,
            "input_units": "tokens",
        }
        self.experience = Experience()
        self.experience_from = None
        self.events = {}
        self.interval_count = 0
        self.unreviewed_events = []
        if options.get("resume"):
            self._validate_resume(task, config, options.get("run_dir"))
        super().__init__(task, model, config=config, **options)
        if not options.get("resume"):
            saved = self.store.read("run_config.json", {})
            self.store.write("run_config.json", {**saved, "reflection_settings": self.reflection_settings})
        # attempts.jsonl is authoritative; discard a mirror written before an interrupted commit.
        self.store.write("experience.json", self.snapshot())

    def _validate_resume(self, task, config, directory):
        if directory is None or not (Path(directory) / "run_config.json").is_file():
            raise ValueError("V24.1 resume requires its original run_config.json")
        saved = json.loads((Path(directory) / "run_config.json").read_text())
        expected = {"method": self.method, "config": asdict(config),
                    "task": json.loads(json.dumps(task.description())),
                    "reflection_settings": self.reflection_settings}
        differences = [name for name, value in expected.items() if saved.get(name) != value]
        if differences:
            raise ValueError("Incompatible V24.1 resume settings: " + ", ".join(differences))

    def snapshot(self):
        return {"experience": self.experience.snapshot(), "experience_from": self.experience_from,
                "interval_count": self.interval_count, "unreviewed_events": list(self.unreviewed_events)}

    def evolution_prompt(self, attempt, parent, reference):
        attempt.record["best_before"] = self.best.evaluation.fitness
        notes = self.experience.active if attempt.operator != "Explore" else []
        text, instruction, view = context(self.archive, parent, attempt.operator, reference, self.config, notes)
        attempt.record["context"] = view
        attempt.record["experience_from"] = self.experience_from if notes else None
        attempt.record["experience_ids"] = [n["id"] for n in notes]
        return self.prompts._render(text, instruction, has_scores=attempt.operator != "Explore")

    def _observe(self, attempt):
        event = observe(attempt, self.archive)
        if event is None:
            return False
        self.events[event.id] = event
        if attempt.operator == "Init":
            return False
        self.interval_count += 1
        self.unreviewed_events.append(event.id)
        return True

    def finish(self, attempt):
        super().finish(attempt)
        if not self._observe(attempt):
            return
        if self.interval_count == INTERVAL:
            self.interval_count = 0
            if self.stop_reason is None:
                self._reflect(attempt)
                attempt.record["reflection"]["snapshot"] = self.snapshot()
        self.store.write("experience.json", self.snapshot())

    def _reflect(self, attempt):
        reflection = {"status": "deferred", "unreviewed_before": list(self.unreviewed_events),
                      "shown_event_ids": [], "reviewed_event_ids": []}
        attempt.record["reflection"] = reflection
        try:
            notes = self.experience.active
            material = build(self.events, self.unreviewed_events, notes,
                             lambda evidence: reflection_prompt(self.task, notes, evidence),
                             self.measure, self.input_limit)
        except Exception as exc:
            reflection.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        reflection.update(input_size=material.input_size, input_units=self.reflection_settings["input_units"],
                          input_limit=self.input_limit, deferred_event_ids=list(material.deferred_ids))
        if material.reason is not None:
            reflection["reason"] = material.reason
            return
        reflection["shown_event_ids"] = list(material.event_ids)
        record = {"stage": "reflect", "max_tokens": self.reflection_max_tokens}
        content = self._request(ReflectionRequest(self.reflection_model, self.reflection_max_tokens),
                                material.prompt, record)
        attempt.record.setdefault("aux", []).append({"role": "reflect", **record})
        self.store.append("calls.jsonl", {**record, "attempt_id": attempt.id,
                                          "prompt": material.prompt, "response": content})
        if content is None:
            reflection.update(status="failed", error=record.get("error", "No reflection response"))
            return
        try:
            update = self.experience.apply(content, set(material.event_ids))
        except (TypeError, ValueError) as exc:
            reflection.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            return
        read = set(material.event_ids)
        reviewed = [i for i in self.unreviewed_events if i in read]
        self.unreviewed_events = [i for i in self.unreviewed_events if i not in read]
        self.experience_from = attempt.id
        reflection.update(status="success", reviewed_event_ids=reviewed, **update)

    def restore(self, attempt):
        super().restore(attempt)
        if not self._observe(attempt):
            return
        if self.interval_count == INTERVAL:
            self.interval_count = 0
        reflection = attempt.record.get("reflection")
        if reflection is not None:
            state = reflection["snapshot"]
            self.experience = Experience(state["experience"])
            self.experience_from = state["experience_from"]
            self.interval_count = state["interval_count"]
            self.unreviewed_events = list(state["unreviewed_events"])

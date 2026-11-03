"""Role-specific evolution views: code only for the program under change; parent choice and reflections see text only."""

from dataclasses import dataclass, field
import re

from .templates import ANALYSIS_REQUIREMENTS, TEMPLATES


@dataclass
class Prompt:
    text: str
    metadata: dict = field(default_factory=dict)
    provided: list[dict] = field(default_factory=list)


def render_experiences(entries):
    """Global hints: one line each, with the conditions under which they are relevant."""
    return "\n".join("- " + e["content"]["summary"] + " Applies when: " + e["content"]["applicable_when"]
                      for e in entries)


def formation(archive, attempts, pid, limit):
    ids, seen = [], set()
    while pid in archive and archive[pid].attempt_id is not None and len(ids) < limit:
        program = archive[pid]
        if pid in seen:
            raise ValueError("Cyclic program ancestry")
        seen.add(pid)
        attempt = attempts[program.attempt_id - 1]
        if attempt.operator == "Init":
            break
        ids.append(attempt.id)
        pid = attempt.parent_id
    return list(reversed(ids))


class PromptBuilder:
    def __init__(self, task, config, count):
        self.task, self.config, self.count = task, config, count

    def output(self, mode):
        return TEMPLATES["Generator output"].format(analysis_requirement=ANALYSIS_REQUIREMENTS[mode])

    def complete(self, text, role="generator", metadata=None, provided=()):
        return Prompt(text, {**(metadata or {}), "prompt_tokens": self.count(text)}, list(provided))

    def version(self, program, title=None):
        evaluation = program.evaluation
        outcome = f"fitness={evaluation.fitness}" if program.valid else (
            "timed out; fitness=unavailable" if evaluation and evaluation.status == "timeout" else
            "program error; fitness=unavailable" if evaluation else "not evaluated")
        label = f"[{title} P{program.id}]" if title else f"P{program.id}:"
        return (f"{label}\n{outcome}\n{program.design or ''}\n"
                f"```python\n{program.code}\n```")

    def query(self, archive, attempts, parent):
        ids = formation(archive, attempts, parent.id, self.config.history_steps)
        descriptions = [parent.design or ""]
        for tid in ids:
            descriptions.extend(archive[pid].design or "" for pid in attempts[tid - 1].version_ids)
        return "\n".join(dict.fromkeys(" ".join(d.split()) for d in descriptions if d.strip()))

    @staticmethod
    def evaluated(archive, attempt):
        return [archive[pid] for pid in attempt.version_ids if archive[pid].evaluation is not None]

    def facts(self, archive, attempt):
        parent = archive.get(attempt.parent_id)
        baseline = parent.evaluation.fitness if parent and parent.valid else None
        details = []
        rows = [f"Attempt {attempt.id}. Selected parent: P{attempt.parent_id}; fitness={baseline}.",
                "| Version | Phase | Edited from | Execution | Fitness | Gain vs selected parent |",
                "| --- | --- | --- | --- | --- | --- |"]
        for version in self.evaluated(archive, attempt):
            evaluation = version.evaluation
            if version.valid:
                gain = baseline - evaluation.fitness if baseline is not None else None
                change = f"{gain:+.6g}" if gain is not None else "unavailable"
                status, fitness = "valid", str(evaluation.fitness)
            else:
                status = "timed out" if evaluation.status == "timeout" else "program error"
                fitness, change = "unavailable", "unavailable"
            rows.append(f"| P{version.id} | {version.phase} | P{version.edited_from} | {status} | {fitness} | {change} |")
            details.append(f"P{version.id} analysis: {version.analysis or ''}")
            details.append(f"P{version.id} design: {version.design or ''}")
            if evaluation.status == "error":
                kinds = re.findall(r"\b([A-Za-z_][A-Za-z_0-9.]*(?:Error|Exception))\b", evaluation.failure or "")
                details.append(f"P{version.id} failure type: " + (kinds[-1] if kinds else "Program error"))
        if not self.evaluated(archive, attempt):
            rows.append("No candidate was evaluated.")
        if attempt.reflection:
            details.append(f"Attempt {attempt.id} reflection: {attempt.reflection}")
        return "\n".join([*rows, *details])

    def _fields(self):
        return {"task_description": self.task.task_description, "function_template": self.task.function_template,
                "score_definition": self.task.score_definition}

    def _group(self, archive, attempts, ids):
        return "\n\n".join(self.facts(archive, attempts[tid - 1]) for tid in ids)

    @staticmethod
    def _omit_empty(template, fields):
        for name, value in fields.items():
            if not value:
                template = re.sub(r"\[[^\n]+\]\n\{" + re.escape(name) + r"\}\n*", "", template)
        return template.format(**fields)

    def initialization(self, archive):
        blocks, shown = [], []
        for program in sorted(archive.values(), key=lambda p: p.id):
            if program.id == 0:
                blocks.append(self.version(program, "Initial Example"))
            elif program.operator == "Init" and program.evaluation is not None:
                outcome = str(program.evaluation.fitness) if program.valid else program.evaluation.status
                blocks.append(f"Initial design P{program.id}: {program.design or ''}\nOutcome: {outcome}")
            else:
                continue
            shown.append(program.id)
        return self.complete(TEMPLATES["Initialization"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, score_definition=self.task.score_definition,
            initial_context_block="\n\n".join(blocks), output_contract=self.output("Initialization")),
            metadata={"initial_context_ids": shown})

    def generation(self, archive, attempts, parent, retrieved):
        """Only the program under change appears as code; history is design, analysis, outcome and reflection."""
        main = formation(archive, attempts, parent.id, self.config.history_steps) if self.config.generation_history else []
        sides = [a.id for a in attempts if a.completed and a.parent_id == parent.id
                 and self.evaluated(archive, a)][-self.config.alternative_count:] if (
                     self.config.generation_history and self.config.alternative_count) else []
        fields = {**self._fields(), "formation_trajectory": self._group(archive, attempts, main),
                  "previous_attempts": self._group(archive, attempts, sides),
                  "positive_experience": render_experiences([e for e in retrieved if e["polarity"] == "positive"]),
                  "negative_experience": render_experiences([e for e in retrieved if e["polarity"] == "negative"]),
                  "current_algorithm": self.version(parent), "output_contract": self.output("Generator")}
        metadata = {"main_attempt_ids": main, "alternative_attempt_ids": sides,
                    "provided_experience_ids": [e["id"] for e in retrieved], "visible_code_version_ids": [parent.id]}
        return self.complete(self._omit_empty(TEMPLATES["Generator"], fields), metadata=metadata, provided=retrieved)

    def selection(self, archive, attempts, ids):
        """Candidate starting points side by side: how each was formed and what was tried from it; text only."""
        blocks = []
        for pid in ids:
            program = archive[pid]
            formed = formation(archive, attempts, pid, self.config.history_steps)
            tried = [a.id for a in attempts if a.completed and a.parent_id == pid and self.evaluated(archive, a)]
            recent = tried[-self.config.alternative_count:] if self.config.alternative_count else []
            blocks.append("\n".join([
                f"[Program P{pid}]\nfitness={program.evaluation.fitness}\nDesign: {program.design or ''}",
                "How it was formed:\n" + (self._group(archive, attempts, formed) if formed else "An initial design."),
                f"Attempts made from it so far: {len(tried)}."
                + ("\n" + self._group(archive, attempts, recent) if recent else "")]))
        fields = {**self._fields(), "candidate_programs": "\n\n".join(blocks)}
        return self.complete(TEMPLATES["Selector"].format(**fields), "selector", metadata={"candidate_ids": list(ids)})

    def reflection(self, archive, attempts, attempt):
        """Short-term reflection on the latest attempt in its local trajectory; text only."""
        mode = self.config.reflection_mode
        main = formation(archive, attempts, attempt.parent_id, max(0, self.config.history_steps - 1)) if mode == "trajectory" else []
        sides = [a.id for a in attempts if a.completed and a.id < attempt.id
                 and a.parent_id == attempt.parent_id and self.evaluated(archive, a)][-self.config.alternative_count:] if (
                     mode != "local" and self.config.alternative_count) else []
        parent = archive[attempt.parent_id]
        fields = {**self._fields(), "preceding_evolution": self._group(archive, attempts, main),
                  "other_attempts": self._group(archive, attempts, sides),
                  "latest_attempt": f"Selected parent P{parent.id} design: {parent.design or ''}\n\n"
                                    + self.facts(archive, attempt)}
        metadata = {"main_attempt_ids": [*main, attempt.id], "alternative_attempt_ids": sides}
        return self.complete(self._omit_empty(TEMPLATES["Reflector"], fields), "reflector", metadata=metadata)

    def long_term(self, archive, attempts, reflected):
        """Long-term reflection forms the global pools from recent short-term reflections and their outcomes."""
        blocks = []
        for attempt in reflected:
            parent = archive[attempt.parent_id]
            first = min(attempt.version_ids)
            before = min((p.evaluation.fitness for p in archive.values() if p.valid and p.id < first), default=None)
            outcome = next(v for v in reversed(self.evaluated(archive, attempt)))
            result = (f"fitness={outcome.evaluation.fitness}" if outcome.valid else
                      "timed out" if outcome.evaluation.status == "timeout" else "program error")
            blocks.append(f"Attempt {attempt.id}: selected parent fitness={parent.evaluation.fitness}; "
                          f"result {result}; best fitness before this attempt={before}.\n"
                          f"Design: {outcome.design or ''}\nReflection: {attempt.reflection}")
        fields = {**self._fields(), "recent_reflections": "\n\n".join(blocks)}
        metadata = {"reflection_attempt_ids": [a.id for a in reflected]}
        return self.complete(self._omit_empty(TEMPLATES["Long-term reflector"], fields), "long_term_reflector",
                             metadata=metadata)

    def repair_format(self, original, failure):
        return self.complete(TEMPLATES["Format repair"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, original_response=original,
            parsing_feedback=failure, output_contract=self.output("Format repair")))

    def repair_program(self, program):
        instruction = TEMPLATES["Timeout instruction" if program.evaluation.status == "timeout" else "Error instruction"]
        return self.complete(TEMPLATES["Program repair"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, failed_design_block=program.design or "",
            repair_feedback=program.evaluation.failure, failed_version_block=f"```python\n{program.code}\n```",
            repair_instruction=instruction, output_contract=self.output("Program repair")))

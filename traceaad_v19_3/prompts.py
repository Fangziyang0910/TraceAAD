"""Complete code comparisons and bounded context for generation and reflection."""

from dataclasses import dataclass, field
import difflib
import json

from .schemas import response_format
from .templates import ANALYSIS_REQUIREMENTS, TEMPLATES


class ContextOverflow(ValueError):
    pass


@dataclass
class Prompt:
    text: str
    metadata: dict = field(default_factory=dict)
    provided: list[dict] = field(default_factory=list)


def render_experiences(entries):
    blocks = []
    for polarity in ("positive", "negative"):
        notes = [e for e in entries if e["polarity"] == polarity]
        if not notes:
            continue
        lines = [f'[{polarity.title()} Design Experience]']
        for entry in notes:
            content = entry["content"]
            lines.extend(["Summary: " + content["summary"],
                          "Advice: " + " ".join(content["recommendations"]),
                          "Applies when: " + content["applicable_when"], ""])
        blocks.append("\n".join(lines).strip())
    return "\n\n".join(blocks)


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

    def short(self, text, limit=192):
        text = text or ""
        if self.count(text) <= limit:
            return text
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if self.count(text[:mid] + " …") <= limit:
                low = mid
            else:
                high = mid - 1
        return text[:low] + " …"

    def output(self, mode):
        return TEMPLATES["Generator output"].format(analysis_requirement=ANALYSIS_REQUIREMENTS[mode])

    def complete(self, text, role="generator", metadata=None, provided=()):
        prompt_tokens = self.count(text)
        schema_tokens = self.count(json.dumps(response_format(role), ensure_ascii=False))
        reserve = self.config.reflection_max_tokens if role == "reflector" else self.config.max_tokens
        total = prompt_tokens + schema_tokens + self.config.protocol_tokens + reserve
        if total > self.config.context_window:
            raise ContextOverflow(f"Full input and output reservation need {total} tokens; window={self.config.context_window}")
        return Prompt(text, {**(metadata or {}), "prompt_tokens": prompt_tokens,
                             "schema_tokens": schema_tokens, "output_reservation": reserve}, list(provided))

    def version(self, program, title):
        evaluation = program.evaluation
        outcome = f"fitness={evaluation.fitness}" if program.valid else (
            "timed out; fitness=unavailable" if evaluation and evaluation.status == "timeout" else
            "program error; fitness=unavailable" if evaluation else "not evaluated")
        return (f"[{title} P{program.id}]\n{outcome}\n{self.short(program.design)}\n"
                f"```python\n{program.code}\n```")

    def query(self, archive, attempts, parent):
        ids = formation(archive, attempts, parent.id, self.config.history_steps)
        descriptions = [parent.design or ""]
        for tid in ids:
            descriptions.extend(archive[pid].design or "" for pid in attempts[tid - 1].version_ids)
        return "\n".join(self.short(d, 256) for d in dict.fromkeys(descriptions))

    def facts(self, archive, attempt):
        parent = archive.get(attempt.parent_id)
        baseline = parent.evaluation.fitness if parent and parent.valid else None
        rows = [f"Attempt {attempt.id}. Selected parent: P{attempt.parent_id}; fitness={baseline}.",
                "| Version | Phase | Edited from | Execution | Fitness | Gain vs selected parent |",
                "| --- | --- | --- | --- | --- | --- |"]
        for pid in attempt.version_ids:
            version = archive[pid]
            evaluation = version.evaluation
            if evaluation and evaluation.status == "success":
                gain = baseline - evaluation.fitness if baseline is not None else None
                relation = "better" if gain is not None and gain > 0 else "worse" if gain is not None and gain < 0 else "unchanged"
                change = f"{gain:+.6g} ({relation})" if gain is not None else "unavailable"
                status, fitness = "valid", str(evaluation.fitness)
            else:
                status = "timed out" if evaluation and evaluation.status == "timeout" else "program error" if evaluation else "evaluation interrupted"
                fitness, change = "unavailable", "unavailable"
            rows.append(f"| P{pid} | {version.phase} | P{version.edited_from} | {status} | {fitness} | {change} |")
        for pid in attempt.version_ids:
            version = archive[pid]
            rows.append(f"Change P{version.edited_from} -> P{pid}: {self.short(version.analysis, 128)}")
            if version.evaluation and version.evaluation.status == "error":
                lines = (version.evaluation.failure or "").strip().splitlines()
                rows.append("Failure: " + self.short(lines[-1] if lines else "Program error", 96))
        if not attempt.version_ids:
            rows.append("No candidate was evaluated.")
        return "\n".join(rows)

    def edits(self, archive, attempt, known):
        known = set(known)
        blocks = []
        for pid in reversed(attempt.version_ids):
            after = archive[pid]
            before = archive.get(after.edited_from)
            if before is None:
                continue
            header = f"Code comparison: P{before.id} -> P{after.id}\n"
            diff = "\n".join(difflib.unified_diff(before.code.splitlines(), after.code.splitlines(),
                fromfile=f"P{before.id}", tofile=f"P{after.id}", lineterm=""))
            pair = "\n".join(f"P{v.id}: code provided elsewhere in this context." if v.id in known
                             else f"P{v.id}:\n```python\n{v.code}\n```" for v in (before, after))
            if not diff:
                diff_view = "No source change."
            else:
                diff_view = f"```diff\n{diff}\n```"
            if before.id not in known and after.id not in known:
                anchor = min((before, after), key=lambda v: self.count(v.code))
                diff_view = f"P{anchor.id}:\n```python\n{anchor.code}\n```\n" + diff_view
            body = min((pair, diff_view), key=self.count)
            blocks.append(header + body)
            known.update((before.id, after.id))
        return "\n\n".join(reversed(blocks)), known

    def pack(self, archive, attempts, main, sides, current, notes, *, historical_advice=False):
        ids = list(dict.fromkeys([*main, *sides]))
        facts = {tid: self.facts(archive, attempts[tid - 1]) for tid in ids}
        edits, shown, selected, old_advice = {}, {current.id}, [], []

        def blocks():
            def group(title, members):
                if not members:
                    return ""
                return title + "\n\n" + "\n\n".join(facts[t] + "\n" + edits.get(
                    t, "Evaluation only; code change unavailable.") for t in members)
            return (group("[Evaluated Evolution]", main), group("[Alternative Variants]", sides),
                    render_experiences(selected), "\n\n".join(old_advice))

        def size():
            return self.count("\n\n".join(blocks()))

        if size() > self.config.material_tokens:
            raise ContextOverflow("Evaluation summaries exceed the history/experience material budget")

        def add_edits(tid):
            nonlocal shown
            proposed, known = self.edits(archive, attempts[tid - 1], shown)
            edits[tid] = proposed
            if size() <= self.config.material_tokens:
                shown = known
            else:
                del edits[tid]

        if main:
            add_edits(main[-1])
        for note in sorted(notes, key=lambda e: e.get("retrieval_score", 0), reverse=True):
            selected.append(note)
            if size() > self.config.material_tokens:
                selected.pop()
        for tid in reversed(main[:-1]):
            add_edits(tid)
        for tid in reversed(sides):
            add_edits(tid)
        if historical_advice:
            seen = {e["id"] for e in selected}
            for tid in reversed(ids):
                prior = [e for e in attempts[tid - 1].provided_experiences if e["id"] not in seen]
                if not prior:
                    continue
                old_advice.append(f"[Earlier advice provided in attempt {tid}]\n" + render_experiences(prior))
                if size() > self.config.material_tokens:
                    old_advice.pop()
                else:
                    seen.update(e["id"] for e in prior)
        main_block, side_block, note_block, old_block = blocks()
        metadata = {"main_attempt_ids": main, "alternative_attempt_ids": sides,
                    "complete_change_attempt_ids": sorted(edits),
                    "outcome_only_attempt_ids": [i for i in ids if i not in edits],
                    "provided_experience_ids": [e["id"] for e in selected],
                    "material_tokens": size(), "visible_code_version_ids": sorted(shown)}
        return main_block, side_block, note_block, old_block, metadata, selected

    def initialization(self, archive):
        blocks = []
        for program in sorted(archive.values(), key=lambda p: p.id):
            if program.id == 0 or (program.valid and program.operator == "Init" and program.duplicate_of is None):
                block = self.version(program, "Initial Design")
                if self.count("\n\n".join([*blocks, block])) <= self.config.material_tokens:
                    blocks.append(block)
        text = TEMPLATES["Initialization"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, score_definition=self.task.score_definition,
            initial_context_block="\n\n".join(blocks), output_contract=self.output("Initialization"))
        return self.complete(text, metadata={"initial_context_ids": [p.id for p in archive.values()
            if self.version(p, "Initial Design") in blocks]})

    def generation(self, archive, attempts, parent, retrieved):
        main = formation(archive, attempts, parent.id, self.config.history_steps)
        sides = [a.id for a in attempts if a.completed and a.parent_id == parent.id and a.evaluations][-self.config.alternative_count:] if self.config.alternative_count else []
        args = self.pack(archive, attempts, main, sides, parent, retrieved)
        main_block, side, notes, _, metadata, shown = args
        text = TEMPLATES["Generator"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, score_definition=self.task.score_definition,
            evolution_block=main_block, alternative_variants_block=side,
            positive_experience_block=render_experiences([e for e in shown if e["polarity"] == "positive"]),
            negative_experience_block=render_experiences([e for e in shown if e["polarity"] == "negative"]),
            current_algorithm_block=self.version(parent, "Current Algorithm"), output_contract=self.output("Generator"))
        return self.complete(text, metadata=metadata, provided=shown)

    def reflection(self, archive, attempts, attempt, current):
        mode = self.config.reflection_mode
        previous = formation(archive, attempts, attempt.parent_id, max(0, self.config.history_steps - 1)) if mode == "trajectory" else []
        main = [*previous, attempt.id]
        sides = [a.id for a in attempts if a.completed and a.id != attempt.id and a.parent_id == attempt.parent_id and a.evaluations][-self.config.alternative_count:] if mode != "local" and self.config.alternative_count else []
        main_block, side, notes, old, metadata, shown = self.pack(
            archive, attempts, main, sides, current, attempt.provided_experiences, historical_advice=True)
        text = TEMPLATES["Reflector"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, score_definition=self.task.score_definition,
            provided_experience_block=notes + ("\n\n" + old if old else ""),
            evaluated_trajectory_block=main_block, alternative_variants_block=side,
            evaluated_candidate_block=self.version(current, "Latest Evaluated Algorithm"),
            output_contract=TEMPLATES["Reflector output"])
        return self.complete(text, "reflector", metadata=metadata, provided=shown)

    def repair_format(self, original, failure):
        return self.complete(TEMPLATES["Format repair"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, original_response=original,
            parsing_feedback=failure, output_contract=self.output("Format repair")))

    def repair_program(self, program):
        instruction = TEMPLATES["Timeout instruction" if program.evaluation.status == "timeout" else "Error instruction"]
        return self.complete(TEMPLATES["Program repair"].format(task_description=self.task.task_description,
            function_template=self.task.function_template, failed_design_block=self.short(program.design),
            repair_feedback=program.evaluation.failure,
            failed_version_block=f"```python\n{program.code}\n```",
            repair_instruction=instruction, output_contract=self.output("Program repair")))

"""Prompts for TraceAAD."""

import re

from .config import DEFAULT_OPERATOR_WEIGHTS
from .history import code_diff, path, score_text, verdict

SCORE_MEANINGS = {
    "tsp_construct": "the average length of the constructed tours",
    "cvrp_aco": "the average total length of the best routes found by the ant colony",
    "op_aco": "the negative average total prize of the best tours found by the ant colony",
    "online_bin_packing": "the average number of bins used",
    "fssp_gls": "the average makespan",
    "graph_colouring": "the average number of colors used",
    "jssp_construct": "the average makespan",
    "vrptw_construct": "the average total travel distance of the constructed routes",
}
INITIAL = "[Your Task: Design an Algorithm]\nWrite an algorithm that achieves the best score you can on this task."
ANOTHER_INITIAL = "[Your Task: Design a Better Algorithm]\nUse the algorithms and scores above to design another algorithm that aims for better solutions. Base your approach on the computations in their code."
REFINE = "[Your Task: Refine]\nImprove the current algorithm within its existing decision process. Change how its computations are performed or used, and retain useful parts that the change does not replace."
EXPLORE = "[Your Task: Explore]\nWrite a version of the current algorithm that uses a different computation to decide its output, aiming for better solutions. Retain existing computations that remain useful."
CROSSOVER = "[Your Task: Crossover]\nCombine useful computations from the current and reference algorithms to improve solution quality. Choose parts based on what their code computes and how they can work together."
REPAIR = "[Your Task: Repair]\nFix the reported failure. Preserve the algorithm's decision logic except where it causes the failure."
REPAIR_TIMEOUT = "[Your Task: Repair a Timeout]\nThe program did not finish within the time limit. Change it so that it finishes within the limit, keeping the main decisions of its algorithm."
RECORDS = "Scores and failures are evaluation results. Design is the description written when a program was generated; use the code to understand its computations. A repaired program inherits the failed program's Design."
FORMATION_INTRO = "The steps that produced the current algorithm, oldest first. Each step gives its evaluated outcome, Design, and code diff from the previous version. If a step required repair, the diff ends at the repaired program and the first failure is stated separately."
IDEA_DISPLAY_CHARS = 2400
ERROR_DISPLAY_CHARS = 240


def idea_view(idea):
    return " ".join((idea or "").split())[:IDEA_DISPLAY_CHARS]


ANALYSIS_INSTRUCTIONS = {
    "Refine": "which computation you will change, how you will change it, and why this might improve solution quality",
    "Crossover": "what each algorithm computes, which parts you will combine, and how those parts will work together",
    "Repair": "what caused the reported failure, and what change fixes it",
    "Timeout": "which computations you will reduce or replace to meet the reported limit, and how you will preserve the algorithm's main decisions",
    "Explore": "what computation you will replace or add, and how it will change the algorithm's decisions",
}
INIT_ANALYSIS = "what determines the score on this task, and what kind of algorithm should do well"
BETTER_ANALYSIS = "what computations the algorithms above use, and what approach you will implement to seek better solutions"
DESIGN = "describe the main computations in the returned code and, when modifying an algorithm, the main change, in at most 200 words"


def json_format(analysis, *, design=True):
    fields = [f'"analysis": explain {analysis}.']
    if design:
        fields.append(f'"design": {DESIGN}.')
    fields.append(
        '"code": the complete Python program. Write no comments or docstrings in the code.'
    )
    count = "three string fields" if design else "two string fields"
    return f"[Output Format]\nReply with one JSON object with {count}:\n" + "\n".join(fields)


def short_error(text):
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return (lines[-1] if lines else "unknown error")[:ERROR_DISPLAY_CHARS]


CALLED_ONCE = "The function is called once per instance, before the ant colony starts."
MANY_TIMES = re.compile(
    "Use efficient NumPy operations because the function is evaluated many\\s+times\\."
)
SERVICE = "At a customer, the vehicle waits until the time window opens if it arrives early, then spends the customer's service duration before moving on. The current time passed to the function already includes the waiting and service durations of the customers served so far; the service durations themselves are not passed to the function."
OP_OBJECTIVE = "\nThe evaluation objective is the negative mean collected prize; lower is better."


def corrected_description(task, description):
    """The task text with the facts about computation corrected."""
    if task in ("cvrp_aco", "op_aco"):
        description, replaced = MANY_TIMES.subn(CALLED_ONCE, description)
        if replaced != 1 and CALLED_ONCE not in description:
            raise ValueError(
                f"{task}: the task text no longer has the expected computation sentence"
            )
    if task == "vrptw_construct":
        description += "\n\n" + SERVICE
    return description


class PromptBuilder:
    # Instructions that a subclass may replace together with the hooks
    # _what, _change, _current, _reference, _root and _format, which decide how
    # programs other than the edited one appear. tried_note ends the header of
    # the programs already generated from the edited one.
    records = RECORDS
    formation_intro = FORMATION_INTRO
    crossover = CROSSOVER
    another_initial = ANOTHER_INITIAL
    tried_note = ""

    def __init__(self, llm, task, evaluation, programs, config):
        # The task text every method sees; TraceAAD states the score in its own line.
        description = evaluation.task_description.replace(OP_OBJECTIVE, "")
        self.llm = llm
        self.programs, self.config = (programs, config)
        self.score_meaning = SCORE_MEANINGS.get(
            task, getattr(evaluation, "score_meaning", "the task objective")
        )
        description = corrected_description(task, description.strip())
        self.common = [
            "[Task]\n" + description,
            f"Score: {self.score_meaning}. Lower is better.",
            "[Target Function]\n```python\n"
            + str(evaluation.template_program).strip()
            + "\n```\nKeep the function name, arguments and return contract exactly as shown. The program must be self-contained: include every import, constant and helper it uses.",
        ]

    def _result(self, sections, action, edge_ids=()):
        prompt = "\n\n".join(section for section in sections if section)
        return {
            "prompt": prompt,
            "input_tokens": self.llm.count_prompt_tokens(prompt),
            "action": action,
            "history_edge_ids": list(edge_ids),
            "reference_history_edge_ids": [],
            "child_ids": [],
            "progress_ids": [],
            "token_count_mode": getattr(self.llm, "prompt_token_count_mode", "serving_tokenizer"),
        }

    def failure(self, program):
        if program.failure["kind"] == "timeout":
            return "timed out"
        failure = program.failure
        kind = failure["kind"]
        if kind == "invalid_source":
            return "the program could not be used: " + short_error(failure.get("error"))
        where = f" at `{failure['line']}`" if failure.get("line") else ""
        if kind == "invalid_output":
            return "invalid output: " + short_error(failure.get("error")) + where
        return "runtime error: " + short_error(failure.get("error")) + where

    @staticmethod
    def steps(sequence):
        """The valid versions on a formation path, each with the failed first version
        its step produced before the repair (None when the step did not fail)."""
        steps, failed = ([], None)
        for program in sequence:
            if program.valid:
                steps.append((program, failed))
                failed = None
            else:
                failed = program
        return steps

    def _edge(self, parent, child, failed, index, *, latest=False, subject="current"):
        event = failed or child
        action = event.action
        if action == "Crossover" and event.reference_id is not None:
            action += f" with an algorithm scoring {score_text(self.programs[event.reference_id].fitness)}"
        if failed is not None:
            action += ", then Repair"
        heading = f"Step {index}"
        if latest:
            heading += f" (latest: produced the {subject} algorithm)"
        end = f"score {score_text(child.fitness)} ({verdict(parent.fitness, child.fitness)})"
        result = f"{heading} · {action} · score {score_text(parent.fitness)} → {end}"
        if failed is not None:
            result += f"\n  First version failed: {self.failure(failed)}"
        result += f"\n  {self._what(child)}"
        result += self._change(parent, child)
        return result

    def _what(self, program, missing=""):
        """How a listed step, attempt or improvement describes the program it produced."""
        return f"Design: {idea_view(program.idea) or missing}"

    def _change(self, parent, child):
        """The change a formation step made, as shown after its Design."""
        return f"\n  Code diff (previous → current):\n```diff\n{code_diff(parent.code, child.code)}\n```"

    def _formation(
        self, sequence, count, *, title="How the Current Algorithm Was Formed", subject="current"
    ):
        steps = self.steps(sequence)
        root, root_failed = steps[0]
        failed_note = (
            f" (its first version failed: {self.failure(root_failed)})" if root_failed else ""
        )
        if len(steps) == 1:
            return (
                f"[{title}]\nThe {subject} algorithm is an initial design{failed_note}; no changes have been recorded yet.\nDesign: {idea_view(root.idea)}",
                [],
            )
        start = max(1, len(steps) - count)
        lines = [f"[{title}]", self.formation_intro.replace("current algorithm", f"{subject} algorithm")]
        if start == 1:
            lines.append(
                f"Start · initial algorithm · score {score_text(root.fitness)}{failed_note}\n  Design: {idea_view(root.idea)}"
            )
        else:
            lines.append(
                f"The path has {len(steps) - 1} steps; showing the most recent {len(steps) - start} steps."
            )
        for index in range(start, len(steps)):
            lines.append(
                self._edge(
                    steps[index - 1][0],
                    steps[index][0],
                    steps[index][1],
                    index,
                    latest=index == len(steps) - 1,
                    subject=subject,
                )
            )
        return (
            lines[0] + "\n" + lines[1] + "\n\n" + "\n\n".join(lines[2:]),
            [p.id for p, _ in steps[start:]],
        )

    def children_from(self, node):
        return [
            p
            for p in sorted(self.programs.values(), key=lambda p: p.id)
            if p.parent_id == node.id
            and p.repair_of is None
            and (p.action in DEFAULT_OPERATOR_WEIGHTS)
        ]

    def _outcome(self, program, start):
        if program.valid:
            return (
                f"score {score_text(program.fitness)} ({verdict(start.fitness, program.fitness)})"
            )
        return "failed: " + self.failure(program)

    def _children_section(self, node):
        attempts = self.children_from(node)
        shown = min(self.config.attempts_shown, len(attempts))
        if not attempts or shown == 0:
            return ("", [])
        listed = attempts[-shown:]
        repairs = {p.repair_of: p for p in self.programs.values() if p.repair_of is not None}
        improved = sum(self._improved(repairs.get(p.id, p), node) for p in attempts)
        header = f"[Programs Generated From the Current Algorithm]\n{len(attempts)} program{('' if len(attempts) == 1 else 's')} generated from the current algorithm; {improved} scored better than it, after repair where needed. "
        header += (
            "All are listed, oldest first, with their evaluated outcomes."
            if len(listed) == len(attempts)
            else f"The most recent {len(listed)} are listed, oldest first, with their evaluated outcomes."
        )
        if self.tried_note:
            header += " " + self.tried_note
        lines = []
        for program in listed:
            action = program.action
            if action == "Crossover" and program.reference_id in self.programs:
                action += f" with an algorithm scoring {score_text(self.programs[program.reference_id].fitness)}"
            outcome = self._outcome(program, node)
            if repair := repairs.get(program.id):
                outcome += "; repaired: " + self._outcome(repair, node)
            lines.append(f"Program {program.id} · {action} · {self._what(program, '(none)')}\n  → {outcome}")
        return header + "\n\n" + "\n\n".join(lines), [p.id for p in listed]

    def _improved(self, program, node):
        return program.valid and verdict(node.fitness, program.fitness) == "improved"

    def improvements(self):
        """Valid programs that set a new search best, after the first, oldest first."""
        best, result = (None, [])
        for program in sorted((p for p in self.programs.values() if p.valid), key=lambda p: p.id):
            if best is None or program.fitness < best.fitness:
                if best is not None:
                    result.append((best, program))
                best = program
        return (best, result)

    def _progress_section(self, shown):
        best, steps = self.improvements()
        listed = steps[-shown:] if shown else []
        text = "[How the Best Score Improved in This Search]\n"
        if listed:
            text += (
                "Programs that set a new best score, oldest first"
                + (
                    "."
                    if len(listed) == len(steps)
                    else f"; the most recent {len(listed)} of {len(steps)} are listed."
                )
                + "\n\n"
            )
            entries = []
            for previous, program in listed:
                action = program.action
                source = self.programs.get(program.parent_id)
                if source is not None and (not source.valid):
                    action = source.action + ", then Repair"
                    source = self.programs.get(source.parent_id)
                origin = (
                    "as an initial algorithm"
                    if source is None
                    else f"from an algorithm scoring {score_text(source.fitness)}"
                )
                entries.append(
                    f"Program {program.id} · {action} {origin} → {score_text(program.fitness)} (previous best {score_text(previous.fitness)})\n  {self._what(program)}"
                )
            text += "\n\n".join(entries) + "\n\n"
        text += f"Best score found so far in this search: {score_text(best.fitness)}."
        return (text, [program.id for _, program in listed])

    def initial(self, roots):
        sections = list(self.common)
        if roots:
            sections.append(self.records)
            numbered = {node.id: index for index, node in enumerate(roots, 1)}
            entries = [
                self._root(node, numbered[node.id])
                for node in sorted(roots, key=lambda n: (n.fitness, n.id))
            ]
            header = f"[Algorithms Designed So Far]\nEvaluated starting algorithms for this task, listed best first. Score is {self.score_meaning}; lower is better."
            sections.append(header + "\n\n" + "\n\n".join(entries))
        sections.extend(
            [self.another_initial, self._format(BETTER_ANALYSIS)]
            if roots
            else [INITIAL, self._format(INIT_ANALYSIS)]
        )
        return self._result(sections, "Init")

    def _root(self, node, number):
        """An earlier initial algorithm as shown to the next initialization."""
        return f"Algorithm {number} · {self.measured(node)} · Design: {idea_view(node.idea)}\n```python\n{node.code.rstrip()}\n```"

    def build(self, action, parent, *, reference=None):
        if action not in DEFAULT_OPERATOR_WEIGHTS:
            raise ValueError(action)
        sequence = path(parent, self.programs)
        current = self._current(parent)
        tried, tried_ids = self._children_section(parent)
        depth = len(self.steps(sequence)) - 1
        if action == "Refine":
            history, ids = self._formation(sequence, min(self.config.history_depth, depth))
            sections = self.common + [
                self.records,
                history,
                tried,
                current,
                REFINE,
                self._format(ANALYSIS_INSTRUCTIONS["Refine"], edit=True),
            ]
            result = self._result(sections, "Refine", ids)
        elif action == "Explore":
            best, progress_ids = self._progress_section(self.config.progress_shown)
            sections = self.common + [
                self.records,
                tried,
                best,
                current,
                EXPLORE,
                self._format(ANALYSIS_INSTRUCTIONS["Explore"], edit=True),
            ]
            result = self._result(sections, "Explore")
            result["progress_ids"] = progress_ids
        else:
            if reference is None:
                raise ValueError("Crossover requires a reference")
            ref_section = self._reference(reference)
            reference_sequence = path(reference, self.programs)
            recent, ids = self._formation(sequence, min(4, depth))
            ref_history, ref_ids = self._formation(
                reference_sequence,
                min(4, len(self.steps(reference_sequence)) - 1),
                title="How the Reference Algorithm Was Formed",
                subject="reference",
            )
            sections = self.common + [
                self.records,
                recent,
                ref_history,
                tried,
                current,
                ref_section,
                self.crossover,
                self._format(ANALYSIS_INSTRUCTIONS["Crossover"], edit=True),
            ]
            result = self._result(sections, "Crossover", ids)
            result["reference_history_edge_ids"] = ref_ids
        result["child_ids"] = tried_ids
        return result

    def _reference(self, reference):
        """The reference algorithm of a Crossover."""
        return f"[Reference Algorithm]\nAnother evaluated algorithm from this search.\n{self.measured(reference)}\nDesign: {idea_view(reference.idea)}\n```python\n{reference.code.rstrip()}\n```"

    def _format(self, analysis, *, design=True, edit=False):
        """The reply format of a request; ``edit`` marks a request that modifies an existing algorithm."""
        return json_format(analysis, design=design)

    def error_text(self, failed):
        failure = failed.failure
        if failure["kind"] != "timeout":
            if failure["kind"] == "invalid_source":
                return f"The program could not be used: {failure['error']}"
            if failure["kind"] == "runtime_error":
                raised = f"\nRaised at: `{failure['line']}`" if failure.get("line") else ""
                return (failure["error"] or "")[-4000:] + raised
            return failure["error"] or "Invalid output"
        limit, mean = (
            failure.get("function_seconds_limit"),
            failure.get("mean_function_cpu_seconds"),
        )
        if limit is None:
            return "Timed out."
        rule = f"The limit is {limit:g} s of CPU time inside the function per instance, on average over the instances."
        if mean is not None:
            return f"Timed out: the function used {mean:.2f} s of CPU time per instance on average. {rule}"
        if (
            failure.get("error_type") == "TimeoutError"
            and failure.get("timeout_seconds") is not None
        ):
            used = failure.get("function_cpu_seconds")
            spent = (
                f"; the function had used {used:.2f} s of CPU time on it"
                if used is not None
                else ""
            )
            return f"Timed out: one instance did not finish within {failure['timeout_seconds']:g} s{spent}. {rule}"
        return f"Timed out. {rule}"

    def repair(self, failed):
        """Repair a failed program."""
        sections = list(self.common)
        failed_section = f"[Failed Program]\nOriginal Design: {idea_view(failed.idea)}\n```python\n{failed.code.rstrip()}\n```"
        timeout = failed.failure["kind"] == "timeout"
        sections += [
            failed_section,
            f"[Error]\n{self.error_text(failed)}",
            REPAIR_TIMEOUT if timeout else REPAIR,
            self._format(ANALYSIS_INSTRUCTIONS["Timeout" if timeout else "Repair"], design=False),
        ]
        return self._result(sections, "Repair")

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node.code.rstrip()}\n```"

    def measured(self, program):
        return f"Score: {score_text(program.fitness)}"

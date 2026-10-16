"""Prompts for TraceAAD."""

from .history import code_diff, final_program, path, score_text, verdict

SCORES = {
    "tsp_construct": ("the average length of the constructed tours", False),
    "cvrp_aco": (
        "the average total length of the best routes found by the ant colony",
        False,
    ),
    "op_aco": (
        "the negative average total prize of the best tours found by the ant colony",
        False,
    ),
    "online_bin_packing": ("the average number of bins used", False),
    "vrptw_construct": (
        "the average total travel distance of the constructed routes",
        False,
    ),
}
OPERATORS = ("Refine", "Explore", "Crossover")
INITIAL = "[Your Task: Design an Algorithm]\nWrite an algorithm that achieves the best score you can on this task."
ANOTHER_INITIAL = "[Your Task: Design a Better Algorithm]\nWrite an algorithm built on a better core idea than those of the algorithms above."
KNOWN = "Every version and attempt shown above has already been evaluated."
REFINE = f"[Your Task: Refine]\nDevelop the current algorithm further: keep its core idea and write a version that scores better than it. {KNOWN}"
EXPLORE = "[Your Task: Explore]\nWrite an algorithm that scores better than the best found so far by changing how the current algorithm makes its decisions, not by tuning it."
CROSSOVER = f"[Your Task: Crossover]\nBring into the current algorithm what the reference algorithm does well: keep the current algorithm's core idea and write a version that scores better than it. {KNOWN}"
REPAIR = "[Your Task: Repair]\nThe program failed during evaluation. Fix it so that it runs correctly within the time limit, keeping the algorithm it was meant to implement."
REPAIR_TIMEOUT = "[Your Task: Repair a Timeout]\nThe program did not finish within the time limit. Change it so that it finishes within the limit, keeping the main decisions of its algorithm."
FORMATION_INTRO = "The steps that produced the current algorithm, oldest first. Every version shown has been evaluated: each step gives its outcome, the Design of the version it produced, and the code diff from the previous version. When the first version of a step failed and was repaired, the step states the failure and shows the change to the repaired version."
IDEA_DISPLAY_CHARS = 2400
ERROR_DISPLAY_CHARS = 240


def idea_view(idea):
    return " ".join((idea or "").split())[:IDEA_DISPLAY_CHARS]


ANALYSIS = {
    "Refine": "what limits the current algorithm, and what change keeps its core idea and makes it score better",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to bring it in while keeping the current algorithm's core idea",
    "Repair": "what caused the failure, and how to fix it",
    "Timeout": "which computation takes the time, and how to change it so that the program finishes within the limit while keeping its main decisions",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
}
INIT_ANALYSIS = (
    "what determines the score on this task, and what kind of algorithm should do well"
)
BETTER_ANALYSIS = "what limits the core ideas of the algorithms above, and what better core idea overcomes it"
DESIGN = (
    "describe the design idea of the algorithm you will implement in at most 200 words"
)


def json_format(analysis=None, design=True, code=True):
    fields = [
        *([f'"analysis": {analysis}. It will not be shown again.'] if analysis else []),
        *([f'"design": {DESIGN}.'] if design else []),
        *(
            [
                '"code": the complete Python program. Write no comments or docstrings in the code.'
            ]
            if code
            else []
        ),
    ]
    count = {1: "one string field", 2: "two string fields", 3: "three string fields"}[
        len(fields)
    ]
    return f"[Output Format]\nReply with one JSON object with {count}:\n" + "\n".join(
        fields
    )


def short_error(text):
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return (lines[-1] if lines else "unknown error")[:ERROR_DISPLAY_CHARS]


import copy
import re
from benchmarks.tasks import FIXED_TASKS
from .verification import contain_prompt, verify_prompt

CALLED_ONCE = "The function is called once per instance, before the ant colony starts."
MANY_TIMES = re.compile(
    "Use efficient NumPy operations because the function is evaluated many\\s+times\\."
)
SERVICE = "At a customer, the vehicle waits until the time window opens if it arrives early, then spends the customer's service duration before moving on. The current time passed to the function already includes the waiting and service durations of the customers served so far; the service durations themselves are not passed to the function."
CVRP_COMPUTATION = "Within the evaluation time limit, the function\nmay perform any computation on the complete instance before returning its prior."
COMPUTATION = (
    " Within the time limit, the function may perform any computation on its inputs.",
    "",
)
OP_OBJECTIVE = (
    "\nThe evaluation objective is the negative mean collected prize; lower is better."
)
FIXED_SCORE = "the mean deviation of the final objective from a reference value, in percent of the reference; a negative score beats the reference"
REPAIR = "[Your Task: Repair]\nThe program failed during evaluation. Fix it, keeping the algorithm it was meant to implement."
MATCH = 'Each design below describes an algorithm for the same task.\n\n[Design A]\n{design}\n\n[Earlier Designs in This Search]\n{listed}\n\nWhich earlier designs are built on the same main computation as Design A?\nReply with one JSON object with two fields, in this order:\n"analysis": a string: the main computation of Design A, and for each earlier design whether it is built on the same main computation.\n"same": the numbers of those earlier designs, as a list of integers; an empty list if none.'
PROPOSAL = "[Your Proposal]\n{proposal}"
EARLIER = "[An Earlier Program for the Same Computation]\nAn evaluated program from this search, built on the same main computation as your Design.\n{measured}\nDesign: {design}\n```python\n{code}\n```"
IMPLEMENT = "[Your Task: Implement]\nImplement your Design in the current algorithm."


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


def match_prompt(design, ranked):
    """The match request for a proposal's Design and the ranked earlier programs (also used offline)."""
    listed = "\n".join(
        (f"[{program.id}] {idea_view(program.idea)}" for program, _ in ranked)
    )
    return MATCH.format(design=idea_view(design), listed=listed)


WRITTEN_FOR = (
    "[Proposal]\nThe current algorithm was written for this proposal.\n{proposal}"
)
MADE_FOR = "[Proposal]\nThis proposal was made for the current algorithm.\n{proposal}"
EARLIER_VERSION = "[An Earlier Program for This Proposal]\nWritten from the current algorithm for this proposal.\n{measured}\n```python\n{code}\n```"
READING = "[Reading of the Code]\nRead against the Design: {missing}"
EARLIER_READING = "[Reading of the Earlier Program]\nRead against the Design and against the current algorithm, from which it was written:\n{sentences}"
COMPLETE = "[Your Task: Complete]\nChange the current algorithm so that it carries out the Design, and keep the rest of what it computes."
COMPLETE_ANALYSIS = "what the current algorithm computes, and what it must compute to carry out the Design"


class PromptBuilder:
    ACTIONS = OPERATORS
    UNCHANGED_ROOT = "an initial design"
    INITIAL = INITIAL
    ANOTHER_INITIAL = ANOTHER_INITIAL
    REFINE = REFINE
    REPAIR = REPAIR
    REPAIR_TIMEOUT = REPAIR_TIMEOUT
    EXPLORE = EXPLORE
    CROSSOVER = CROSSOVER
    FORMATION_INTRO = FORMATION_INTRO
    ANALYSIS = ANALYSIS
    PROPOSAL = PROPOSAL
    EARLIER = EARLIER
    IMPLEMENT = IMPLEMENT

    def __init__(self, llm, task, evaluation, programs, config):
        evaluation = copy.copy(evaluation)
        if task in FIXED_TASKS:
            settings = ", ".join(
                (f"{name}={value}" for name, value in evaluation.outer_settings.items())
            )
            evaluation.task_description = evaluation.DESCRIPTION + (
                f"\nThe fixed solver settings are {settings}." if settings else ""
            )
            evaluation.design_notes = ""
            evaluation.score_meaning = FIXED_SCORE
        else:
            evaluation.task_description = (
                evaluation.task_description.replace(CVRP_COMPUTATION, "")
                .replace(OP_OBJECTIVE, "")
                .rstrip()
            )
        self.llm, self.task, self.evaluation = (llm, task, evaluation)
        self.programs, self.config = (programs, config)
        if task in SCORES:
            meaning, higher = SCORES[task]
        else:
            meaning, higher = (
                getattr(evaluation, "score_meaning", "the task objective"),
                False,
            )
        self.score_meaning, self.higher_is_better = (meaning, higher)
        description = self.task_text(evaluation.task_description.strip())
        notes = getattr(evaluation, "design_notes", "")
        if notes and notes.strip() not in description:
            description += "\n\n" + notes.strip()
        self.common = [
            "[Task]\n" + description,
            self.evaluation_text(meaning, higher),
            "[Target Function]\n```python\n"
            + str(evaluation.template_program).strip()
            + "\n```\nKeep the function name, arguments and return contract exactly as shown. The program must be self-contained: include every import, constant and helper it uses. Within the time limit, the function may perform any computation on its inputs.",
        ]
        self.common[2] = self.common[2].replace(*COMPUTATION)

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def _render(self, sections):
        return "\n\n".join((section for section in sections if section))

    def _result(self, sections, action, edge_ids=()):
        prompt = self._render(sections)
        return {
            "prompt": prompt,
            "input_tokens": self.count(prompt),
            "action": action,
            "history_edge_ids": list(edge_ids),
            "reference_history_edge_ids": [],
            "child_ids": [],
            "progress_ids": [],
            "token_count_mode": getattr(
                self.llm, "prompt_token_count_mode", "serving_tokenizer"
            ),
        }

    def failure(self, program):
        if program.failure["kind"] == "timeout":
            return "timed out"
        "What happened when a failed program was evaluated."
        failure = program.failure
        kind = failure["kind"]
        if kind == "invalid_source":
            return "the program could not be used: " + short_error(failure.get("error"))
        where = f" at `{failure['line']}`" if failure.get("line") else ""
        if kind == "invalid_output":
            return "invalid output: " + short_error(failure.get("error")) + where
        return "runtime error: " + short_error(failure.get("error")) + where

    def design(self, program):
        shown = idea_view(program.idea)
        missing, dropped = self._reading(program.id)
        if missing:
            shown += f" (read against this Design: {missing})"
        if dropped:
            shown += f" (besides this Design: {dropped})"
        return shown

    def followup(self, attempt, node, sequence_ids):
        """For an Explore attempt: what reading its program found, and what completing it gave."""
        exploration = attempt.exploration
        if attempt.action != "Explore" or not exploration or exploration["step"] != 0:
            return ""
        first = final_program(attempt, self.programs)
        text = ""
        if first.status == "valid":
            missing, dropped = self._reading(first.id)
            if missing:
                text += f"; read against its Design: {missing.rstrip('.')}"
            if dropped:
                text += f"; besides its Design: {dropped.rstrip('.')}"
        completions = sorted(
            (
                a
                for a in self.programs.values()
                if a.action == "Complete" and a.exploration["id"] == exploration["id"]
            ),
            key=lambda a: a.id,
        )
        for index, completion in enumerate(completions):
            final = final_program(completion, self.programs)
            text += f"; {('completed' if index == 0 else 'completed again')}: {self._outcome(final, node, sequence_ids)}"
            if final.status == "valid":
                missing, dropped = self._reading(final.id)
                text += ", still not carrying out the Design" if missing else ""
                text += (
                    ", still dropping part of the start algorithm" if dropped else ""
                )
        return text

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
            action += f" with an algorithm scoring {score_text(self.programs[event.reference_id].score)}"
        if failed is not None:
            action += ", then Repair"
        heading = f"Step {index}"
        if latest:
            heading += f" (latest: produced the {subject} algorithm)"
        end = f"score {score_text(child.score)} ({verdict(parent.score, child.score, self.higher_is_better)})"
        result = f"{heading} · {action} · score {score_text(parent.score)} → {end}"
        if failed is not None:
            result += f"\n  First version failed: {self.failure(failed)}"
        result += f"\n  Design: {self.design(child)}"
        result += f"\n  Code diff (previous → current):\n```diff\n{code_diff(parent.code, child.code)}\n```"
        return result

    def _formation(
        self,
        sequence,
        count,
        *,
        title="How the Current Algorithm Was Formed",
        subject="current",
        origin="initial algorithm",
    ):
        steps = self.steps(sequence)
        root, root_failed = steps[0]
        failed_note = (
            f" (its first version failed: {self.failure(root_failed)})"
            if root_failed
            else ""
        )
        if len(steps) == 1:
            return (
                f"[{title}]\nThe {subject} algorithm is {self.UNCHANGED_ROOT.format(origin=origin)}{failed_note}; no changes have been recorded yet.\nDesign: {self.design(root)}",
                [],
            )
        start = max(1, len(steps) - count)
        lines = [
            f"[{title}]",
            self.FORMATION_INTRO.replace("current algorithm", f"{subject} algorithm"),
        ]
        if start == 1:
            lines.append(
                f"Start · {origin} · score {score_text(root.score)}{failed_note}\n  Design: {self.design(root)}"
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
            and (p.action in OPERATORS)
        ]

    def _outcome(self, program, start, sequence_ids):
        if program.valid:
            return f"score {score_text(program.score)} ({verdict(start.score, program.score, self.higher_is_better)})"
        return "failed: " + self.failure(program)

    def _individual_lines(self, node, programs, sequence_ids):
        lines = []
        for program in programs:
            action = program.action
            if action == "Crossover" and program.reference_id in self.programs:
                action += f" with an algorithm scoring {score_text(self.programs[program.reference_id].score)}"
            outcome = self._outcome(program, node, sequence_ids)
            repair = final_program(program, self.programs)
            if repair.id != program.id:
                outcome += "; repaired: " + self._outcome(repair, node, sequence_ids)
            outcome += self.followup(program, node, sequence_ids)
            design = idea_view(program.idea) or "(none)"
            lines.append(
                f"Program {program.id} · {action} · Design: {design}\n  → {outcome}"
            )
        return lines

    def _attempts_section(self, node, attempts, shown, sequence_ids):
        if not attempts or shown == 0:
            return ("", [])
        listed = attempts[-shown:]
        improved = sum(
            (
                1
                for a in attempts
                if self._improved(final_program(a, self.programs), node)
            )
        )
        header = f"[Attempts From the Current Algorithm]\n{len(attempts)} attempt{('' if len(attempts) == 1 else 's')} started from the current algorithm; {improved} produced a new algorithm scoring better than it. "
        header += (
            "All are listed, oldest first, with their evaluated outcomes."
            if len(listed) == len(attempts)
            else f"The most recent {len(listed)} are listed, oldest first, with their evaluated outcomes."
        )
        return (
            header
            + "\n\n"
            + "\n\n".join(self._individual_lines(node, listed, sequence_ids)),
            [a.id for a in listed],
        )

    def _improved(self, program, node):
        return (
            program.valid
            and verdict(node.score, program.score, self.higher_is_better) == "improved"
        )

    def improvements(self):
        """Valid programs that set a new search best, after the first, oldest first."""
        best, result = (None, [])
        for program in sorted(
            (p for p in self.programs.values() if p.valid), key=lambda p: p.id
        ):
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
                "The changes that set a new best score, oldest first"
                + (
                    "."
                    if len(listed) == len(steps)
                    else f"; the most recent {len(listed)} of {len(steps)} are listed."
                )
                + "\n\n"
            )
            entries = []
            for previous, program in listed:
                event, action = (program, program.action)
                source = self.programs.get(program.parent_id)
                if source is not None and (not source.valid):
                    event, action = (source, source.action + ", then Repair")
                    source = self.programs.get(source.parent_id)
                origin = (
                    "as an initial algorithm"
                    if source is None
                    else f"from an algorithm scoring {score_text(source.score)}"
                )
                entries.append(
                    f"Program {event.id} · {action} {origin} → {score_text(program.score)} (previous best {score_text(previous.score)})\n  Design: {self.design(program)}"
                )
            text += "\n\n".join(entries) + "\n\n"
        text += f"Best score found so far in this search: {score_text(best.score)}."
        return (text, [program.id for _, program in listed])

    def initial(self, roots):
        sections = list(self.common)
        if roots:
            numbered = {node.id: index for index, node in enumerate(roots, 1)}
            entries = [
                f"Algorithm {numbered[node.id]} · {self.measured(node)} · Design: {self.design(node)}\n```python\n{node.code.rstrip()}\n```"
                for node in sorted(roots, key=lambda n: (n.fitness, n.id))
            ]
            direction = "higher" if self.higher_is_better else "lower"
            header = f"[Algorithms Designed So Far]\nEvaluated starting algorithms for this task, listed best first. Score is {self.score_meaning}; {direction} is better."
            sections.append(header + "\n\n" + "\n\n".join(entries))
        sections.extend(
            [self.ANOTHER_INITIAL, json_format(BETTER_ANALYSIS)]
            if roots
            else [self.INITIAL, json_format(INIT_ANALYSIS)]
        )
        return self._result(sections, "Init")

    def build(self, action, parent, *, reference=None):
        if action not in self.ACTIONS:
            raise ValueError(action)
        sequence = path(parent, self.programs)
        sequence_ids = {p.id for p in sequence}
        attempts = self.children_from(parent)
        current = self._current(parent)
        shown = min(self.config.attempts_shown, len(attempts))
        tried, tried_ids = self._attempts_section(parent, attempts, shown, sequence_ids)
        depth = len(self.steps(sequence)) - 1
        if action == "Refine":
            history, ids = self._formation(
                sequence, min(self.config.history_depth, depth)
            )
            sections = self.common + [
                current,
                history,
                tried,
                self.REFINE,
                self.answer_format("Refine"),
            ]
            result = self._result(sections, "Refine", ids)
        elif action == "Explore":
            best, progress_ids = self._progress_section(self.config.progress_shown)
            sections = self.common + [
                current,
                tried,
                best,
                self.EXPLORE,
                self.answer_format("Explore"),
            ]
            result = self._result(sections, "Explore")
            result["progress_ids"] = progress_ids
        else:
            if reference is None:
                raise ValueError("Crossover requires a reference")
            ref_section = f"[Reference Algorithm]\nAnother evaluated algorithm from this search.\n{self.measured(reference)}\nDesign: {self.design(reference)}\n```python\n{reference.code.rstrip()}\n```"
            reference_sequence = path(reference, self.programs)
            recent, ids = self._formation(sequence, min(4, depth))
            ref_history, ref_ids = self._formation(
                reference_sequence,
                min(4, len(self.steps(reference_sequence)) - 1),
                title="How the Reference Algorithm Was Formed",
                subject="reference",
            )
            sections = self.common + [
                current,
                recent,
                tried,
                ref_section,
                ref_history,
                self.CROSSOVER,
                self.answer_format("Crossover"),
            ]
            result = self._result(sections, "Crossover", ids)
            result["reference_history_edge_ids"] = ref_ids
        result["child_ids"] = tried_ids
        return result

    def answer_format(self, action):
        if action == "Explore" and self.config.linking:
            return json_format(self.ANALYSIS["Explore"], code=False)
        return json_format(self.ANALYSIS[action])

    def error_text(self, failed):
        failure = failed.failure
        if failure["kind"] != "timeout":
            failure = failed.failure
            if failure["kind"] == "invalid_source":
                return f"The program could not be used: {failure['error']}"
            if failure["kind"] == "runtime_error":
                raised = (
                    f"\nRaised at: `{failure['line']}`" if failure.get("line") else ""
                )
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
        failed_section = f"[Failed Program]\nDesign: {idea_view(failed.idea)}\n```python\n{failed.code.rstrip()}\n```"
        timeout = failed.failure["kind"] == "timeout"
        sections += [
            failed_section,
            f"[Error]\n{self.error_text(failed)}",
            self.REPAIR_TIMEOUT if timeout else self.REPAIR,
            json_format(
                self.ANALYSIS["Timeout" if timeout else "Repair"], design=False
            ),
        ]
        return self._result(sections, "Repair")

    def match(self, design, ranked):
        prompt = match_prompt(design, ranked)
        return (prompt, self.count(prompt))

    def implement_format(self):
        return json_format(design=False)

    def implement(self, explore_prompt, proposal, earlier):
        """Keep the proposal, original Explore material and matched implementation together."""
        head = explore_prompt[: explore_prompt.rindex("[Output Format]")].rstrip()
        proposal_block = self.PROPOSAL.format(proposal=proposal) if proposal else ""
        earlier_block = (
            self.EARLIER.format(
                measured=self.measured(earlier),
                design=idea_view(earlier.idea),
                code=earlier.code.rstrip(),
            )
            if earlier is not None
            else ""
        )
        tail = [self.IMPLEMENT, self.implement_format()]
        prompt = self._render([head, proposal_block, earlier_block, *tail])
        return {"prompt": prompt, "input_tokens": self.count(prompt)}

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node.code.rstrip()}\n```"

    def task_text(self, description):
        return corrected_description(self.task, description)

    def evaluation_text(self, meaning, higher):
        return f"Score: {meaning}. {('Higher' if higher else 'Lower')} is better."

    def measured(self, program):
        return f"Score: {score_text(program.score)}"

    def _reading(self, program_id):
        verdict = self.programs[program_id].verdict or {}
        return (
            verdict.get("missing") if verdict.get("implemented") is False else None,
            verdict.get("dropped") if verdict.get("kept") is False else None,
        )

    def verify(self, design, program):
        prompt = verify_prompt(self.common[0], design, program.code)
        return (prompt, self.count(prompt))

    def contain(self, design, program, start):
        prompt = contain_prompt(self.common[0], design, start.code, program.code)
        return (prompt, self.count(prompt))

    def complete(self, base, proposal, verdict, earlier, version=None):
        """Make ``base`` carry out its proposal's Design and keep the rest. ``base`` is the latest version when
        it kept the start algorithm, else the start itself, with the version shown."""
        if version is None:
            head = [self._current(base), WRITTEN_FOR.format(proposal=proposal)]
            reading = READING.format(missing=verdict["missing"])
        else:
            head = [
                self._current(base),
                MADE_FOR.format(proposal=proposal),
                EARLIER_VERSION.format(
                    measured=self.measured(version), code=version.code.rstrip()
                ),
            ]
            sentences = [
                s
                for s in (
                    verdict["missing"] if verdict["implemented"] is False else None,
                    verdict["dropped"],
                )
                if s
            ]
            reading = EARLIER_READING.format(sentences="\n".join(sentences))
        earlier_block = (
            self.EARLIER.format(
                measured=self.measured(earlier),
                design=idea_view(earlier.idea),
                code=earlier.code.rstrip(),
            )
            if earlier is not None
            else ""
        )
        tail = [reading, COMPLETE, json_format(COMPLETE_ANALYSIS, design=False)]
        return self._result(self.common + head + [earlier_block] + tail, "Complete")

"""V10.16 prompts: every shown program carries its measurements, and every operator sees the experience around it.

Context states facts attached to the object they describe; instructions are
one goal sentence (docs/03-现象与检验/2026-10-01-生成目标与计算限制.md).
V10.16 adds the experience the search has gathered, not only the programs it
kept: the attempts that started from the current algorithm with their measured
outcomes (failures and reproductions included), the improvements of the search
best for Explore, failed first versions folded into their repaired steps, and the call
count and time inside the function for every evaluation.
"""

from .history import code_diff, path, score_text, verdict


SCORES = {
    "tsp_construct": ("the average length of the constructed tours", False),
    "cvrp_aco": ("the average total length of the best routes found by the ant colony", False),
    "op_aco": ("the average total prize of the best tours found by the ant colony", True),
    "online_bin_packing": ("the average number of bins used", False),
    "vrptw_construct": ("the average total travel distance of the constructed routes", False),
}

OPERATORS = ("Refine", "Explore", "Crossover")

INITIAL = """[Your Task: Design an Algorithm]
Write an algorithm for this task that scores as well as possible, built on a clear core idea."""

ANOTHER_INITIAL = """[Your Task: Design a Different Algorithm]
Write an algorithm built on a core idea different from those of the algorithms above that scores better than all of them."""

# Every operator aims past everything the prompt shows. The attempts listed
# under the current algorithm are shown versions too, so a known outcome is
# never an answer.
REFINE = """[Your Task: Refine]
Continue this line of development: write an algorithm that scores better than every version and attempt shown above."""

REFINE_ROOT = """[Your Task: Refine]
Continue developing the current algorithm: write a version that scores better than it and than every attempt shown above."""

EXPLORE = """[Your Task: Explore]
Write an algorithm that scores better than the best found so far by changing how the current algorithm makes its decisions, not by tuning it."""

CROSSOVER = """[Your Task: Crossover]
Combine the two lines of development: write an algorithm that scores better than every version and attempt shown above, bringing into the current algorithm what the reference algorithm does well."""

REPAIR = """[Your Task: Repair]
The program failed during evaluation. Fix it so that it runs correctly within the time limit, keeping the algorithm it was meant to implement."""

FORMATION_INTRO = ("The steps that produced the current algorithm, oldest first. Every version shown has been evaluated: "
                   "each step gives its outcome, the Design of the version it produced, and the code diff "
                   "from the previous version. When the first version of a step failed and was repaired, "
                   "the step states the failure and shows the change to the repaired version.")

IDEA_DISPLAY_CHARS = 2400
ERROR_DISPLAY_CHARS = 240


class ContextTooLong(ValueError):
    pass


def idea_view(idea):
    return " ".join((idea or "").split())[:IDEA_DISPLAY_CHARS]


ANALYSIS = {
    "Refine": "what limits this line of development so far, and what change should take it past every version and attempt",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to combine them",
    "Repair": "what caused the failure, and how to fix it",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
    "Init": None,
}


def output_format(action):
    lines = ["[Output Format]"]
    if ANALYSIS[action]:
        lines += ["Reply in this order:",
                  f"Analysis: <a few sentences: {ANALYSIS[action]}. It will not be shown again>"]
    else:
        lines.append("Reply with a Design followed by one Python code block:")
    lines += ["Design: <one or two sentences (at most 60 words) stating the core idea of the algorithm you will implement>",
              "Code:\n```python\n<the complete program>\n```",
              "Write no comments or docstrings in the code, and nothing after the code block."]
    return "\n".join(lines)


def short_error(text):
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return (lines[-1] if lines else "unknown error")[:ERROR_DISPLAY_CHARS]


class PromptBuilder:
    def __init__(self, llm, task, evaluation, programs, attempts, config):
        self.llm, self.task, self.evaluation = llm, task, evaluation
        self.programs, self.attempts, self.config = programs, attempts, config
        if task in SCORES:
            meaning, higher = SCORES[task]
        else:
            meaning, higher = "the task fitness", True
        self.higher_is_better = higher
        description = evaluation.task_description.strip()
        notes = getattr(evaluation, "design_notes", "")
        if notes:
            description += "\n\n" + notes.strip()
        self.timeout = evaluation.timeout_seconds
        timeout = format(self.timeout, "g") if self.timeout is not None else "an unspecified number of"
        self.common = [
            "[Task]\n" + description,
            "[Evaluation]\nEach program is evaluated on a fixed set of training instances.\n"
            f"Score: {meaning}. {'Higher' if higher else 'Lower'} is better.\n"
            f"The whole evaluation must finish within {timeout} seconds.",
            "[Target Function]\n```python\n" + str(evaluation.template_program).strip() +
            "\n```\nKeep the function name, arguments and return contract exactly as shown. "
            "The program must be self-contained: include every import, constant and helper it uses. "
            "Within the time limit, the function may perform any computation on its inputs.",
        ]

    # ---------- facts about one program ----------

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def block_count(self, text):
        return self.llm.count_tokens(text)

    def _render(self, sections):
        return "\n\n".join(section for section in sections if section)

    def _result(self, sections, action, edge_ids=(), trims=()):
        prompt = self._render(sections)
        return {"prompt": prompt, "input_tokens": self.count(prompt), "action": action,
                "history_edge_ids": list(edge_ids), "trims": list(trims),
                "reference_history_edge_ids": [], "attempt_ids": [], "progress_ids": [],
                "token_count_mode": getattr(self.llm, "prompt_token_count_mode", "serving_tokenizer")}

    @staticmethod
    def _seconds(seconds):
        return "under 0.1 s" if seconds < 0.05 else f"about {seconds:.1f} s"

    @classmethod
    def _calls(cls, calls, seconds):
        return (f"{calls} call{'' if calls == 1 else 's'} to the function, {cls._seconds(seconds)} inside it"
                if calls is not None else "")

    def measured(self, program):
        """Score, evaluation time and calls of a valid program."""
        parts = [f"Score: {score_text(program['score'])}"]
        if program.get("eval_seconds"):
            parts.append(f"Evaluation time: about {max(program['eval_seconds'], 0.1):.1f} s")
        calls = self._calls(program.get("calls"), program.get("function_seconds") or 0.0)
        if calls:
            parts.append(calls)
        return " · ".join(parts)

    def failure(self, program):
        """What happened when a failed program was evaluated."""
        failure = program["failure"]
        kind = failure["kind"]
        if kind == "timeout":
            limit = (f"the {format(self.timeout, 'g')} s time limit" if self.timeout is not None
                     else "the time limit")
            calls, inside = failure.get("calls"), failure.get("function_seconds") or 0.0
            if calls is None:
                return f"stopped at {limit}"
            if failure.get("call_running"):
                return (f"stopped at {limit} inside call {calls + 1} to the function, "
                        f"after {calls} completed call{'' if calls == 1 else 's'} "
                        f"({self._seconds(inside)} inside the function in total)")
            return f"stopped at {limit} after {self._calls(calls, inside)}"
        if kind == "invalid_source":
            return "the program could not be used: " + short_error(failure.get("error"))
        where = f" at `{failure['line']}`" if failure.get("line") else ""
        if kind == "invalid_output":
            return "invalid output: " + short_error(failure.get("error")) + where
        return "runtime error: " + short_error(failure.get("error")) + where

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node['code'].rstrip()}\n```"

    # ---------- formation path ----------

    @staticmethod
    def steps(sequence):
        """The valid versions on a formation path, each with the failed first version
        its step produced before the repair (None when the step did not fail).

        A failed version is folded into the repair made from it: what was tried
        and how it failed is experience, but its code is not a version to build on.
        """
        steps, failed = [], None
        for program in sequence:
            if program["valid"]:
                steps.append((program, failed))
                failed = None
            else:
                failed = program
        return steps

    def _edge(self, parent, child, failed, index, *, latest=False, subject="current"):
        event = failed or child  # the generation that started from ``parent``
        action = event["action"]
        if action == "Crossover" and event.get("reference_id") is not None:
            action += f" with an algorithm scoring {score_text(self.programs[event['reference_id']]['score'])}"
        if failed is not None:
            action += ", then Repair"
        heading = f"Step {index}"
        if latest:
            heading += f" (latest: produced the {subject} algorithm)"
        end = (f"score {score_text(child['score'])} "
               f"({verdict(parent['score'], child['score'], self.higher_is_better)})")
        if child.get("eval_seconds"):
            end += f", evaluation time about {max(child['eval_seconds'], 0.1):.1f} s"
        result = f"{heading} · {action} · score {score_text(parent['score'])} → {end}"
        if failed is not None:
            result += f"\n  First version failed: {self.failure(failed)}"
        result += f"\n  Design: {idea_view(child['idea'])}"
        result += f"\n  Code diff (previous → current):\n```diff\n{code_diff(parent['code'], child['code'])}\n```"
        return result

    def _formation(self, sequence, count, *, title="How the Current Algorithm Was Formed", subject="current"):
        steps = self.steps(sequence)
        root, root_failed = steps[0]
        failed_note = f" (its first version failed: {self.failure(root_failed)})" if root_failed else ""
        if len(steps) == 1:
            return (f"[{title}]\nThe {subject} algorithm is an initial design{failed_note}; no changes have been "
                    f"recorded yet.\nDesign: {idea_view(root['idea'])}"), []
        start = max(1, len(steps) - count)
        lines = [f"[{title}]", FORMATION_INTRO.replace("current algorithm", f"{subject} algorithm")]
        if start == 1:
            lines.append(f"Start · initial algorithm · score {score_text(root['score'])}{failed_note}\n"
                         f"  Design: {idea_view(root['idea'])}")
        else:
            lines.append(f"The path has {len(steps)-1} steps; showing the most recent {len(steps)-start} steps.")
        for index in range(start, len(steps)):
            lines.append(self._edge(steps[index-1][0], steps[index][0], steps[index][1], index,
                                    latest=index == len(steps)-1, subject=subject))
        return lines[0] + "\n" + lines[1] + "\n\n" + "\n\n".join(lines[2:]), [p["id"] for p, _ in steps[start:]]

    # ---------- experience around the current algorithm ----------

    def attempts_from(self, node):
        """Operator attempts that started from ``node``, oldest first."""
        return [a for a in sorted(self.attempts.values(), key=lambda a: a["id"])
                if a["parent_id"] == node["id"] and a.get("repair_of") is None and a["action"] in OPERATORS]

    def _outcome(self, attempt, start, sequence_ids):
        """The measured outcome of one attempt, relative to the program it started from."""
        status = attempt["status"]
        program = self.programs.get(attempt.get("program_id"))
        if status == "delivery_failed":
            return "the reply contained no usable program"
        if status == "valid":
            return (f"score {score_text(program['score'])} "
                    f"({verdict(start['score'], program['score'], self.higher_is_better)})"
                    + (f", evaluation time about {max(program['eval_seconds'], 0.1):.1f} s"
                       if program.get("eval_seconds") else ""))
        if status == "copied_reference":
            return "the same code as its reference algorithm"
        if status in {"duplicate", "known_failure"} and program is not None:
            if program["id"] == start["id"]:
                return "the same code as the algorithm it started from"
            where = ("an earlier version on this path" if program["id"] in sequence_ids
                     else "an algorithm already evaluated in this search")
            if program["valid"]:
                return f"the same code as {where} (score {score_text(program['score'])})"
            return f"the same code as a program that already failed ({self.failure(program)})"
        if program is not None and not program["valid"]:
            return "failed: " + self.failure(program)
        return "failed"

    def _attempt_lines(self, node, attempts, sequence_ids):
        repairs = self._repairs()
        lines = []
        for attempt in attempts:
            action = attempt["action"]
            if action == "Crossover" and attempt.get("reference_id") in self.programs:
                action += f" with an algorithm scoring {score_text(self.programs[attempt['reference_id']]['score'])}"
            outcome = self._outcome(attempt, node, sequence_ids)
            repair = repairs.get(attempt["id"])
            if repair is not None:
                if repair.get("program_id") is not None and repair["program_id"] == attempt.get("program_id"):
                    outcome += "; the repair returned the same failing program"
                else:
                    outcome += "; repaired: " + self._outcome(repair, node, sequence_ids)
            design = idea_view(attempt.get("idea")) or "(none)"
            lines.append(f"Attempt {attempt['id']} · {action} · Design: {design}\n  → {outcome}")
        return lines

    def _repairs(self):
        return {a["repair_of"]: a for a in self.attempts.values() if a.get("repair_of") is not None}

    def _attempts_section(self, node, attempts, shown, sequence_ids):
        if not attempts or shown == 0:
            return "", []
        listed = attempts[-shown:]
        repairs = self._repairs()
        improved = sum(1 for a in attempts if self._improved(repairs.get(a["id"], a), node))
        header = (f"[Attempts From the Current Algorithm]\n{len(attempts)} attempt{'' if len(attempts) == 1 else 's'} "
                  f"started from the current algorithm; {improved} produced a new algorithm scoring better than it. ")
        header += ("All are listed, oldest first, with their evaluated outcomes." if len(listed) == len(attempts)
                   else f"The most recent {len(listed)} are listed, oldest first, with their evaluated outcomes.")
        return header + "\n\n" + "\n\n".join(self._attempt_lines(node, listed, sequence_ids)), [a["id"] for a in listed]

    def _improved(self, final, node):
        """Whether an attempt (its repair, if it had one) produced a new program better than ``node``."""
        program = self.programs.get(final.get("program_id"))
        return (final["status"] == "valid" and program is not None and
                verdict(node["score"], program["score"], self.higher_is_better) == "improved")

    # ---------- global experience for Explore ----------

    def improvements(self):
        """Valid programs that set a new search best, after the first, oldest first."""
        best, result = None, []
        for program in sorted((p for p in self.programs.values() if p["valid"]), key=lambda p: p["id"]):
            if best is None or program["fitness"] > best["fitness"]:
                if best is not None:
                    result.append((best, program))
                best = program
        return best, result

    def _progress_section(self, shown):
        best, steps = self.improvements()
        listed = steps[-shown:] if shown else []
        text = "[How the Best Score Improved in This Search]\n"
        if listed:
            text += ("The changes that set a new best score, oldest first"
                     + ("." if len(listed) == len(steps) else
                        f"; the most recent {len(listed)} of {len(steps)} are listed.") + "\n\n")
            entries = []
            for previous, program in listed:
                # A repaired program is credited to the generation it repaired.
                event, action = program, program["action"]
                source = self.programs.get(program["parent_id"])
                if source is not None and not source["valid"]:
                    event, action = source, source["action"] + ", then Repair"
                    source = self.programs.get(source["parent_id"])
                origin = ("as an initial algorithm" if source is None
                          else f"from an algorithm scoring {score_text(source['score'])}")
                timing = (f" · evaluation time about {max(program['eval_seconds'], 0.1):.1f} s"
                          if program.get("eval_seconds") else "")
                entries.append(f"Attempt {event['id']} · {action} {origin} → "
                               f"{score_text(program['score'])} (previous best {score_text(previous['score'])})"
                               f"{timing}\n  Design: {idea_view(program['idea'])}")
            text += "\n\n".join(entries) + "\n\n"
        text += f"Best score found so far in this search: {score_text(best['score'])}."
        return text, [program["id"] for _, program in listed]

    # ---------- prompts ----------

    def initial(self, roots):
        sections = list(self.common)
        trims = []
        shown = list(roots) if len(roots) >= 4 else []

        def root_section():
            entries = [f"Algorithm {i} · {self.measured(n)} · Design: {idea_view(n['idea'])}\n"
                       f"```python\n{n['code'].rstrip()}\n```" for i, n in enumerate(shown, 1)]
            return "[Algorithms Designed So Far]\n" + "\n\n".join(entries)

        if shown:
            while len(shown) > 1 and self.block_count(root_section()) > self.config.root_tokens:
                trims.append(f"root:{shown.pop(0)['id']}")
            sections.append(root_section())
        sections.extend([ANOTHER_INITIAL if shown else INITIAL, output_format("Init")])
        while shown and len(shown) > 1 and self.count(self._render(sections)) > self.config.max_input_tokens:
            trims.append(f"root:{shown.pop(0)['id']}")
            sections[3] = root_section()
        result = self._result(sections, "Init", trims=trims)
        if result["input_tokens"] > self.config.max_input_tokens or (shown and self.block_count(root_section()) > self.config.root_tokens):
            raise ContextTooLong("initial prompt exceeds context with one required root")
        return result

    def build(self, action, parent, *, reference=None):
        if action not in OPERATORS:
            raise ValueError(action)
        sequence = path(parent, self.programs)
        sequence_ids = {p["id"] for p in sequence}
        attempts = self.attempts_from(parent)
        current = self._current(parent)
        trims = []
        shown = min(self.config.attempts_shown, len(attempts))
        depth = len(self.steps(sequence)) - 1
        if action == "Refine":
            count = min(self.config.history_depth, depth)
            while True:
                history, ids = self._formation(sequence, count)
                tried, tried_ids = self._attempts_section(parent, attempts, shown, sequence_ids)
                sections = self.common + [current, history, tried, REFINE_ROOT if depth == 0 else REFINE,
                                          output_format("Refine")]
                result = self._result(sections, "Refine", ids, trims)
                result["attempt_ids"] = tried_ids
                if result["input_tokens"] <= self.config.max_input_tokens:
                    return result
                if count > 1:
                    count -= 1
                    trims.append("oldest_history")
                elif shown > 0:
                    shown -= 1
                    trims.append("oldest_attempt")
                else:
                    raise ContextTooLong("Refine prompt with required latest history exceeds context")
        if action == "Explore":
            progress = self.config.progress_shown
            while True:
                tried, tried_ids = self._attempts_section(parent, attempts, shown, sequence_ids)
                best, progress_ids = self._progress_section(progress)
                sections = self.common + [current, tried, best, EXPLORE, output_format("Explore")]
                result = self._result(sections, "Explore", trims=trims)
                result["attempt_ids"], result["progress_ids"] = tried_ids, progress_ids
                if result["input_tokens"] <= self.config.max_input_tokens:
                    return result
                if progress > 0 and progress_ids:
                    progress -= 1
                    trims.append("oldest_improvement")
                elif shown > 0:
                    shown -= 1
                    trims.append("oldest_attempt")
                else:
                    raise ContextTooLong("minimum Explore prompt exceeds context")
        if reference is None:
            raise ValueError("Crossover requires a reference")
        ref_section = (f"[Reference Algorithm]\nAnother evaluated algorithm from this search.\n"
                       f"{self.measured(reference)}\nDesign: {idea_view(reference['idea'])}\n"
                       f"```python\n{reference['code'].rstrip()}\n```")
        reference_sequence = path(reference, self.programs)
        counts = [min(4, depth), min(4, len(self.steps(reference_sequence)) - 1)]
        while True:
            recent, ids = self._formation(sequence, counts[0])
            ref_history, ref_ids = self._formation(reference_sequence, counts[1],
                                                   title="How the Reference Algorithm Was Formed", subject="reference")
            tried, tried_ids = self._attempts_section(parent, attempts, shown, sequence_ids)
            sections = self.common + [current, recent, tried, ref_section, ref_history, CROSSOVER,
                                      output_format("Crossover")]
            result = self._result(sections, "Crossover", ids, trims)
            result["reference_history_edge_ids"] = ref_ids
            result["attempt_ids"] = tried_ids
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            candidates = [i for i, count in enumerate(counts) if count > 1]
            if candidates:
                side = max(candidates, key=lambda i: self.block_count([recent, ref_history][i]))
                counts[side] -= 1
                trims.append("oldest_history" if side == 0 else "oldest_reference_history")
            elif shown > 0:
                shown -= 1
                trims.append("oldest_attempt")
            else:
                break
        fallback = self.build("Refine", parent)
        fallback["trims"] = trims + ["crossover_context_fallback"] + fallback["trims"]
        return fallback

    def repair(self, failed, error_text):
        """Repair a failed program; a timeout is stated as the measurement it is."""
        sections = list(self.common)
        failed_section = f"[Failed Program]\nDesign: {idea_view(failed['idea'])}\n```python\n{failed['code'].rstrip()}\n```"
        sections += [failed_section, f"[Error]\n{error_text}", REPAIR, output_format("Repair")]
        result = self._result(sections, "Repair")
        if result["input_tokens"] > self.config.max_input_tokens:
            raise ContextTooLong("repair prompt exceeds context")
        return result

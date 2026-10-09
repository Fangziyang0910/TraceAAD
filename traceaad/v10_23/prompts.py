"""V10.23 context: each step is defined by the kind of change it makes, and decides from material that suits it.

What a step changes in the code follows mostly from the changes shown in its
context, then from the question its analysis answers, and least from its goal
sentence. A Deepen goal on Refine's material made Refine's small edits; the
same goal on the task and the current algorithm alone made searches 12-17
times as costly (docs/03-现象与检验/2026-10-06-V10.20重放与ACO评价噪声.md).
Formation paths are about 70% of Refine and Crossover prompts and mostly
show coefficient edits, so the search repeats the kind of change it has
already made: in V10.22's TSP runs a Crossover between two programs that
differed in one coefficient set that coefficient to the reference's value,
and from cheap starting programs only 2% of Refine and Explore children added
computation (V10.21: 20-29%). The search stayed with one-step scoring rules
(6.31 at attempt 700 against 5.87-5.89).

The steps therefore differ in the kind of change, as in LACE, EoH and
ShinkaEvolve, and each decides from its own material:

- Refine changes one part of the computation and keeps the rest; it reads the
  formation path, the record of earlier small changes.
- Explore changes the kind of computation that decides the output. It reads
  only the current algorithm and the attempts made from it, so no record of
  coefficient edits frames the change. Its goal states the difference between
  a formula that estimates what an option leads to and a computation that
  works it out by constructing, simulating or improving what follows. Asked
  instead which effects of its output the algorithm leaves out, the model
  added one more feature to the formula on every task (a return-to-start term,
  a machine-load term); with the difference stated, first versions on the same
  early V10.22 states completed tours (TSP 6.91 -> 6.21), simulated the rest of
  a schedule or computed the makespan after each move
  (docs/03-现象与检验/2026-10-09-V10.23行为检查.md).
- Crossover combines the computations of two algorithms; it reads their code
  and scores, not their formation paths, and does not presume that the
  reference is better.
- Develop continues a new computation an Explore introduced; it reads the
  formation path from the program the Explore started from, not the older
  edits, which led development back to the old program.

Design states what the algorithm computes, so later steps read computations.
A timed-out program is reported with how far its run got: the calls the
function returned from on the instance that ran out, of the calls the framework
makes per instance (from this search's valid programs). Without it a repair
cannot tell a 2-fold overshoot from a 100-fold one; Explore's first versions
time out on most tasks (TSP stopped after 0-20 of 48 calls, FSSP 0-46 of 200,
graph colouring 300-480 of about 480).
The prompts still say nothing about time, and name no method.
"""

import statistics

from traceaad.common.prompts import ContextTooLong, KNOWN, idea_view
from traceaad.common.history import path
from traceaad.v10_22.prompts import PromptBuilder as V1022Prompts

REFINE = f"""[Your Task: Refine]
Improve one part of the current algorithm's computation and keep the rest unchanged, so that it scores better than the current algorithm. {KNOWN}"""

EXPLORE = f"""[Your Task: Explore]
Write a version of the current algorithm that decides its output through a different kind of computation, not by changing terms or parameters. A formula over features of the options only estimates what each option leads to; a computation can also work out what an option leads to, by constructing, simulating or improving the part of the solution that follows from it. Keep the parts of the current algorithm that the new computation still uses. The new version should score better than the current algorithm. {KNOWN}"""

CROSSOVER = f"""[Your Task: Crossover]
Write one algorithm that combines computations of the current algorithm and the reference algorithm, keeping from each what helps the score, so that it scores better than the current algorithm. {KNOWN}"""

DEVELOP = f"""[Your Task: Develop]
The steps above, from the algorithm the exploration started from, changed how the algorithm arrives at its output. Develop this new computation: keep it, and improve how it is implemented and used, so that the algorithm scores better than it does now. {KNOWN}"""

ANALYSIS = {
    "Refine": "which part of the current algorithm's computation you will change, and how the change makes it score better",
    "Explore": ("what the current algorithm works out before it decides its output and what it only estimates, "
                "and what the new version will work out instead"),
    "Crossover": "what each algorithm computes that the other does not, and which of these computations the new algorithm keeps",
    "Develop": "what in the new computation does not yet work well, and how this version improves it",
    "Repair": "what caused the failure, and how to fix it",
    "Init": None,
}

DESIGN = "Design: <one or two sentences (at most 60 words): what the algorithm you implement computes to produce its output>"


class PromptBuilder(V1022Prompts):
    ACTIONS = ("Refine", "Explore", "Crossover", "Develop")
    REFINE = REFINE
    REFINE_ROOT = REFINE
    EXPLORE = EXPLORE
    CROSSOVER = CROSSOVER
    DEVELOP = DEVELOP
    ANALYSIS = ANALYSIS

    def calls_per_instance(self):
        """How many times the framework calls the function on one instance, from the search's valid programs."""
        n = getattr(self.evaluation, "n_instance", None)
        counts = [p["calls"] / n for p in self.programs.values() if n and p.get("valid") and p.get("calls")]
        if not counts:
            return None
        total = round(statistics.median(counts))
        return str(total) if max(counts) - min(counts) < 0.5 else f"about {total}"

    def failure(self, program):
        failure = program["failure"]
        if failure["kind"] != "timeout" or failure.get("calls") is None:
            return super().failure(program)
        # How far the run got, so a repair can tell how much to cut; no seconds, as elsewhere.
        done, total = failure["calls"], self.calls_per_instance()
        if total == "1":
            return "timed out: the evaluation stopped before the function returned from its single call on an instance"
        of = f" of its {total}" if total else ""
        return (f"timed out: on one instance, the evaluation stopped after the function "
                f"had returned from {done}{of} calls")

    def error_text(self, failed):
        if failed["failure"]["kind"] == "timeout":
            text = self.failure(failed)
            return text[0].upper() + text[1:] + "."
        return super().error_text(failed)

    def output_format(self, action):
        lines = ["[Output Format]"]
        if self.ANALYSIS[action]:
            lines += ["Reply in this order:",
                      f"Analysis: <a few sentences: {self.ANALYSIS[action]}. It will not be shown again>"]
        else:
            lines.append("Reply with a Design followed by one Python code block:")
        lines += [DESIGN, "Code:\n```python\n<the complete program>\n```", self.CLOSING]
        return "\n".join(lines)

    def build(self, action, parent, *, reference=None, source=None):
        if action == "Crossover":
            return self.crossover(parent, reference)
        if action == "Develop":
            return self.develop(parent, source)
        return super().build(action, parent, reference=reference)

    def _attempts_only(self, action, parent, before, after):
        """The current algorithm, its attempts and ``before``/``after`` sections; attempts are trimmed oldest first."""
        sequence_ids = {p["id"] for p in path(parent, self.programs)}
        attempts = self.attempts_from(parent)
        shown, trims = min(self.config.attempts_shown, len(attempts)), []
        while True:
            tried, ids = self._attempts_section(parent, attempts, shown, sequence_ids)
            sections = self.common + [self._current(parent), *before, tried, *after,
                                      getattr(self, action.upper()), self.output_format(action)]
            result = self._result(sections, action, trims=trims)
            result["attempt_ids"] = ids
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            if shown == 0:
                raise ContextTooLong(f"minimum {action} prompt exceeds context")
            shown -= 1
            trims.append("oldest_attempt")

    def crossover(self, parent, reference):
        if reference is None:
            raise ValueError("Crossover requires a reference")
        section = (f"[Reference Algorithm]\nAnother evaluated algorithm from this search.\n"
                   f"{self.measured(reference)}\nDesign: {idea_view(reference['idea'])}\n"
                   f"```python\n{reference['code'].rstrip()}\n```")
        return self._attempts_only("Crossover", parent, [], [section])

    def develop(self, parent, source):
        """The new computation's own path: the steps from the program the exploration started from."""
        sequence = path(parent, self.programs)
        steps = self.steps(sequence)
        start = next(i for i, (p, _) in enumerate(steps) if p["id"] == source["id"])
        history, edge_ids = self._formation(sequence, len(steps) - 1 - start)
        result = self._attempts_only("Develop", parent, [history], [])
        result["history_edge_ids"] = edge_ids
        return result

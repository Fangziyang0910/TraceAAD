"""V10.21 context: time in the unit a program spends it.

Time. V10.20 stated one limit for the whole training evaluation, while the
function is called once per instance (ACO) or thousands of times per instance
(graph colouring). Programs that read the clock set the whole limit as the
budget of one call (CVRP: 115 s in a function called once for each of 10
instances; graph colouring: 45 s in a function called 7,113 times) and timed
out; Deepen, the step that spends more time, timed out most (8.7%). Time
belongs to an instance, so the evaluation states the per-instance budget and
every measurement states the time per instance and the calls per instance.

Comments. The ban on code comments stays. Lifting it (V10.21's first
launch) spent 21.6% of output tokens on comments, which canonical storage
drops, and runtime errors did not fall.
"""

from benchmarks.tasks import CO_TASKS, HELDOUT_TIMEOUT, INSTANCE_SECONDS
from traceaad.common.history import score_text
from traceaad.common.prompts import KNOWN
from traceaad.v10_20.prompts import ANALYSIS as V1020_ANALYSIS, PromptBuilder as V1020Prompts, test_sets

DEEPEN = f"""[Your Task: Deepen]
Keep the current algorithm's decision rule and use it to guide a search that spends more of the per-instance time budget on finding better decisions, so that the program scores better than the current algorithm. {KNOWN}"""

ANALYSIS = {**V1020_ANALYSIS,
            "Deepen": ("which decisions of the current algorithm a search guided by it could improve, and how much "
                       "computation per instance that search can use within the time budget")}


def duration(seconds):
    if seconds >= 1:
        return f"about {seconds:.1f} s"
    if seconds >= 0.01:
        return f"about {seconds:.2f} s"
    if seconds >= 0.001:
        return f"about {seconds * 1000:.0f} ms"
    return "under 1 ms"


class PromptBuilder(V1020Prompts):
    ANALYSIS = ANALYSIS
    DEEPEN = DEEPEN

    def __init__(self, llm, task, evaluation, programs, attempts, config):
        self.instances = max(1, int(getattr(evaluation, "n_instance", 1) or 1))
        timeout = evaluation.timeout_seconds
        self.budget = (INSTANCE_SECONDS if task in CO_TASKS else
                       timeout / self.instances if timeout is not None else None)
        super().__init__(llm, task, evaluation, programs, attempts, config)
        self.ANALYSIS = ANALYSIS

    def evaluation_text(self, meaning, higher):
        if self.task not in CO_TASKS:
            return super().evaluation_text(meaning, higher)
        order = "" if "lower is better" in meaning.lower() else " Lower is better."
        return (f"[Evaluation]\nEach program is evaluated on {self.training_set}.\n"
                f"Score: {meaning}.{order}\n"
                f"Each instance has a time budget of {INSTANCE_SECONDS} seconds, which includes the fixed solver. "
                f"Instances are evaluated one after another, so the whole evaluation must finish within "
                f"{self._limit()} seconds.\n"
                f"After the search, the final program is tested on {test_sets(self.task)}, "
                f"under the same per-instance budget ({format(HELDOUT_TIMEOUT[self.task], 'g')} seconds in total).")

    def per_instance(self, seconds):
        return duration(seconds / self.instances)

    def measured(self, program):
        parts = [f"Score: {score_text(program['score'])}"]
        if program.get("eval_seconds"):
            of_budget = f" of the {format(self.budget, 'g')} s budget" if self.budget is not None else ""
            parts.append(f"Time per instance: {self.per_instance(program['eval_seconds'])}{of_budget}")
        calls = program.get("calls")
        if calls:
            per_call = duration((program.get("function_seconds") or 0.0) / calls)
            parts.append(f"{calls / self.instances:.0f} call{'' if calls == self.instances else 's'} "
                         f"to the function per instance, {per_call} each")
        return " · ".join(parts)

    def elapsed(self, seconds):
        return f"time per instance {self.per_instance(seconds)}"

    def limit_text(self):
        if self.budget is None:
            return super().limit_text()
        return (f"the {self._limit()} s limit for {self.instances} instance{'' if self.instances == 1 else 's'} "
                f"({format(self.budget, 'g')} s per instance)")

    def error_text(self, failed):
        if failed["failure"]["kind"] != "timeout":
            return super().error_text(failed)
        text = "The evaluation " + self.failure(failed) + "."
        source = self.programs.get(failed["parent_id"])
        if source is not None and source.get("valid") and source.get("eval_seconds"):
            text += f" The algorithm it was developed from measures: {self.measured(source)}."
        return text

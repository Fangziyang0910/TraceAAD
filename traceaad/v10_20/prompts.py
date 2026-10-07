"""V10.20 context: true facts about computation, and the Deepen step.

Each step is defined by its goal and by the material it decides from
(docs/01-搜索方法/当前规则引导的搜索.md). Deepen keeps the current algorithm's
rule and decides with more search; it sees the current algorithm, its measured
cost and the earlier Deepen attempts from it, not the formation path: a path of
many small edits made the model continue with small edits whatever the goal said.

Three facts change against V10.17 for every step:

- The ACO task texts said the function "is evaluated many times"; it is called
  once per instance, before the colony starts. The VRPTW text did not say that
  the evaluator adds each served customer's service duration to the time.
- Each measured program states its evaluation time against the time limit.
- The evaluation section states the training instances and the larger test
  sets the final program runs on, with their time limits.

Every number comes from ``benchmarks``, so the text follows the protocol.
The new fixed-scale tasks state their primary held-out data through the same interface.
"""

import re

from benchmarks.cvrp_aco.dataset import get_split_spec as cvrp_split
from benchmarks.generated_data_config import get_generated_task_kwargs
from benchmarks.op_aco.dataset import get_split_spec as op_split
from benchmarks.tasks import CLASSES, HELDOUT_TIMEOUT, FIXED_TASKS, SCALES
from traceaad.common.history import final_attempt
from traceaad.common.prompts import ANALYSIS as SHARED_ANALYSIS, ContextTooLong, KNOWN
from traceaad.v10_17.prompts import PromptBuilder as V1017Prompts

DEEPEN = f"""[Your Task: Deepen]
Keep the current algorithm's decision rule and use it to guide a search that spends more of the time limit on finding better decisions, so that the program scores better than the current algorithm. {KNOWN}"""

ANALYSIS = {**SHARED_ANALYSIS,
            "Deepen": ("which decisions of the current algorithm a search guided by it could improve, and how much "
                       "computation that search can use within the time limit on the training and on the larger "
                       "test instances")}

CALLED_ONCE = "The function is called once per instance, before the ant colony starts."
MANY_TIMES = re.compile(r"Use efficient NumPy operations because the function is evaluated many\s+times\.")
SERVICE = ("At a customer, the vehicle waits until the time window opens if it arrives early, then spends the "
           "customer's service duration before moving on. The current time passed to the function already includes "
           "the waiting and service durations of the customers served so far; the service durations themselves are "
           "not passed to the function.")

UNITS = {"tsp_construct": "nodes", "cvrp_aco": "customers", "op_aco": "nodes", "vrptw_construct": "customers"}
ACO_SPLITS = {"cvrp_aco": cvrp_split, "op_aco": op_split}


def _join(values):
    values = [str(v) for v in values]
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def _count(n):
    return f"{n:,}"


def corrected_description(task, description):
    """The task text with the facts about computation corrected."""
    if task in ACO_SPLITS:
        description, replaced = MANY_TIMES.subn(CALLED_ONCE, description)
        if replaced != 1 and CALLED_ONCE not in description:
            raise ValueError(f"{task}: the task text no longer has the sentence V10.20 corrects")
    if task == "vrptw_construct":
        description += "\n\n" + SERVICE
    return description


def training_instances(task, evaluation):
    """The training set of the evaluation actually used, in words."""
    if task in FIXED_TASKS:
        return evaluation.instance_description
    if task == "online_bin_packing":
        specs = get_generated_task_kwargs(task, "train")["dataset_specs"]
        count = sum(s["n_instances"] * len(s["capacities"]) for s in specs)
        items = _join(_count(s["n_items"]) for s in specs)
        capacities = _join(sorted({c for s in specs for c in s["capacities"]}))
        return f"{count} training instances: {items} items, each with bin capacities {capacities}"
    return f"{evaluation.n_instance} training instances with {evaluation.problem_size} {UNITS[task]}"


def test_sets(task):
    """The held-out sets the final program is tested on, in words."""
    if task in FIXED_TASKS:
        data = CLASSES[task].DATASET
        return data.describe('test')
    if task == "online_bin_packing":
        specs = get_generated_task_kwargs(task, "eval")["dataset_specs"]
        counts = {s["n_instances"] for s in specs}
        capacities = _join(sorted({c for s in specs for c in s["capacities"]}))
        if len(counts) != 1:
            raise ValueError("bin packing test sets differ in size")
        return (f"{counts.pop()} instances each of {_join(_count(s['n_items']) for s in specs)} items "
                f"with bin capacities {capacities}")
    if task in ACO_SPLITS:
        counts = {ACO_SPLITS[task](f"test_{scale}").n_instances for scale in SCALES[task]}
    else:
        counts = {get_generated_task_kwargs(task, "eval")["n_instance"]}
    if len(counts) != 1:
        raise ValueError(f"{task} test sets differ in size")
    return f"{counts.pop()} instances each with {_join(SCALES[task])} {UNITS[task]}"


class PromptBuilder(V1017Prompts):
    ACTIONS = ("Refine", "Explore", "Crossover", "Deepen")
    ANALYSIS = ANALYSIS

    def __init__(self, llm, task, evaluation, programs, attempts, config):
        if task in FIXED_TASKS:
            self.ANALYSIS = {**type(self).ANALYSIS,
                            'Deepen': 'which decisions of the current algorithm a search guided by it could improve, and how much computation that search can use within the time limit on the training and independent same-scale test instances'}
        self.facts = task in HELDOUT_TIMEOUT
        self.training_set = training_instances(task, evaluation) if self.facts else None
        super().__init__(llm, task, evaluation, programs, attempts, config)

    def task_text(self, description):
        return corrected_description(self.task, description) if self.facts else description

    def evaluation_text(self, meaning, higher):
        if not self.facts:
            return super().evaluation_text(meaning, higher)
        return (f"[Evaluation]\nEach program is evaluated on {self.training_set}.\n"
                f"Score: {meaning}. {'Higher' if higher else 'Lower'} is better.\n"
                f"The whole evaluation must finish within {self._limit()} seconds.\n"
                f"After the search, the final program is tested on {test_sets(self.task)}; "
                f"each test set must finish within {format(HELDOUT_TIMEOUT[self.task], 'g')} seconds.")

    def build(self, action, parent, *, reference=None):
        if action == "Deepen":
            return self.deepen(parent)
        return super().build(action, parent, reference=reference)

    def deepen(self, parent):
        """The current algorithm, its measured cost and the earlier Deepen attempts from it."""
        tried = [a for a in self.attempts_from(parent) if a["action"] == "Deepen"]
        shown, trims = min(self.config.attempts_shown, len(tried)), []
        while True:
            section, ids = self._deepen_attempts(parent, tried, shown)
            sections = self.common + [self._current(parent), section, DEEPEN, self.output_format("Deepen")]
            result = self._result(sections, "Deepen", trims=trims)
            result["attempt_ids"] = ids
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            if shown == 0:
                raise ContextTooLong("Deepen prompt exceeds context")
            shown -= 1
            trims.append("oldest_attempt")

    def _deepen_attempts(self, parent, tried, shown):
        if not tried or shown == 0:
            return "", []
        listed = tried[-shown:]
        improved = sum(self._improved(final_attempt(a, self.attempts), parent) for a in tried)
        header = (f"[Earlier Deepen Attempts From the Current Algorithm]\n{len(tried)} earlier Deepen "
                  f"attempt{'' if len(tried) == 1 else 's'} started from the current algorithm; {improved} produced "
                  "a new algorithm scoring better than it. "
                  + ("All are listed" if len(listed) == len(tried) else f"The most recent {len(listed)} are listed")
                  + ", oldest first, with their evaluated outcomes.")
        lines = self._attempt_lines(parent, listed, {parent["id"]})
        return header + "\n\n" + "\n\n".join(lines), [a["id"] for a in listed]

    def _limit(self):
        return format(self.timeout, "g") if self.timeout is not None else "an unspecified number of"

    def evaluation_time(self, seconds):
        text = super().evaluation_time(seconds)
        return text if self.timeout is None else f"{text} of the {format(self.timeout, 'g')} s limit"

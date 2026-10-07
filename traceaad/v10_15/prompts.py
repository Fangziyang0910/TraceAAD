"""V10.15's compact English prompts and deterministic context trimming."""

from traceaad.common.history import code_diff, path, score_text, verdict
from traceaad.common.prompts import ContextTooLong, PromptBuilder as MeasuredPrompts


SCORES = {
    "tsp_construct": ("the average length of the constructed tours", False),
    "cvrp_aco": ("the average total length of the best routes found by the ant colony", False),
    "op_aco": ("the average total prize of the best tours found by the ant colony", True),
    "online_bin_packing": ("the average number of bins used", False),
    "vrptw_construct": ("the average total travel distance of the constructed routes", False),
}

# Every operator asks for the same thing: a better algorithm, written from the
# information shown. Operators differ only in which information they build on
# and how far the result may move from the current algorithm.
INITIAL = """[Your Task: Design an Algorithm]
Write an algorithm for this task that scores as well as possible, built on a clear core idea."""

ANOTHER_INITIAL = """[Your Task: Design a Different Algorithm]
Write an algorithm built on a core idea different from those of the algorithms above that scores better than all of them."""

# Every operator aims past everything the prompt shows, not just past the
# current algorithm: an earlier version is then never an answer, whichever way
# the latest step went.
REFINE = """[Your Task: Refine]
Continue this line of development: write an algorithm that scores better than every version shown above."""

REFINE_ROOT = """[Your Task: Refine]
Continue developing the current algorithm: write a version that scores better than it."""

# Explore changes how the current algorithm decides and aims past the search
# best; it does not start over. A fixed-parent paired test on TSP
# (docs/03-现象与检验/2026-10-01-生成目标与计算限制.md §4) found that
# "write a new algorithm with a different core idea" never beat the run's best.
EXPLORE = """[Your Task: Explore]
Write an algorithm that scores better than the best found so far by changing how the current algorithm makes its decisions, not by tuning it."""

EXPLORE_CARDS = EXPLORE + "\nThe reference designs show other approaches found in this search; draw on them as inspiration."

CROSSOVER = """[Your Task: Crossover]
Combine the two lines of development: write an algorithm that scores better than every version shown above, bringing into the current algorithm what the reference algorithm does well."""

REPAIR = """[Your Task: Repair]
The program failed during evaluation. Fix it so that it runs correctly within the time limit, keeping the algorithm it was meant to implement."""

FORMATION_INTRO = ("The steps that produced the current algorithm, oldest first. Every version shown has been evaluated: "
                   "each step gives the score change, the Design of the version it produced, and the code diff "
                   "from the previous version.")

IDEA_DISPLAY_CHARS = 2400  # about 400 words: room for a full description, guard against run-ons


def idea_view(idea):
    return " ".join((idea or "").split())[:IDEA_DISPLAY_CHARS]


# What an operator's Analysis decides before any code is written. Operators
# that modify a given algorithm risk reproducing a known program, and a brief
# targeted diagnosis prevents that; operators asked for a new algorithm
# already carry that pressure, and an Analysis only added cost there.
ANALYSIS = {
    "Refine": "what limits this line of development so far, and what change should take it past every version",
    "Crossover": "what the reference algorithm does well that the current algorithm lacks, and how to combine them",
    "Repair": "what caused the failure, and how to fix it",
    "Explore": "what the current algorithm cannot capture, and what different computation should capture it",
    "Init": None,
}


def output_format(action):
    # Paired Qwen tests (docs/03-现象与检验/2026-10-01-写代码前的决策与Design格式.md):
    # on Refine a few-sentence targeted Analysis cut duplicates from 25% to 5%
    # at the cost of a 170-word Design; on Crossover it gave the best rank; on
    # Explore a Design alone ranked best at under half the tokens (on random
    # parents and by rank; the structural Explore of the diagnosis study adds an
    # Analysis because it asks for a rewrite of the current program). Longer or
    # free-form analysis and native thinking did worse, and the kept Design's
    # length had no measurable effect, so it stays at one or two sentences.
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


class PromptBuilder:
    failure = MeasuredPrompts.failure
    _seconds = staticmethod(MeasuredPrompts._seconds)
    _calls = MeasuredPrompts._calls

    def __init__(self, llm, task, evaluation, programs, attempts, config):
        self.llm, self.task, self.evaluation = llm, task, evaluation
        self.timeout = evaluation.timeout_seconds
        self.programs, self.attempts, self.config = programs, attempts, config
        if task in SCORES:
            meaning, higher = SCORES[task]
        else:
            meaning, higher = "the task objective", False
        self.higher_is_better = higher
        description = evaluation.task_description.strip()
        notes = getattr(evaluation, "design_notes", "")
        if notes:
            description += "\n\n" + notes.strip()
        timeout = (format(evaluation.timeout_seconds, "g") if evaluation.timeout_seconds is not None
                   else "an unspecified number of")
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

    def count(self, text):
        return self.llm.count_prompt_tokens(text)

    def block_count(self, text):
        return self.llm.count_tokens(text)

    def _render(self, sections):
        return "\n\n".join(section for section in sections if section)

    def _result(self, sections, action, edge_ids=(), trims=()):
        prompt = self._render(sections)
        size = self.count(prompt)
        return {"prompt": prompt, "input_tokens": size, "action": action,
                "history_edge_ids": list(edge_ids), "trims": list(trims),
                "reference_history_edge_ids": [], "explore_reference_ids": [],
                "token_count_mode": getattr(self.llm, "prompt_token_count_mode", "serving_tokenizer")}

    @staticmethod
    def _measured(node):
        # Evaluation time is a property of each shown program, like its score,
        # so the model can judge how much more computation fits the limit.
        seconds = node.get("eval_seconds")
        text = f"Score: {score_text(node['score'])}"
        return text + (f" · Evaluation time: about {max(seconds, 0.1):.1f} s" if seconds else "")

    def _current(self, node):
        return f"[Current Algorithm]\n{self._measured(node)}\n```python\n{node['code'].rstrip()}\n```"

    def _edge(self, parent, child, index, *, latest=False, subject="current"):
        event = child
        first = self.programs.get(child["parent_id"])
        if child.get("repaired") and first is not None and not first.get("valid", True):
            event = first
        action = event["action"]
        if action == "Crossover" and event.get("reference_id") is not None:
            reference = self.programs[event["reference_id"]]
            action += f" with an algorithm scoring {score_text(reference['score'])}"
        heading = f"Step {index}"
        if latest:
            heading += f" (latest: produced the {subject} algorithm)"
        heading += (f" · {action} · score {score_text(parent['score'])} → "
                    f"{score_text(child['score'])} ({verdict(parent['score'], child['score'], self.higher_is_better)})")
        result = heading + f"\n  Design: {idea_view(child['idea'])}"
        result += f"\n  Code diff (previous → current):\n```diff\n{code_diff(parent['code'], child['code'])}\n```"
        return result

    def _formation(self, sequence, count, *, title="How the Current Algorithm Was Formed", subject="current"):
        root = sequence[0]
        if len(sequence) == 1:
            return (f"[{title}]\nThe {subject} algorithm is an initial design; no changes have been recorded yet.\n"
                    f"Design: {idea_view(root['idea'])}"), []
        start = max(1, len(sequence) - count)
        lines = [f"[{title}]", FORMATION_INTRO.replace("current algorithm", f"{subject} algorithm")]
        if start == 1:
            lines.append(f"Start · initial algorithm · score {score_text(root['score'])}\n  Design: {idea_view(root['idea'])}")
        else:
            lines.append(f"The path has {len(sequence)-1} steps; showing the most recent {len(sequence)-start} steps.")
        for index in range(start, len(sequence)):
            lines.append(self._edge(sequence[index-1], sequence[index], index,
                                    latest=index == len(sequence)-1, subject=subject))
        return lines[0] + "\n" + lines[1] + "\n\n" + "\n\n".join(lines[2:]), [n["id"] for n in sequence[start:]]

    def initial(self, roots):
        sections = list(self.common)
        trims = []
        shown = list(roots) if len(roots) >= 4 else []

        def root_section():
            entries = [f"Algorithm {i} · {self._measured(n)} · Design: {idea_view(n['idea'])}\n"
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

    def _reference_ideas(self, references, best_score):
        sections = []
        if references:
            lines = ["[Reference Designs from This Search]",
                     "Other evaluated algorithms, each with its score and Design."]
            lines.extend(f"Reference {i} · Score {score_text(n['score'])} · Design: {idea_view(n['idea'])}"
                         for i, n in enumerate(references, 1))
            sections.append("\n".join(lines))
        sections.append(f"[Search Best]\nBest score found so far in this search: {score_text(best_score)}.")
        return self._render(sections)

    def build(self, action, parent, *, reference=None, references=(), best_score=None):
        if action not in {"Refine", "Explore", "Crossover"}:
            raise ValueError(action)
        sequence = [p for p in path(parent, self.programs) if p.get("valid", True)]
        trims = []
        current = self._current(parent)
        if action == "Refine":
            count = min(self.config.history_depth, len(sequence)-1)
            while True:
                history, ids = self._formation(sequence, count)
                sections = self.common + [current, history, REFINE_ROOT if len(sequence) == 1 else REFINE,
                                                   output_format("Refine")]
                result = self._result(sections, "Refine", ids, trims)
                if result["input_tokens"] <= self.config.max_input_tokens:
                    return result
                if count > 1:
                    count -= 1
                    trims.append("oldest_history")
                else:
                    raise ContextTooLong("Refine prompt with required latest history exceeds context")
        if action == "Explore":
            shown = list(references[:self.config.explore_cards])
            while True:
                ideas = self._reference_ideas(shown, best_score)
                sections = self.common + [current, ideas, EXPLORE_CARDS if shown else EXPLORE,
                                                   output_format("Explore")]
                result = self._result(sections, "Explore", trims=trims)
                result["explore_reference_ids"] = [n["id"] for n in shown]
                if result["input_tokens"] <= self.config.max_input_tokens:
                    return result
                if not shown:
                    raise ContextTooLong("minimum Explore prompt exceeds context")
                trims.append(f"explore_reference:{shown.pop()['id']}")
        if reference is None:
            raise ValueError("Crossover requires a reference")
        ref_section = (f"[Reference Algorithm]\nAnother evaluated algorithm from this search.\n"
                       f"{self._measured(reference)}\nDesign: {idea_view(reference['idea'])}\n"
                       f"```python\n{reference['code'].rstrip()}\n```")
        reference_sequence = [p for p in path(reference, self.programs) if p.get("valid", True)]
        counts = [min(4, len(sequence)-1), min(4, len(reference_sequence)-1)]
        while True:
            recent, ids = self._formation(sequence, counts[0])
            ref_history, ref_ids = self._formation(reference_sequence, counts[1],
                                                  title="How the Reference Algorithm Was Formed", subject="reference")
            sections = self.common + [current, recent, ref_section, ref_history, CROSSOVER,
                                               output_format("Crossover")]
            result = self._result(sections, "Crossover", ids, trims)
            result["reference_history_edge_ids"] = ref_ids
            if result["input_tokens"] <= self.config.max_input_tokens:
                return result
            candidates = [i for i, count in enumerate(counts) if count > 1]
            if not candidates:
                break
            side = max(candidates, key=lambda i: self.block_count([recent, ref_history][i]))
            counts[side] -= 1
            trims.append("oldest_history" if side == 0 else "oldest_reference_history")
        fallback = self.build("Refine", parent)
        fallback["trims"] = trims + ["crossover_context_fallback"] + fallback["trims"]
        return fallback

    def repair(self, program):
        failed_code, idea = program["code"], program["idea"]
        error_text = MeasuredPrompts.error_text(self, program)
        failed = f"[Failed Program]\nDesign: {idea_view(idea)}\n```python\n{failed_code.rstrip()}\n```"
        sections = self.common + [failed, f"[Error]\n{error_text}", REPAIR, output_format("Repair")]
        result = self._result(sections, "Repair")
        if result["input_tokens"] > self.config.max_input_tokens:
            raise ContextTooLong("repair prompt exceeds context")
        return result

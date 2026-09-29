"""V10.15's compact English prompts and deterministic context trimming."""

from .history import change_summary, code_diff, path, score_text, verdict


SCORES = {
    "tsp_construct": ("the average length of the constructed tours", False),
    "cvrp_aco": ("the average total length of the best routes found by the ant colony", False),
    "op_aco": ("the average total prize of the best tours found by the ant colony", True),
    "online_bin_packing": ("the average number of bins used", False),
    "vrptw_construct": ("the average total travel distance of the constructed routes", False),
}

INITIAL = """[Your Task: Design an Initial Algorithm]
Design one complete, competitive algorithm for this task.
- Base it on a clear decision principle and implement that principle carefully.
- Use what the inputs make available. The function may compute more than a single formula, for example derive intermediate quantities or examine the consequences of a choice, as long as the evaluation stays within the time limit.
- Do not return a placeholder or a trivial baseline."""

ANOTHER_INITIAL = """[Your Task: Design Another Initial Algorithm]
Design one complete, competitive algorithm whose core decision principle differs from every algorithm above.
- Notice what the algorithms above have in common, and build yours on a different principle or on information they do not use.
- You may reuse a helpful detail, but the main idea must be different.
- Do not return a placeholder or a trivial baseline."""

REFINE = """[Your Task: Refine]
Improve the current algorithm with one focused change.
- Build on what the history shows is working: parts introduced by improving steps are probably doing useful work, so keep them unless your change needs to alter them.
- Let the recent steps guide the next one: push further in a direction that improved the score, or correct or undo a recent change that made it worse.
- The change must be able to alter the decisions the function makes. Rescaling all scores, or applying the same monotone transform to them, leaves the chosen option unchanged.
- If the structure is sound, recalibrating a few influential parameters is a valid focused change.
- Prefer replacing or simplifying logic over stacking new layers, and leave unrelated parts of the program unchanged."""

REFINE_ROOT = """[Your Task: Refine]
Improve the current algorithm with one focused change.
- Identify the part of the algorithm that most limits the quality of its decisions, and improve that part.
- The change must be able to alter the decisions the function makes. Rescaling all scores, or applying the same monotone transform to them, leaves the chosen option unchanged.
- If the structure is sound, recalibrating a few influential parameters is a valid focused change.
- Prefer replacing or simplifying logic over stacking new layers, and leave unrelated parts of the program unchanged."""

EXPLORE = """[Your Task: Explore]
Find a materially different way to solve this task better than the current algorithm.
- First identify the main limitation of the current approach: information it ignores, decisions it systematically gets wrong, or situations it cannot represent.
- Then change the core of the algorithm to remove that limitation: what it computes from the inputs, how it evaluates a choice before committing to it, or how it turns signals into a decision. Tuning parameters or making a small local edit is not enough.
- You may keep useful parts of the current program or start from scratch, but do not simply restate an idea listed above.
- The new algorithm must be complete and competitive on its own, and must stay within the time limit."""

CROSSOVER = """[Your Task: Crossover]
Improve the current algorithm by transplanting one mechanism from the reference algorithm.
- Compare the two programs and find one computation in the reference that the current algorithm lacks and that addresses one of its weaknesses, for example an additional signal, a feasibility or look-ahead check, or a different way of combining terms.
- Integrate that mechanism into the current algorithm and adapt it so that it works with the existing parts. Keep the current algorithm's framework and its working components.
- The reference may score lower overall and still contain a useful mechanism.
- Do not copy the reference or return a program that is essentially one of the two inputs. The transplanted mechanism must be able to change the current algorithm's decisions."""

REPAIR = """[Your Task: Repair]
Fix the program so that it runs correctly, while keeping its intended design.
- Change only what is needed to remove the failure.
- If the evaluation timed out, reduce the cost of the most expensive computation instead of dropping the idea.
- If the output was invalid, make sure the function returns exactly what the target function's contract requires."""

FORMATION_INTRO = ('These are the most recent steps on the path that produced the current algorithm, oldest first. '
                   '"Change" is computed from the code; "Idea" is what was intended at the time and may not match the code exactly. Scores are measured.')


class ContextTooLong(ValueError):
    pass


def idea_view(idea):
    return " ".join((idea or "").split())[:300]


def output_format(what):
    return ("[Output Format]\nReply with exactly one Idea line followed by one Python code block:\n"
            f"Idea: <one sentence, at most 300 characters, describing {what}>\n"
            "Code:\n```python\n<the complete program>\n```\n"
            "Write no comments or docstrings in the code, and nothing after the code block.")


class PromptBuilder:
    def __init__(self, llm, task, evaluation, archive, config):
        self.llm, self.task, self.evaluation = llm, task, evaluation
        self.archive, self.config = archive, config
        if task in SCORES:
            meaning, higher = SCORES[task]
        else:
            meaning, higher = "the task fitness", True
        self.higher_is_better = higher
        description = evaluation.task_description.strip()
        notes = getattr(evaluation, "design_notes", "")
        if notes:
            description += "\n\n" + notes.strip()
        timeout = (format(evaluation.timeout_seconds, "g") if evaluation.timeout_seconds is not None
                   else "an unspecified number of")
        self.common = [
            "[Task]\n" + description,
            "[Evaluation]\nEach candidate program is run on a fixed set of training instances.\n"
            f"Score: {meaning}. {'Higher' if higher else 'Lower'} is better.\n"
            f"The whole evaluation must finish within {timeout} seconds, so keep the computation efficient.",
            "[Target Function]\n```python\n" + str(evaluation.template_program).strip() +
            "\n```\nKeep the function name, arguments and return contract exactly as shown. "
            "The program must be self-contained: include every import, constant and helper it uses.",
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
                "token_count_mode": getattr(self.llm, "prompt_token_count_mode", "serving_tokenizer")}

    def _current(self, node):
        return f"[Current Algorithm]\nScore: {score_text(node['score'])}\n```python\n{node['code'].rstrip()}\n```"

    def _edge(self, parent, child, index, *, latest=False, use_diff=False):
        action = child["action"]
        if action == "Crossover" and child.get("reference_id") is not None:
            reference = self.archive[child["reference_id"]]
            action += f" with an algorithm scoring {score_text(reference['score'])}"
        heading = f"Step {index}"
        if latest:
            heading += " (latest: produced the current algorithm)"
        heading += (f" · {action} · score {score_text(parent['score'])} → "
                    f"{score_text(child['score'])} ({verdict(parent['score'], child['score'], self.higher_is_better)})")
        result = heading + f"\n  Idea: {idea_view(child['idea'])}"
        if use_diff:
            result += f"\n  Code diff (previous → current):\n```diff\n{code_diff(parent['code'], child['code'])}\n```"
        else:
            result += f"\n  Change: {change_summary(parent['code'], child['code'])}"
        return result

    def _formation(self, sequence, count, *, diff=True, title="How the Current Algorithm Was Formed"):
        root = sequence[0]
        if len(sequence) == 1:
            return (f"[{title}]\nThe current algorithm is an initial design; no changes have been recorded yet.\n"
                    f"Idea: {idea_view(root['idea'])}"), []
        start = max(1, len(sequence) - count)
        lines = [f"[{title}]", FORMATION_INTRO]
        if start == 1:
            lines.append(f"Start · initial algorithm · score {score_text(root['score'])}\n  Idea: {idea_view(root['idea'])}")
        else:
            lines.append(f"The path has {len(sequence)-1} steps; showing the most recent {len(sequence)-start} steps.")
        for index in range(start, len(sequence)):
            lines.append(self._edge(sequence[index-1], sequence[index], index,
                                    latest=index == len(sequence)-1,
                                    use_diff=diff and index == len(sequence)-1))
        return lines[0] + "\n" + lines[1] + "\n\n" + "\n\n".join(lines[2:]), [n["id"] for n in sequence[start:]]

    def initial(self, roots):
        sections = list(self.common)
        trims = []
        shown = list(roots) if len(roots) >= 4 else []

        def root_section():
            entries = [f"Algorithm {i} · Score {score_text(n['score'])} · Idea: {idea_view(n['idea'])}\n"
                       f"```python\n{n['code'].rstrip()}\n```" for i, n in enumerate(shown, 1)]
            return "[Algorithms Designed So Far]\n" + "\n\n".join(entries)

        if shown:
            while len(shown) > 1 and self.block_count(root_section()) > self.config.root_tokens:
                trims.append(f"root:{shown.pop(0)['id']}")
            sections.append(root_section())
        sections.extend([ANOTHER_INITIAL if shown else INITIAL, output_format("the algorithm")])
        while shown and len(shown) > 1 and self.count(self._render(sections)) > self.config.max_input_tokens:
            trims.append(f"root:{shown.pop(0)['id']}")
            sections[3] = root_section()
        result = self._result(sections, "Init", trims=trims)
        if result["input_tokens"] > self.config.max_input_tokens or (shown and self.block_count(root_section()) > self.config.root_tokens):
            raise ContextTooLong("initial prompt exceeds context with one required root")
        return result

    def _ideas(self, sequence, count, best_score):
        if len(sequence) == 1:
            lines = ["The current algorithm is an initial design.", f"Idea: {idea_view(sequence[0]['idea'])}"]
        else:
            start = max(1, len(sequence) - count)
            lines = ["Oldest first. Scores are measured; each idea is the intent stated when that version was written."]
            if start == 1:
                root = sequence[0]
                lines.append(f"Start · score {score_text(root['score'])} · Idea: {idea_view(root['idea'])}")
            for index in range(start, len(sequence)):
                parent, child = sequence[index-1:index+1]
                lines.append(f"Step {index} · {child['action']} · score {score_text(parent['score'])} → "
                             f"{score_text(child['score'])} ({verdict(parent['score'], child['score'], self.higher_is_better)}) "
                             f"· Idea: {idea_view(child['idea'])}")
        lines.append(f"Best score found so far in this search: {score_text(best_score)}.")
        return "[Earlier Ideas on This Line of Development]\n" + "\n".join(lines)

    def build(self, action, parent, *, reference=None, best_score=None):
        if action not in {"Refine", "Explore", "Crossover"}:
            raise ValueError(action)
        sequence = path(parent, self.archive)
        trims = []
        current = self._current(parent)
        if action == "Refine":
            count, diff = min(8, len(sequence)-1), True
            while True:
                history, ids = self._formation(sequence, count, diff=diff)
                sections = self.common + [current, history, REFINE_ROOT if len(sequence) == 1 else REFINE,
                                          output_format("the change you made")]
                result = self._result(sections, "Refine", ids, trims)
                if (result["input_tokens"] <= self.config.max_input_tokens and
                        self.block_count(history) <= self.config.history_tokens):
                    return result
                if count > 1:
                    count -= 1
                    trims.append("oldest_history")
                elif diff and count:
                    diff = False
                    trims.append("latest_diff_to_summary")
                else:
                    raise ContextTooLong("minimum Refine prompt exceeds context")
        if action == "Explore":
            count = min(8, len(sequence)-1)
            while True:
                ideas = self._ideas(sequence, count, best_score)
                sections = self.common + [current, ideas, EXPLORE, output_format("the new algorithm")]
                result = self._result(sections, "Explore", [n["id"] for n in sequence[max(1, len(sequence)-count):]], trims)
                if result["input_tokens"] <= self.config.max_input_tokens:
                    return result
                if count <= 1:
                    raise ContextTooLong("minimum Explore prompt exceeds context")
                count -= 1
                trims.append("oldest_idea")
        if reference is None:
            raise ValueError("Crossover requires a reference")
        recent, ids = self._formation(sequence, min(3, len(sequence)-1), diff=False,
                                      title="Recent Changes to the Current Algorithm")
        ref_section = (f"[Reference Algorithm]\nA different algorithm from another branch of this search.\n"
                       f"Score: {score_text(reference['score'])}\nIdea: {idea_view(reference['idea'])}\n"
                       f"```python\n{reference['code'].rstrip()}\n```")
        sections = self.common + [current, recent, ref_section, CROSSOVER,
                                  output_format("the mechanism you transplanted and how it is integrated")]
        result = self._result(sections, "Crossover", ids, trims)
        if result["input_tokens"] <= self.config.max_input_tokens:
            return result
        sections.remove(recent)
        trims.append("recent_changes")
        result = self._result(sections, "Crossover", (), trims)
        if result["input_tokens"] <= self.config.max_input_tokens:
            return result
        fallback = self.build("Refine", parent)
        fallback["trims"] = trims + ["crossover_context_fallback"] + fallback["trims"]
        return fallback

    def repair(self, failed_code, idea, error_text, *, parent=None):
        if parent is None:
            intro = "This program was written as an initial algorithm, but it failed during evaluation."
        else:
            intro = f"This program was written to improve an algorithm with score {score_text(parent['score'])}, but it failed during evaluation."
        failed = f"[Failed Program]\n{intro}\nIdea: {idea_view(idea)}\n```python\n{failed_code.rstrip()}\n```"
        sections = self.common + [failed, f"[Error]\n{error_text}", REPAIR, output_format("the fix")]
        result = self._result(sections, "Repair")
        if result["input_tokens"] > self.config.max_input_tokens:
            raise ContextTooLong("repair prompt exceeds context")
        return result

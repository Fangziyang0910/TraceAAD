"""V14 prompts: the base prompts plus a design map, a development note and a revisit note."""

from traceaad.history import score_text
from traceaad.prompts import REFINE, PromptBuilder, idea_view

SEARCH_ACTIONS = ("Refine", "Explore", "Crossover", "Mature")
MAP_IDEA_CHARS = 900

DIVERGE = "[Your Task: Design a Different Algorithm]\nThe designs above have already been developed in this search. Write a new algorithm whose main decision rule differs from every one of them: decide the output with a different kind of computation, not by reweighting, extending or combining the listed designs. Write it from scratch and aim for good solution quality."
DIVERGE_ANALYSIS = "which kinds of computation the listed designs rely on, which different computation your algorithm uses to decide its output, and why it can reach good solution quality"
DEVELOPMENT = "[Development Period]\nThe current algorithm is a new design written from scratch; it has not been tuned yet. A few attempts are reserved for developing it before it competes with the established algorithms of this search, whose best score is {best}. Keep its main decision rule. Make it stronger: fix weak steps, choose better parameter values and add the supporting computations it needs."
REVISIT = "[Note]\nA previous reply to this same request returned code identical to {what}. The change it described did not appear in the code. Return a program that actually implements a change: its code must differ from every program already evaluated in this search."


def insert_before_output(prompt, section):
    """Place a section directly before the output format, after the task instruction."""
    marker = "[Output Format]"
    at = prompt.rindex(marker)
    return prompt[:at] + section + "\n\n" + prompt[at:]


class PromptBuilderV14(PromptBuilder):
    def children_from(self, node):
        return [
            p
            for p in sorted(self.programs.values(), key=lambda p: p.id)
            if p.parent_id == node.id and p.repair_of is None and p.action in SEARCH_ACTIONS
        ]

    def failure(self, program):
        if program.failure["kind"] == "revisit":
            return "returned code identical to an existing program"
        return super().failure(program)

    def diverge(self, designs, tried, best):
        """A new algorithm from a map of the designs developed so far; no code is shown."""
        header = f"[Designs Developed So Far]\nDesigns of distinct algorithms already evaluated in this search, as described when each was generated. Code is not shown. The best score found so far is {score_text(best)}; lower is better."
        lines = [f"{i}. {idea_view(p.idea)[:MAP_IDEA_CHARS]}" for i, p in enumerate(designs, 1)]
        sections = self.common + [self.records, header + "\n\n" + "\n\n".join(lines)]
        if tried:
            entries = [
                f"- {idea_view(p.idea)[:MAP_IDEA_CHARS]}\n  Best score after development: {score_text(score) if score is not None else 'no valid program'}"
                for p, score in tried
            ]
            sections.append(
                "[Different Designs Tried Recently]\nNew designs started earlier in this search after progress stalled.\n\n"
                + "\n\n".join(entries)
            )
        sections += [DIVERGE, self._format(DIVERGE_ANALYSIS)]
        result = self._result(sections, "Diverge")
        result["map_ids"] = [p.id for p in designs]
        result["tried_ids"] = [p.id for p, _ in tried]
        return result

    def mature(self, node, best):
        result = self.build("Refine", node)
        note = DEVELOPMENT.format(best=score_text(best))
        result["prompt"] = result["prompt"].replace(REFINE, note + "\n\n" + REFINE, 1)
        result["input_tokens"] = self.llm.count_prompt_tokens(result["prompt"])
        result["action"] = "Mature"
        return result

    def revisit(self, request, what):
        """The original request once more, told that its reply reproduced existing code."""
        result = dict(request)
        result["prompt"] = insert_before_output(request["prompt"], REVISIT.format(what=what))
        result["input_tokens"] = self.llm.count_prompt_tokens(result["prompt"])
        return result

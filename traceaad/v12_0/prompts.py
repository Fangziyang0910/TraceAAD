"""V12.0 prompts: V10.21's prompts, plus the match step and the implementation step of Explore.

The proposal step is V10.21's Explore prompt unchanged; generation stops before the code.
The implementation step keeps that prompt's material and goal, then adds the proposal,
the earlier program (when linked) and the request for code. Nothing is said about the
earlier program beyond facts: it was evaluated in this search and shares the computation.
"""

from traceaad.common.prompts import idea_view
from traceaad.v10_21.prompts import PromptBuilder as V1021Prompts

MATCH = """Each design below describes an algorithm for the same task.

[Design A]
{design}

[Earlier Designs in This Search]
{listed}

Which earlier designs are built on the same main computation as Design A?
End your reply with one line: "Same:" followed by their numbers separated by commas, or "Same: none"."""

PROPOSAL = "[Your Proposal]\n{proposal}"

EARLIER = """[An Earlier Program for the Same Computation]
An evaluated program from this search, built on the same main computation as your Design.
{measured}
Design: {design}
```python
{code}
```"""

IMPLEMENT = """[Your Task: Implement]
Implement your Design in the current algorithm."""


def match_prompt(design, ranked):
    """The match request for a proposal's Design and the ranked earlier programs (also used offline)."""
    listed = "\n".join(f"[{program['id']}] {idea_view(program['idea'])}" for program, _ in ranked)
    return MATCH.format(design=idea_view(design), listed=listed)


class PromptBuilder(V1021Prompts):
    PROPOSAL = PROPOSAL
    EARLIER = EARLIER
    IMPLEMENT = IMPLEMENT

    def match(self, design, ranked):
        prompt = match_prompt(design, ranked)
        return prompt, self.count(prompt)

    def implement_format(self):
        return ("[Output Format]\nReply with one Python code block:\n"
                "Code:\n```python\n<the complete program>\n```\n" + self.CLOSING)

    def implement(self, explore_prompt, parent, proposal, earlier):
        """The implementation request; drops the earlier program, then the Explore material, if too long."""
        head = explore_prompt[:explore_prompt.rindex("[Output Format]")].rstrip()
        proposal_block = self.PROPOSAL.format(proposal=proposal) if proposal else ""
        earlier_block = (self.EARLIER.format(measured=self.measured(earlier), design=idea_view(earlier["idea"]),
                                             code=earlier["code"].rstrip()) if earlier is not None else "")
        tail = [self.IMPLEMENT, self.implement_format()]
        trims = []
        prompt = self._render([head, proposal_block, earlier_block, *tail])
        tokens = self.count(prompt)
        if tokens > self.config.max_input_tokens and earlier_block:
            trims.append("earlier_program")
            prompt = self._render([head, proposal_block, *tail])
            tokens = self.count(prompt)
        if tokens > self.config.max_input_tokens:
            trims.append("explore_material")
            prompt = self._render(self.common + [self._current(parent), proposal_block, *tail])
            tokens = self.count(prompt)
        return {"prompt": prompt, "input_tokens": tokens, "trims": trims}

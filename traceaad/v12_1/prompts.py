"""V12.0 prompts with design verification: the verify and Complete requests, and what readings add to Designs."""

from traceaad.common.history import final_attempt
from traceaad.common.prompts import ContextTooLong, idea_view
from traceaad.v12_0.prompts import PromptBuilder as LinkedPrompts
from .verification import contain_prompt, verify_prompt

WRITTEN_FOR = "[Proposal]\nThe current algorithm was written for this proposal.\n{proposal}"
MADE_FOR = "[Proposal]\nThis proposal was made for the current algorithm.\n{proposal}"

EARLIER_VERSION = """[An Earlier Program for This Proposal]
Written from the current algorithm for this proposal.
{measured}
```python
{code}
```"""

READING = "[Reading of the Code]\nRead against the Design: {missing}"
EARLIER_READING = ("[Reading of the Earlier Program]\nRead against the Design and against the current algorithm, "
                   "from which it was written:\n{sentences}")

COMPLETE = """[Your Task: Complete]
Change the current algorithm so that it carries out the Design, and keep the rest of what it computes."""

COMPLETE_ANALYSIS = ("what the current algorithm computes, and what it must compute to carry out the Design. "
                     "It will not be shown again")


class PromptBuilder(LinkedPrompts):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The search replaces this with its saved verdicts, keyed by program id.
        self.verdicts = {}

    def _reading(self, program_id):
        """What the verdict found: the part of the Design not carried out, and the computation of the start
        algorithm dropped."""
        verdict = self.verdicts.get(program_id) or {}
        return (verdict.get("missing") if verdict.get("implemented") is False else None,
                verdict.get("dropped") if verdict.get("kept") is False else None)

    def verify(self, design, program):
        prompt = verify_prompt(self.common[0], design, program["code"])
        return prompt, self.count(prompt)

    def contain(self, design, program, start):
        prompt = contain_prompt(self.common[0], design, start["code"], program["code"])
        return prompt, self.count(prompt)

    def complete(self, base, proposal, verdict, earlier, version=None):
        """Make ``base`` carry out its proposal's Design and keep the rest. ``base`` is the latest version when
        it kept the start algorithm, else the start itself, with the version shown; drops the earlier
        program of the same computation if too long."""
        if version is None:
            head = [self._current(base), WRITTEN_FOR.format(proposal=proposal)]
            reading = READING.format(missing=verdict["missing"])
        else:
            head = [self._current(base), MADE_FOR.format(proposal=proposal),
                    EARLIER_VERSION.format(measured=self.measured(version), code=version["code"].rstrip())]
            sentences = [s for s in (verdict["missing"] if verdict["implemented"] is False else None,
                                     verdict["dropped"]) if s]
            reading = EARLIER_READING.format(sentences="\n".join(sentences))
        earlier_block = (self.EARLIER.format(measured=self.measured(earlier), design=idea_view(earlier["idea"]),
                                             code=earlier["code"].rstrip()) if earlier is not None else "")
        tail = [reading, COMPLETE,
                f"[Output Format]\nReply in this order:\nAnalysis: <a few sentences: {COMPLETE_ANALYSIS}>\n"
                f"Code:\n```python\n<the complete program>\n```\n{self.CLOSING}"]
        trims = []
        result = self._result(self.common + head + [earlier_block] + tail, "Complete")
        if result["input_tokens"] > self.config.max_input_tokens and earlier_block:
            trims.append("earlier_program")
            result = self._result(self.common + head + tail, "Complete", trims=trims)
        if result["input_tokens"] > self.config.max_input_tokens:
            raise ContextTooLong("complete prompt exceeds context")
        return result

    def design(self, program):
        shown = super().design(program)
        missing, dropped = self._reading(program["id"])
        if missing:
            shown += f" (read against this Design: {missing})"
        if dropped:
            shown += f" (besides this Design: {dropped})"
        return shown

    def followup(self, attempt, node, sequence_ids):
        """For an Explore attempt: what reading its program found, and what completing it gave."""
        exploration = attempt.get("exploration")
        if attempt["action"] != "Explore" or not exploration or exploration["step"] != 0:
            return ""
        first = final_attempt(attempt, self.attempts)
        text = ""
        if first["status"] == "valid":
            missing, dropped = self._reading(first["program_id"])
            if missing:
                text += f"; read against its Design: {missing.rstrip('.')}"
            if dropped:
                text += f"; besides its Design: {dropped.rstrip('.')}"
        completions = sorted((a for a in self.attempts.values()
                              if a["action"] == "Complete" and a["exploration"]["id"] == exploration["id"]),
                             key=lambda a: a["id"])
        for index, completion in enumerate(completions):
            final = final_attempt(completion, self.attempts)
            text += f"; {'completed' if index == 0 else 'completed again'}: {self._outcome(final, node, sequence_ids)}"
            if final["status"] == "valid":
                missing, dropped = self._reading(final["program_id"])
                text += ", still not carrying out the Design" if missing else ""
                text += ", still dropping part of the start algorithm" if dropped else ""
        return text

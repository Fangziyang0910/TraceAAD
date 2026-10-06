"""Version-specific goals and context; measured history is shared."""

from traceaad.common.prompts import PromptBuilder as MeasuredPrompts, ContextTooLong


class PromptBuilder(MeasuredPrompts):

    def _current(self, node):
        return f"[Current Algorithm]\n{self.measured(node)}\n```python\n{node['code'].rstrip()}\n```"

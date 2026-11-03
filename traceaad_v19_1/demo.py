"""Small deterministic example that exercises the real search and evaluator."""

import json

from .task import Task


TEMPLATE = '''def choose(values: list[float]) -> int:
    """Return the index of a small value in the nonempty input list."""
    return 0
'''


def _evaluate(source, function):
    instances = ([8.0, 2.0, 5.0], [4.0, 9.0, 1.0], [3.0, 6.0, 7.0])
    scores = []
    for values in instances:
        index = function(list(values))
        if type(index) is not int or not 0 <= index < len(values):
            raise ValueError(f"Expected an index in [0, {len(values)}), got {index!r}")
        scores.append(values[index])
    return sum(scores) / len(scores)


def make_task() -> Task:
    return Task(
        "Select an element from each list to minimize the mean selected value.",
        TEMPLATE, _evaluate, time_limit=10,
        example_code=TEMPLATE,
        example_design="Select the first element.",
        score_definition="Mean selected value; lower is better.",
    )


class DemoModel:
    """Scripted replies, including a format repair and a program repair."""
    def __init__(self):
        self.calls = 0

    def complete(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return '{"analysis": "Choose a small element", "design": "Select a small value", "code":'
        if self.calls == 3:
            return json.dumps({"analysis": "Select a fixed list position as a simple baseline.",
                               "design": "Select the first element.",
                               "code": "def choose(values):\n    return values[999]"})
        expressions = ["0", "len(values) - 1", "min(range(len(values)), key=values.__getitem__)"]
        expression = expressions[self.calls % len(expressions)]
        return json.dumps({"analysis": f"Use {expression} to select a small input value.",
                           "design": f"Return the index computed by {expression}.",
                           "code": f"def choose(values):\n    return {expression}\n"})

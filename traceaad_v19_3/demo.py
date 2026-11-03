"""Scripted model responses using the real search, reflection and evaluator."""

import json

from .task import Task


def evaluate(source, function):
    return sum(abs(function(x)) for x in (0., 1., 2.)) / 3


def make_task():
    template = 'def solve(x: float) -> float:\n    """Return a value close to zero."""\n    return x + 20\n'
    return Task("Design a numeric rule whose output is close to zero for its inputs.",
                template, evaluate, time_limit=5, example_code=template,
                example_design="Apply a large additive correction.",
                score_definition="Mean absolute output; lower is better.")


class DemoModel:
    model = "scripted-offline-demo"

    def __init__(self):
        self.generations = 0

    def complete(self, prompt, **settings):
        if settings["response_format"]["json_schema"]["name"].endswith("reflector"):
            return json.dumps({"positive": [{
                "summary": "Reducing an excessive additive correction improved the numeric rule.",
                "recommendations": ["Adapt the additive correction to the desired output scale."],
                "applicable_when": "A numeric rule has an excessive additive correction."}], "negative": []})
        self.generations += 1
        if self.generations == 1:
            return '{"code":'
        expression = "x + " + str(20 - self.generations)
        if self.generations == 3:
            expression = "undefined_value"
        return json.dumps({"code": f"def solve(x: float) -> float:\n    return {expression}\n",
                           "analysis": "Reduce the additive correction while retaining the input term.",
                           "design": "Apply a smaller additive correction to the input."})

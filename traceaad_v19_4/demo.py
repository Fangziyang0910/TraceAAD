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
        role = settings["response_format"]["json_schema"]["name"].removeprefix("traceaad_v194_")
        if role == "long_term_reflector":
            return json.dumps({"positive": [{
                "summary": "Scaling an additive correction to the output range appears promising.",
                "applicable_when": "A numeric rule applies a fixed additive correction."}], "negative": []})
        if role == "selector":
            choice = int(prompt.split("[Program P", 1)[1].split("]", 1)[0])
            return json.dumps({"assessment": "The first candidate has the clearest recent trend.", "choice": choice})
        if role == "reflector":
            return json.dumps({"reflection": "A smaller additive correction moved the output toward zero."})
        self.generations += 1
        if self.generations == 1:
            return '{"code":'
        expression = "x + " + str(20 - self.generations)
        if self.generations == 3:
            expression = "undefined_value"
        return json.dumps({"code": f"def solve(x: float) -> float:\n    return {expression}\n",
                           "analysis": "Reduce the additive correction while retaining the input term.",
                           "design": "Apply a smaller additive correction to the input."})


class DemoCounter:
    identity = {"demo_tokenizer": "word-count-v1"}

    def __call__(self, text):
        return len(text.split())

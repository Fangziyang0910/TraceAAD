import json

import pytest

from traceaad_base import Task


@pytest.fixture
def task():
    return Task("Return a finite score; lower is better.",
                "def solve(x):\n    return x\n",
                lambda source, function: function(3), time_limit=3)


@pytest.fixture
def reply():
    def make(value=1, *, body=None, design="A candidate algorithm",
             analysis="Use a simple scoring computation to minimize the returned value."):
        return json.dumps({"analysis": analysis, "design": design,
                           "code": "def solve(x):\n    " + (body or f"return {value!r}") + "\n"})
    return make


@pytest.fixture
def scripted_model():
    class ScriptedModel:
        def __init__(self, replies):
            self.replies = iter(replies)
            self.prompts = []
            self.settings = []

        def complete(self, prompt, **settings):
            self.prompts.append(prompt)
            self.settings.append(settings)
            value = next(self.replies)
            if isinstance(value, BaseException):
                raise value
            return value
    return ScriptedModel

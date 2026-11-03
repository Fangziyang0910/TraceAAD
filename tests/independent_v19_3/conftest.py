import json
import os

import pytest

from traceaad_v19_3 import Task

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Words:
    identity = {"test_tokenizer": "space-count-v1"}

    def __call__(self, text):
        return len(text.split())


class Scripted:
    model = "scripted-test"

    def __init__(self, replies):
        self.replies = iter(replies)
        self.prompts, self.settings = [], []

    def complete(self, prompt, **settings):
        self.prompts.append(prompt)
        self.settings.append(settings)
        value = next(self.replies)
        if isinstance(value, BaseException):
            raise value
        return value


@pytest.fixture
def task():
    return Task("Design a numeric rule; lower is better.", "def solve(x):\n    return x\n",
                lambda source, function: function(3), time_limit=3)


@pytest.fixture
def counter():
    return Words()


@pytest.fixture
def model():
    return Scripted


@pytest.fixture
def reply():
    def create(value=1, *, body=None):
        return json.dumps({"code": "def solve(x):\n    " + (body or f"return {value!r}") + "\n",
                           "analysis": "Adapt the bounded lookahead decision rule.",
                           "design": "A bounded lookahead decision rule."})
    return create


@pytest.fixture
def lesson():
    return {"summary": "Bounded lookahead supported an improved decision rule.",
            "recommendations": ["Adapt bounded lookahead to the existing rule."],
            "applicable_when": "A decision rule uses bounded lookahead."}


@pytest.fixture
def reflection(lesson):
    return json.dumps({"positive": [lesson], "negative": []})

"""Whether a program is a test of the Design it was written for, read in two separate requests:
it carries out the Design, and it keeps the computations of its start algorithm that the Design does not replace.

Shared by the search and by the offline replay of earlier runs.
"""

from traceaad.v12_1.delivery import DeliveryError, read_json


VERIFY = """{task}

[Design]
{design}

[Program]
```python
{code}
```

Does the program carry out the main computation the Design describes? Judge by what the code computes and returns, not by its names.
Reply with one JSON object with two string fields, in this order:
"computes": what the program computes, in a few sentences.
"missing": an empty string if the program carries out the Design's main computation; otherwise one sentence naming the part of the Design's main computation that the code does not carry out."""

CONTAIN = """{task}

[Start Algorithm]
```python
{start}
```

[Design]
{design}

[Program]
```python
{code}
```

The program was written to carry out the Design in the start algorithm. Does it keep the computations of the start algorithm that the Design does not replace? Judge by what the code computes, not by its names. A changed constant is not a dropped computation, and neither is a computation whose result the start algorithm uses only for what the Design replaces, or does not use.
Reply with one JSON object with two string fields, in this order:
"computations": the computations of the start algorithm in a few words each, and for each whether the Design replaces it and whether the program keeps it.
"dropped": an empty string if the program keeps every computation of the start algorithm that the Design does not replace; otherwise one sentence naming the computations that the Design does not replace and the program does not keep."""

SENTENCE_CHARS = 400


def verify_prompt(task, design, code):
    """The request that reads the program against its Design; ``task`` is the task section of the search prompts."""
    from .prompts import idea_view

    return VERIFY.format(task=task, design=idea_view(design), code=code.rstrip())


def contain_prompt(task, design, start, code):
    """The request that reads the program against the start algorithm it was written from."""
    from .prompts import idea_view

    return CONTAIN.format(
        task=task, start=start.rstrip(), design=idea_view(design), code=code.rstrip()
    )


def _judgment(text, field):
    """(holds, sentence) from the judgment field: an empty field holds; a reply without it is not judged.
    Only the judgment is read: the description before it is the model's reasoning, in any shape."""
    try:
        reply = read_json(text, "stop", (field,))
    except DeliveryError:
        return None, None
    sentence = " ".join(reply[field].split())
    if sentence.lower().rstrip(".").strip("\"'") in ("", "none", "nothing", "no"):
        return True, None
    return False, sentence[:SENTENCE_CHARS]


def parse_verdict(text):
    """(implemented, missing) from the verify reply; (None, None) when it has no judgment."""
    return _judgment(text, "missing")


def parse_dropped(text):
    """(kept, dropped) from the contain reply; (None, None) when it has no judgment."""
    return _judgment(text, "dropped")

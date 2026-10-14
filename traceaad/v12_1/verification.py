"""Whether a program is a test of the Design it was written for, read in two separate requests:
it carries out the Design, and it keeps the computations of its start algorithm that the Design does not replace.

Shared by the search and by the offline replay of earlier runs.
"""

import re

from traceaad.common.prompts import idea_view

VERIFY = """{task}

[Design]
{design}

[Program]
```python
{code}
```

Does the program carry out the main computation the Design describes? Judge by what the code computes and returns, not by its names.
First state in a few sentences what the program computes. Then end your reply with one line: "Carried out: yes", or "Missing:" followed by one sentence naming the part of the Design's main computation that the code does not carry out."""

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
First list the computations of the start algorithm in a few words each, and for each say whether the Design replaces it and whether the program keeps it. Then end your reply with one line: "Dropped: none", or "Dropped:" followed by one sentence naming the computations that the Design does not replace and the program does not keep."""

SENTENCE_CHARS = 400
LINE = r"^[\s*`\"'#>-]*({labels})[\s*]*:(.*)$"
VERDICT = re.compile(LINE.format(labels="carried out|missing"), re.IGNORECASE | re.MULTILINE)
DROPPED = re.compile(LINE.format(labels="dropped"), re.IGNORECASE | re.MULTILINE)


def verify_prompt(task, design, code):
    """The request that reads the program against its Design; ``task`` is the task section of the search prompts."""
    return VERIFY.format(task=task, design=idea_view(design), code=code.rstrip())


def contain_prompt(task, design, start, code):
    """The request that reads the program against the start algorithm it was written from."""
    return CONTAIN.format(task=task, start=start.rstrip(), design=idea_view(design), code=code.rstrip())


def _last(pattern, text):
    lines = pattern.findall(text or "")
    if not lines:
        return None, None
    label, rest = lines[-1]
    return label.lower(), " ".join(rest.strip().strip("*`\"'").split())


def _none(rest):
    return rest.lower().rstrip(".").strip("\"'") in ("", "none", "nothing", "no")


def parse_verdict(text):
    """(implemented, missing) from the last verdict line; (None, None) when the reply has none."""
    label, rest = _last(VERDICT, text)
    if label is None:
        return None, None
    if label == "carried out":
        return rest.lower().startswith("yes"), None
    return (True, None) if _none(rest) else (False, rest[:SENTENCE_CHARS])


def parse_dropped(text):
    """(kept, dropped) from the last "Dropped:" line; (None, None) when the reply has none."""
    label, rest = _last(DROPPED, text)
    if label is None:
        return None, None
    return (True, None) if _none(rest) else (False, rest[:SENTENCE_CHARS])

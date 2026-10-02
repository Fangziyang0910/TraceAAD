"""Count calls to the candidate function and the time spent inside it.

The evaluation process is forked from the search process, so counters held
in shared memory remain readable after a timeout kills the evaluation: a
timeout becomes a measurement (how far the evaluation got), not only a verdict.
Only outermost calls are counted, so recursion and helper calls through the
target name neither inflate the count nor double the time.
"""

import time

_calls = None
_seconds = None
_depth = 0

MODULE = __name__


def arm(calls, seconds):
    """Install the shared counters inside the evaluation process."""
    global _calls, _seconds, _depth
    _calls, _seconds, _depth = calls, seconds, 0


def call(function, args, kwargs):
    global _depth
    if _calls is None or _depth:
        return function(*args, **kwargs)
    _depth = 1
    start = time.perf_counter()
    try:
        return function(*args, **kwargs)
    finally:
        _seconds.value += time.perf_counter() - start
        _calls.value += 1
        _depth = 0


def instrument(source, name):
    """Source whose target function counts its calls; evaluators that execute
    the program text again (OBP, once per instance) get the counted version too."""
    return (f"{source.rstrip()}\n\n"
            f"import {MODULE} as _traceaad_probe_\n"
            f"_traceaad_target_ = {name}\n\n"
            f"def {name}(*args, **kwargs):\n"
            f"    return _traceaad_probe_.call(_traceaad_target_, args, kwargs)\n")

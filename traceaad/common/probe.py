"""Count calls to the candidate function and the time spent inside it.

The evaluation process is forked from the search process, so counters held
in shared memory remain readable after a timeout kills the evaluation: a
timeout becomes a measurement (how far the evaluation got), not only a verdict.
Only outermost calls are counted, so recursion and helper calls through the
target name neither inflate the count nor double the time. The start of the
call in progress is shared as well, so a timeout inside one call is told
apart from many calls that together run out of time.

CPU time inside the function is counted as well: the budget is CPU time, so
time the process waits while other processes use its core does not count.
"""

import time

_calls = None
_seconds = None
_started = None
_cpu = None
_cpu_started = None
_depth = 0

MODULE = __name__


def arm(calls, seconds, started, cpu=None, cpu_started=None):
    """Install the shared counters inside the evaluation process."""
    global _calls, _seconds, _started, _cpu, _cpu_started, _depth
    _calls, _seconds, _started, _cpu, _cpu_started, _depth = calls, seconds, started, cpu, cpu_started, 0


def process_cpu():
    """CPU time of this process, all threads, in nanosecond resolution: the clock
    /proc/<pid>/task/*/schedstat reports, so the search process reads the same value.
    os.times() counts 10 ms ticks, too coarse for functions called thousands of times."""
    return time.process_time()


def call(function, args, kwargs):
    global _depth
    if _calls is None or _depth:
        return function(*args, **kwargs)
    _depth = 1
    start = time.monotonic()  # system-wide clock: the search process reads it too
    cpu = process_cpu() if _cpu is not None else None
    if cpu is not None:
        _cpu_started.value = cpu
    _started.value = start
    try:
        return function(*args, **kwargs)
    finally:
        _seconds.value += time.monotonic() - start
        if cpu is not None:
            _cpu.value += process_cpu() - cpu
        _calls.value += 1
        _started.value = 0.0
        _depth = 0


def instrument(source, name):
    """Source whose target function counts its calls; evaluators that execute
    the program text again (OBP, once per instance) get the counted version too."""
    return (f"{source.rstrip()}\n\n"
            f"import {MODULE} as _traceaad_probe_\n"
            f"_traceaad_target_ = {name}\n\n"
            f"def {name}(*args, **kwargs):\n"
            f"    return _traceaad_probe_.call(_traceaad_target_, args, kwargs)\n")

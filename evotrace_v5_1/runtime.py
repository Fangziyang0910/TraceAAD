"""Bounded execution in a separate process; afterwards its whole process group is killed."""

import ctypes
import multiprocessing
import os
import signal
import sys
import traceback


class ProcessTimeout(TimeoutError):
    pass


class ProcessFailure(RuntimeError):
    pass


def _worker(connection, function, args):
    """Run in a new process group that dies with its parent; send ("result", value) or ("error", traceback)."""
    os.setsid()
    if sys.platform.startswith("linux"):
        ctypes.CDLL(None).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    try:
        outcome = ("result", function(*args))
    except BaseException:  # noqa: BLE001 - report candidate termination to the parent process
        outcome = ("error", traceback.format_exc())
    connection.send(outcome)
    connection.close()


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.kill()
    process.join(timeout=5)
    process.close()


def isolated_call(function, args, timeout: float):
    """Return function(*args) computed in a separate process; the deadline covers the whole call."""
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, function, args))
    process.start()
    writer.close()
    try:
        if not reader.poll(timeout):
            raise ProcessTimeout(f"Execution exceeded {timeout:g} seconds")
        tag, payload = reader.recv()
    except EOFError as exc:
        raise ProcessFailure("Execution process exited without returning a result") from exc
    finally:
        reader.close()
        _stop(process)
    if tag == "error":
        raise ProcessFailure(payload)
    return payload

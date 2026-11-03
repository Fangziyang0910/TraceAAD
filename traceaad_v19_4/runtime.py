"""Bounded execution in a separate process and cleanup of its child processes."""

import ctypes
import multiprocessing
import os
import signal
import sys
import time
import traceback


class ProcessTimeout(TimeoutError):
    pass


class ProcessFailure(RuntimeError):
    pass


def _worker(connection, function, args):
    if hasattr(os, "setsid"):
        os.setsid()
    if sys.platform.startswith("linux"):
        parent = os.getppid()
        try:
            ctypes.CDLL(None).prctl(1, int(signal.SIGKILL))
            if os.getppid() != parent or parent == 1:
                os._exit(1)
        except (AttributeError, OSError):
            pass
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    started = time.monotonic()
    connection.send(("started", started))
    try:
        value = function(*args)
        outcome = ("result", value, time.monotonic() - started)
    except BaseException:
        outcome = ("error", traceback.format_exc(), time.monotonic() - started)
    try:
        connection.send(outcome)
    finally:
        connection.close()


def _descendants(pid):
    """Read Linux process ancestry before terminating the worker."""
    found = []
    try:
        with open(f"/proc/{pid}/task/{pid}/children") as stream:
            child_text = stream.read()
    except OSError:
        return found
    for value in child_text.split():
        child = int(value)
        found.extend(_descendants(child))
        found.append(child)
    return found


def _stop(process):
    descendants = _descendants(process.pid)
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # The worker starts a fresh session. Its group may outlive the worker itself.
    if hasattr(os, "killpg"):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.is_alive():
        process.kill()
    process.join(timeout=5)
    process.close()


def isolated_call(function, args, timeout: float):
    """Return (value, seconds); the deadline covers the entire function call."""
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    context = multiprocessing.get_context(method)
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(writer, function, args), daemon=False)
    try:
        process.start()
        writer.close()
        if not reader.poll(30):
            raise ProcessFailure("Execution process did not start within 30 seconds")
        tag, started = reader.recv()
        if tag != "started":
            raise ProcessFailure("Execution process did not report its start")
        remaining = max(0.0, started + timeout - time.monotonic())
        if not reader.poll(remaining):
            raise ProcessTimeout(f"Execution exceeded {timeout:g} seconds")
        tag, payload, elapsed = reader.recv()
        if elapsed > timeout:
            raise ProcessTimeout(f"Execution exceeded {timeout:g} seconds")
        if tag == "error":
            raise ProcessFailure(payload)
        return payload, elapsed
    except EOFError as exc:
        raise ProcessFailure("Execution process exited without returning a result") from exc
    finally:
        reader.close()
        writer.close()
        if process.pid is not None:
            _stop(process)
        else:
            process.close()


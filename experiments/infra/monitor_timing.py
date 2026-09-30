"""Search throughput and conservative parallel ETA from recorded budget progress."""

from datetime import datetime, timezone
import math
import time


def timestamp(value):
    """Naive native logs use the server's local timezone; honor explicit offsets."""
    try:
        result = datetime.fromisoformat(value).timestamp()
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def search_timing(run, summary, snapshot=None, *, unit="预算单位", now=None):
    now = time.time() if now is None else now
    snapshot = snapshot or {}
    used, budget, status = run["budget_used"], run["budget"], run["status"]
    phase = snapshot.get("phase") or summary.get("phase")
    done = (status == "finished" or phase in {"freeze", "selection", "selection_failed", "finished"}
            or (budget > 0 and used >= budget))
    result = {"unit": unit, "basis": "wall_time", "elapsed_seconds": None,
              "rate_per_minute": None, "seconds_per_unit": None,
              "eta_seconds": None, "eta_at": None, "state": "unavailable"}

    matched = snapshot.get("completed") == used
    # A live run's checkpoint may trail its candidate count by a step; its
    # elapsed/completed_at pair is still self-consistent, so keep using it.
    use_snapshot = matched or (status == "running" and snapshot.get("completed_at") is not None)
    start = timestamp(snapshot.get("started_at") or summary.get("started_at"))
    last = timestamp(snapshot.get("completed_at")) if use_snapshot else None
    if last is None:
        last = timestamp(summary.get("finished_at") or run.get("updated_at"))
    elapsed = snapshot.get("elapsed") if use_snapshot else None
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed <= 0:
        elapsed = last - start if start is not None and last is not None else None
    else:
        result["basis"] = "active_time"
    # Commit timestamps, never attempt.created_at (generation start), anchor progress.
    idle = max(0., now - last) if last is not None else None
    search_active = status == "running" and not done
    if search_active:
        if result["basis"] == "active_time" and idle is not None:
            elapsed += idle
        elif start is not None:
            elapsed = now - start
            result["basis"] = "wall_time"
        else:
            elapsed = None
    if elapsed is not None and math.isfinite(elapsed) and elapsed > 0:
        result["elapsed_seconds"] = elapsed
        # A handful of startup samples is insufficient for a useful ETA.
        if used >= 3 and elapsed >= 60:
            result["rate_per_minute"] = used * 60 / elapsed
            result["seconds_per_unit"] = elapsed / used

    if done:
        result.update(state="finished" if status == "finished" else "search_complete", eta_seconds=0.)
    elif not search_active:
        result["state"] = "inactive"
    elif budget <= 0 or elapsed is None or elapsed <= 0 or last is None or last > now + 60:
        result["state"] = "unavailable"
        result["rate_per_minute"] = result["seconds_per_unit"] = None
    elif idle > max(900., 5 * max(0., elapsed - idle) / max(1, used)):
        # Heuristic only: a slow evaluation is not proof the process has died.
        result["state"] = "stale"
        result["rate_per_minute"] = result["seconds_per_unit"] = None
    elif result["rate_per_minute"] is None:
        result["state"] = "warming_up"
    else:
        eta = max(0, budget - used) * result["seconds_per_unit"]
        result.update(state="estimated", eta_seconds=eta,
                      eta_at=datetime.fromtimestamp(now + eta, timezone.utc).isoformat())
    return result


def batch_timing(runs):
    """Parallel completion is the slowest outstanding run, not summed durations."""
    pending = [run for run in runs if run["timing"]["state"] not in {"finished", "search_complete"}]
    active = [run for run in pending if run["status"] == "running"]
    measured = [run for run in active if run["timing"]["rate_per_minute"] is not None]
    estimated = [run for run in pending if run["timing"]["eta_seconds"] is not None]
    units = {run["timing"]["unit"] for run in runs}
    result = {"unit": next(iter(units)) if len(units) == 1 else "预算单位",
              "rate_per_minute": None, "rate_runs": len(measured), "active_runs": len(active),
              "eta_runs": len(estimated), "pending_runs": len(pending),
              "eta_seconds": None, "eta_at": None, "state": "unavailable"}
    if measured and len(units) == 1:
        result["rate_per_minute"] = sum(run["timing"]["rate_per_minute"] for run in measured)
    if runs and not pending:
        result.update(state="search_complete", eta_seconds=0.)
    elif pending and len(estimated) == len(pending):
        slowest = max(estimated, key=lambda run: run["timing"]["eta_seconds"])
        result.update(state="estimated", eta_seconds=slowest["timing"]["eta_seconds"],
                      eta_at=slowest["timing"]["eta_at"])
    elif estimated:
        result["state"] = "partial"
    return result

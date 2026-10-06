"""Program facts and one journal commit per completed attempt."""

from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

from traceaad.common.storage import write_json
from traceaad.common.canonical import canonical, key
from traceaad.common.evaluation import failing_line


@dataclass
class Progress:
    phase: str = "roots"
    init_attempts: int = 0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    service_failures: int = 0
    too_long: list[int] = field(default_factory=list)
    finalists: list[int] = field(default_factory=list)
    selection_results: list[dict] = field(default_factory=list)
    selected_id: int | None = None
    repair_id: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    elapsed: float = 0.0


class Facts:
    """Own programs and attempts; valid programs and code indices are views.

    Old journals are converted while reading. Only the last complete commit
    contributes to recovery; an unfinished tail is discarded on the next write.
    """

    def __init__(self, run_dir):
        self.path = Path(run_dir) / "search.jsonl"
        self.summary_path = Path(run_dir) / "logs/run_summary.json"
        self.programs, self.attempts, self.explorations = {}, {}, {}
        self.evaluations = []
        self.state = None
        self.offset = 0
        if not self.path.exists():
            return
        pending = []
        with self.path.open("rb") as handle:
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                if raw.startswith((b'{"kind":"call",', b'{"kind":"request",',
                                   b'{"kind":"artifact",')):
                    continue
                row = json.loads(raw)
                pending.append(row)
                state = row.get("state")
                if state is not None and not state.get("pending"):
                    for committed in pending:
                        self._apply(committed)
                    pending.clear()
                    self.state = state
                    self.offset = handle.tell()

    def _apply(self, row):
        if "attempt" in row:
            attempt = row["attempt"]
            self.attempts[attempt["id"]] = attempt
            if program := row.get("program"):
                self.programs[program["id"]] = program
        else:
            # The original V10.15–19 formats kept valid programs separately and
            # represented failed programs in the attempt that produced them.
            kind, data = row.get("kind"), row.get("data")
            if kind == "node":
                self.programs[data["id"]] = {"valid": True, "failure": None, **data}
            elif kind == "attempt":
                data = dict(data)
                data.setdefault("program_id", data.get("node_id"))
                data.setdefault("repair_of", None)
                if data["repair_of"] is not None:
                    data["action"] = "Repair"
                self.attempts[data["id"]] = data
                if data.get("new_program", True) and data["status"] in {
                        "invalid_source", "runtime_error", "invalid_output", "timeout"}:
                    parent = self.programs.get(data["parent_id"])
                    code = data.get("code") or data.get("completed_code") or data.get("raw_code") or ""
                    try:
                        code = canonical(code)
                    except (SyntaxError, ValueError):
                        code = code.rstrip() + "\n"
                    self.programs[data["id"]] = {
                        "id": data["id"], "key": data.get("key") or key(code), "code": code,
                        "fitness": None, "score": None,
                        "parent_id": data["parent_id"], "action": data.get("executed_action", data["action"]),
                        "reference_id": data["reference_id"], "idea": data["idea"],
                        "depth": parent["depth"] + 1 if parent else 0, "valid": False,
                        "repaired": data["repair_of"] is not None,
                        "failure": {"kind": data["status"], "error": data["error"],
                                    "seconds": data.get("seconds"), "calls": data.get("calls"),
                                    "function_seconds": data.get("function_seconds"),
                                    "call_running": data.get("call_running", False),
                                    "line": failing_line(data["error"], code)
                                    if data["status"] in {"runtime_error", "invalid_output"} else None}}
                    data["program_id"] = data["id"]
                if data["program_id"] is None and data.get("key"):
                    data["program_id"] = next((p["id"] for p in self.programs.values()
                                               if p["key"] == data["key"]), None)
                for name in ("code", "raw_code", "completed_code", "fitness", "score", "key"):
                    data.pop(name, None)
                if data.get("delivery"):
                    data["delivery"] = {k: v for k, v in data["delivery"].items()
                                        if k not in {"submitted_code", "completed_code"}}
            elif kind == "evaluation":
                self.evaluations.append(data)
            elif kind == "exploration":
                self.explorations[data["id"]] = data
        self.evaluations.extend(row.get("evaluations", []))
        if exploration := row.get("exploration"):
            self.explorations[exploration["id"]] = exploration

    @property
    def valid(self):
        return {pid: p for pid, p in self.programs.items() if p["valid"]}

    @property
    def by_key(self):
        return {p["key"]: p for p in self.programs.values()}

    def commit(self, state, *, attempt=None, program=None, evaluations=(), calls=(), exploration=None):
        row = {"kind": "candidate" if attempt else "state", "state": state}
        if attempt:
            state["attempts"] = attempt["id"]
            row.update(candidate_id=attempt["id"], budget_used=attempt["id"],
                       attempt=attempt, program=program,
                       ts=datetime.now().isoformat(timespec="seconds"))
        if evaluations:
            row["evaluations"] = list(evaluations)
        if calls:
            row["calls"] = list(calls)
        if exploration:
            row["exploration"] = exploration
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            # Completed archives may share an inode with the content store.
            if self.path.stat().st_nlink > 1:
                detached = self.path.with_suffix(".detach")
                shutil.copy2(self.path, detached)
                detached.replace(self.path)
            if self.path.stat().st_size != self.offset:
                with self.path.open("r+b") as handle:
                    handle.truncate(self.offset)
        payload = (json.dumps(row, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")) + "\n").encode("utf-8")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        self._apply(row)
        self.state = state
        self.offset = self.path.stat().st_size

    def save_summary(self, summary):
        write_json(self.summary_path, summary)

    def load_summary(self):
        return json.loads(self.summary_path.read_text(encoding="utf-8")) if self.summary_path.exists() else None

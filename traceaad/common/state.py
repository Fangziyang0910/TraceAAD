"""Search facts, compact events and one latest recovery checkpoint."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .storage import Programs, append_jsonl, committed_rows, detach_link, read_json, seal_calls, write_json


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
    started_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat(timespec="seconds"))
    elapsed: float = 0.0


class Facts:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "events.jsonl"
        self.summary_path = self.run_dir / "summary.json"
        self.checkpoint_path = self.run_dir / "resume.json"
        self.sources = Programs(run_dir)
        self.programs, self.attempts, self.explorations = {}, {}, {}
        self.evaluations = []
        checkpoint = read_json(self.checkpoint_path, {})
        self.state = checkpoint.get("state")
        self.files = checkpoint.get("files", {})
        for row in committed_rows(self.run_dir):
            self._apply(row)

    def _apply(self, row):
        if row["kind"] == "candidate":
            attempt = row["attempt"]
            self.attempts[attempt["id"]] = attempt
        if program := row.get("program"):
            self.programs[program["id"]] = {**program, "code": self.sources.get(program["key"])}
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
        # Reading never changes files. A resumed writer discards uncommitted tails.
        for name, size in self.files.items():
            path = self.run_dir / name
            if path.exists() and path.stat().st_size > size:
                detach_link(path)
                with path.open("r+b") as handle:
                    handle.truncate(size)
        row = {"kind": "candidate" if attempt else "progress"}
        calls = list(attempt.get("calls", [])) if attempt else list(calls)
        call_path = self.run_dir / "calls.jsonl"
        if call_path.with_suffix(".jsonl.gz").exists():
            call_path = call_path.with_suffix(".jsonl.gz")
        for call in calls:
            self.files[call_path.name] = append_jsonl(call_path, {
                **call, "prompt": call.get("prompt", attempt["prompt"] if attempt else "")})
        if program:
            source_key = self.sources.add(program["code"])
            row["program"] = {**{k: v for k, v in program.items() if k != "code"}, "key": source_key}
            self.files["programs.jsonl"] = self.sources.path.stat().st_size
        if attempt:
            row["attempt"] = {k: v for k, v in attempt.items() if k not in {"prompt", "calls"}}
            row["attempt"]["call_ids"] = [c["request_id"] for c in calls]
            state["attempts"] = attempt["id"]
            row.update(candidate_id=attempt["id"], budget_used=attempt["id"],
                       x_label="候选尝试", operator=attempt["action"], status=attempt["status"],
                       fitness=program["fitness"] if program and program["valid"] else None,
                       valid=bool(program and program["valid"]), node_id=attempt["program_id"])
        if evaluations:
            row["evaluations"] = list(evaluations)
        if exploration:
            row["exploration"] = exploration
        row["progress"] = {k: state[k] for k in ("phase", "attempts", "elapsed", "started_at",
                                                "model_calls", "input_tokens", "output_tokens")}
        row["ts"] = datetime.now().astimezone().isoformat(timespec="seconds")
        self.files["events.jsonl"] = append_jsonl(self.path, row)
        write_json(self.checkpoint_path, {"state": state, "files": self.files})
        self._apply(row)
        self.state = state

    def save_summary(self, summary):
        write_json(self.summary_path, summary)
        if summary["status"] in {"finished", "search_complete", "selection_failed", "no_valid_root"}:
            seal_calls(self.run_dir)
            self.files = read_json(self.checkpoint_path, {}).get("files", self.files)

    def load_summary(self):
        return read_json(self.summary_path)

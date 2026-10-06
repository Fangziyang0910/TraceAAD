# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, Literal, Optional, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import json
import logging
from threading import Lock, RLock
from datetime import datetime

from core import Function
from pathlib import Path
from traceaad.common.storage import Programs, append_jsonl, read_json, rows, seal_calls, write_json

# Fields that are safe to log from an LLM object (no secrets).
_LLM_SAFE_FIELDS = frozenset(
    {
        "model",
        "base_url",
        "timeout",
        "max_tokens",
        "temperature",
        "enable_thinking",
        "debug_mode",
        "token_count_mode",
    }
)

# Fields that are safe to log from an Evaluation/Problem object.
_EVAL_SAFE_FIELDS = frozenset(
    {
        "task_description",
        "timeout_seconds",
        "n_workers",
        "split",
        "n_instances",
        "n_ants",
        "n_iterations",
        "aco_seed",
    }
)

# Fields that are safe to log from the method object.
_METHOD_SAFE_FIELDS = frozenset(
    {
        "_n_init",
        "_actions_per_iteration",
        "_max_active_trajectories",
        "_max_trajectory_length",
        "_elite_count",
        "_diversity_count",
        "_softmax_temperature",
        "_maximize",
        "_max_context_tokens",
        "_context_token_limit",
        "_output_token_reserve",
        "_action_max_tokens",
        "_code_max_tokens",
        "_management_threshold",
        "_random_seed",
        "_checkpoint_interval",
        "_max_stalled_iterations",
        "_max_consecutive_sample_failures",
        "_max_sample_nums",
    }
)


class ProfilerBase:
    def __init__(
        self,
        log_dir: Optional[str] = None,
        *,
        initial_num_samples=0,
        log_style: Literal["simple", "complex"] = "complex",
        create_random_path=True,
        num_objs=1,
        **kwargs,
    ):
        """Base profiler for recording experimental results.
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            log_style          : 'simple' (one-line per sample) or 'complex' (verbose).
            create_random_path : create a random log_path according to time.
        """
        assert log_style in ["simple", "complex"]

        self._num_objs = num_objs
        self._num_samples = initial_num_samples
        self._process_start_time = datetime.now(ZoneInfo("Asia/Shanghai"))
        self._result_folder = self._process_start_time.strftime("%Y%m%d_%H%M%S")

        self._log_style = log_style
        self._cur_best_function = (
            None if self._num_objs < 2 else [None] * self._num_objs
        )
        self._cur_best_program_sample_order = (
            None if self._num_objs < 2 else [None] * self._num_objs
        )
        self._cur_best_program_score = (
            float("-inf") if self._num_objs < 2 else [float("-inf")] * self._num_objs
        )
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._process_end_time = None
        self._error_count = 0
        self._llm_call_count = 0
        self._method_event_count = 0
        self._method_state_count = 0
        self._finished = False
        self._canonical_best = None
        self._logging_degraded = False

        self._parameters = None
        self._logger_txt = logging.getLogger("root")

        if create_random_path:
            self._log_dir = (
                os.path.join(log_dir, self._result_folder) if log_dir else None
            )
        else:
            self._log_dir = log_dir

        self._register_function_lock = Lock()
        self._artifact_lock = RLock()
        self._sources = Programs(Path(self._log_dir).parent) if self._log_dir else None
        self._llm_calls_path = (
            str(Path(self._log_dir).parent / "calls.jsonl") if self._log_dir else None
        )
        self._method_events_path = (
            str(Path(self._log_dir).parent / "events.jsonl")
            if self._log_dir
            else None
        )
        self._method_state_path = (
            str(Path(self._log_dir).parent / "resume.json") if self._log_dir else None
        )
        self._errors_path = (
            os.path.join(self._log_dir, "errors.jsonl") if self._log_dir else None
        )
        self._run_summary_path = (
            str(Path(self._log_dir).parent / "summary.json") if self._log_dir else None
        )

    def record_parameters(self, llm, prob, method):
        self._parameters = [llm, prob, method]
        self._create_log_path()

    def register_function(
        self, function: Function, program: str = "", *, resume_mode=False
    ):
        """Record an obtained function."""
        try:
            self._register_function_lock.acquire()
            self._num_samples += 1
            self._record_and_print_verbose(
                function, program=program, resume_mode=resume_mode
            )
            if not resume_mode:
                with self._artifact_lock:
                    self._write_json(function, program)
        finally:
            self._register_function_lock.release()

    def finish(self):
        self.write_run_summary(status="finished")
        if self._log_dir:
            seal_calls(Path(self._log_dir).parent)

    def get_logger(self):
        return self._logger_txt

    def resume(self, *args, **kwargs):
        pass

    def log_message(self, message: str):
        if self._log_dir and self._logger_txt.handlers:
            self._logger_txt.info(message)
        else:
            print(message)

    def log_llm_call(self, **payload):
        """Append one LLM interaction to calls.jsonl."""
        if not self._log_dir:
            return
        payload = self._with_common_log_fields(payload)
        payload.setdefault("request_id", self._llm_call_count + 1)
        self._safe_append_jsonl(
            self._llm_calls_path, payload, counter="_llm_call_count"
        )

    def log_method_event(self, event: str | None = None, **payload):
        """Append a method event to events.jsonl."""
        if not self._log_dir:
            return
        if event is not None:
            payload.setdefault("event", event)
        payload = self._with_common_log_fields(payload)
        self._safe_append_jsonl(
            self._method_events_path, {"kind": "method", "data": payload}, counter="_method_event_count"
        )

    def log_method_state(self, phase: str | None = None, **payload):
        """Replace the latest method snapshot in resume.json."""
        if not self._log_dir:
            return
        if phase is not None:
            payload.setdefault("phase", phase)
        payload = self._with_common_log_fields(payload)
        with self._artifact_lock:
            write_json(self._method_state_path, {"method_state": payload})
            self._method_state_count += 1

    def log_error(self, stage: str, exc: Exception | None = None, **payload):
        """Append a structured error record to `errors.jsonl`."""
        if not self._log_dir:
            return
        payload.setdefault("stage", stage)
        if exc is not None:
            payload.setdefault("error_type", type(exc).__name__)
            payload.setdefault("error", self._truncate_text(str(exc), 1000))
            payload.setdefault(
                "traceback",
                self._truncate_text(
                    "".join(
                        traceback.format_exception(type(exc), exc, exc.__traceback__)
                    ),
                    4000,
                ),
            )
        payload = self._with_common_log_fields(payload)
        self._safe_append_jsonl(self._errors_path, payload, counter="_error_count")

    def write_run_summary(self, **payload):
        """Write the final run summary. Idempotent; caller-supplied fields override defaults."""
        with self._artifact_lock:
            if not self._log_dir or self._finished:
                return
            self._process_end_time = datetime.now(ZoneInfo("Asia/Shanghai"))
            config = read_json(Path(self._log_dir).parent / "run_config.json", {})
            status = payload.pop("status", "finished")
            summary = {"status": status, "phase": "finished" if status == "finished" else "stopped",
                       "method": config.get("method"), "budget": config.get("budget", 0),
                       "budget_axis": "样本次数", "budget_used": self._num_samples,
                       "started_at": self._process_start_time.isoformat(),
                       "finished_at": self._process_end_time.isoformat(),
                       "seconds": (self._process_end_time - self._process_start_time).total_seconds(),
                       "num_nodes": self._evaluate_success_program_num,
                       "candidate_count": self._num_samples,
                       "valid_candidate_count": self._evaluate_success_program_num,
                       "best": self._canonical_best, "model_calls": self._llm_call_count,
                       "evaluation_calls": self._evaluate_success_program_num + self._evaluate_failed_program_num,
                       "error_count": self._error_count, "logging_degraded": self._logging_degraded,
                       **payload}
            write_json(self._run_summary_path, summary)
            self._finished = True

    def _write_json(self, function, program="", *, record_type="history", record_sep=200):
        if record_type == "best" or not self._log_dir:
            return
        run_dir = Path(self._log_dir).parent
        config = read_json(run_dir / "run_config.json", {})
        sources = self._sources
        score = float(function.score) if function.score is not None and self._num_objs == 1 else None
        order = self._num_samples
        valid = function.score is not None and bool(np.all(np.isfinite(function.score)))
        metadata = None
        if program:
            metadata = {"id": order, "key": sources.add(program), "fitness": score if valid else None,
                        "score": (-score if config.get("objective") == "min" else score) if valid and score is not None else None,
                        "valid": bool(valid), "action": function.operator or "unknown",
                        "idea": getattr(function, "algorithm", "") or "",
                        "parent_id": None, "depth": 0}
            if self._num_objs > 1:
                metadata["objectives"] = [float(v) if np.isfinite(v) else None for v in function.score] if function.score is not None else None
            if score is not None and valid and (self._canonical_best is None or score > self._canonical_best["fitness"]):
                self._canonical_best = metadata
        elapsed = (datetime.now(ZoneInfo("Asia/Shanghai")) - self._process_start_time).total_seconds()
        attempt = {"id": order, "action": function.operator or "unknown",
                   "idea": getattr(function, "algorithm", "") or "",
                   "program_id": order if metadata else None, "parent_id": None, "repair_of": None,
                   "function_key": sources.add(str(function)),
                   "sample_time": function.sample_time, "evaluate_time": function.evaluate_time,
                   "status": "valid" if valid else "invalid_output"}
        append_jsonl(run_dir / "events.jsonl", {
            "kind": "candidate", "candidate_id": order, "budget_used": order, "x_label": "样本次数",
            "operator": attempt["action"], "status": attempt["status"], "fitness": score if valid else None,
            "valid": bool(valid), "node_id": attempt["program_id"], "attempt": attempt, "program": metadata,
            "ts": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "progress": {"attempts": order, "phase": "search", "elapsed": elapsed,
                         "started_at": self._process_start_time.isoformat()}})

    def _record_and_print_verbose(self, function, program="", *, resume_mode=False):
        function_str = str(function).strip("\n")
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        operator = function.operator

        # update best function
        if self._num_objs < 2:
            if score is not None and score > self._cur_best_program_score:
                self._cur_best_function = function
                self._cur_best_program_score = score
                self._cur_best_program_sample_order = self._num_samples
        else:
            if score is not None:
                for i in range(self._num_objs):
                    if score[i] > self._cur_best_program_score[i]:
                        self._cur_best_function[i] = function
                        self._cur_best_program_score[i] = score[i]
                        self._cur_best_program_sample_order[i] = self._num_samples

        if not resume_mode:
            if self._log_style == "complex":
                print("================= Evaluated Function =================")
                print(f"{function_str}")
                print("------------------------------------------------------")
                print(f"Operator     : {operator}")
                print(f"Score        : {str(score)}")
                print(f"Sample time  : {str(sample_time)}")
                print(f"Evaluate time: {str(evaluate_time)}")
                print(f"Sample orders: {str(self._num_samples)}")
                print("------------------------------------------------------")
                print(f"Current best score: {self._cur_best_program_score}")
                print("======================================================\n")
            else:
                if score is None:
                    if self._num_objs < 2:
                        print(
                            f"Sample{self._num_samples}: Score=None    Cur_Best_Score={self._cur_best_program_score: .3f}"
                        )
                    else:
                        best_scores_str = ", ".join(
                            [f"{s: .3f}" for s in self._cur_best_program_score]
                        )
                        print(
                            f"Sample{self._num_samples}: Score=None    Cur_Best_Score=[{best_scores_str}]"
                        )
                else:
                    if self._num_objs < 2:
                        print(
                            f"Sample{self._num_samples}: Score={score: .3f}     Cur_Best_Score={self._cur_best_program_score: .3f}"
                        )
                    else:
                        scores_str = ", ".join([f"{s: .3f}" for s in score])
                        best_scores_str = ", ".join(
                            [f"{s: .3f}" for s in self._cur_best_program_score]
                        )
                        print(
                            f"Sample{self._num_samples}: Score=[{scores_str}]     Cur_Best_Score=[{best_scores_str}]"
                        )

        # update statistics about function
        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time is not None:
            self._tot_sample_time += sample_time

        if evaluate_time:
            self._tot_evaluate_time += evaluate_time

    def _create_log_path(self):
        self._llm_calls_path = str(Path(self._log_dir).parent / "calls.jsonl")
        self._method_events_path = str(Path(self._log_dir).parent / "events.jsonl")
        self._method_state_path = str(Path(self._log_dir).parent / "resume.json")
        self._errors_path = os.path.join(self._log_dir, "errors.jsonl")
        self._run_summary_path = str(Path(self._log_dir).parent / "summary.json")
        os.makedirs(self._log_dir, exist_ok=True)

        file_name = self._log_dir + "/run_log.txt"
        file_mode = "a" if os.path.isfile(file_name) else "w"

        self._logger_txt.setLevel(level=logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        for hdlr in self._logger_txt.handlers[:]:
            self._logger_txt.removeHandler(hdlr)
        fileout = logging.FileHandler(file_name, mode=file_mode)
        fileout.setLevel(logging.INFO)
        fileout.setFormatter(formatter)
        self._logger_txt.addHandler(fileout)
        self._logger_txt.addHandler(logging.StreamHandler(sys.stdout))

        llm, prob, method = self._parameters
        self._log_safe_parameters("LLM", llm, _LLM_SAFE_FIELDS)
        self._log_safe_parameters("Problem", prob, _EVAL_SAFE_FIELDS)
        self._log_safe_parameters("Method", method, _METHOD_SAFE_FIELDS)

    def _log_safe_parameters(self, title: str, obj: Any, allowed: frozenset[str]):
        """Log only whitelisted attributes.

        Never iterate `obj.__dict__`: LLM clients hold `api_key` and other
        credentials, which must not reach any log artifact.
        """
        self._logger_txt.info(
            "===================================================================="
        )
        self._logger_txt.info(f"{title} Parameters")
        self._logger_txt.info(
            "--------------------------------------------------------------------"
        )
        self._logger_txt.info(f"  - {title}: {obj.__class__.__name__}")
        for attr in sorted(allowed):
            if not hasattr(obj, attr):
                continue
            value = getattr(obj, attr)
            if callable(value):
                continue
            self._logger_txt.info(
                f"  - {attr.lstrip('_')}: {self._truncate_text(value, 500)}"
            )

    def _safe_append_jsonl(self, path: str, payload: dict, *, counter: str):
        """Append one JSONL record, flagging (not hiding) logging failures."""
        try:
            with self._artifact_lock:
                self._append_jsonl(path, payload)
                setattr(self, counter, getattr(self, counter) + 1)
        except Exception as exc:  # noqa: BLE001 - logging must not break the search
            self._note_logging_degraded(f"failed to append {path}: {exc}")

    def _note_logging_degraded(self, message: str):
        """Record that process evidence is incomplete, and warn once."""
        with self._artifact_lock:
            already_warned = self._logging_degraded
            self._logging_degraded = True
        if not already_warned:
            print(
                f"[profiler] WARNING logging degraded: {message}",
                file=sys.stderr,
                flush=True,
            )

    def _append_jsonl(self, path: str, payload: dict):
        if not self._log_dir or path is None:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as jsonl_file:
            jsonl_file.write(
                json.dumps(payload, ensure_ascii=False, default=self._json_default)
                + "\n"
            )

    def _with_common_log_fields(self, payload: dict):
        payload = dict(payload)
        payload.setdefault(
            "timestamp", datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        )
        payload.setdefault("profiler_sample_order", self._num_samples)
        return payload

    @staticmethod
    def _truncate_text(value: Any, limit: int) -> str:
        text = "" if value is None else str(value)
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    @staticmethod
    def _json_default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if hasattr(value, "__dict__"):
            return value.__dict__
        return str(value)

    @classmethod
    def load_logfile(cls, logdir, valid_only=False) -> Tuple[List[str], List[float]]:
        """Load (program_source, fitness) from the run owning this log directory."""
        run_dir = Path(logdir).parent
        sources = Programs(run_dir)
        programs, functions, scores = {}, [], []
        for row in rows(run_dir / "events.jsonl"):
            if row.get("program"):
                programs[row["program"]["id"]] = row["program"]
            if row["kind"] != "candidate" or valid_only and not row["valid"]:
                continue
            program = programs.get(row["node_id"])
            code = sources.get(program["key"]) if program else sources.get(row["attempt"].get("function_key"))
            functions.append(code or "")
            scores.append(row["fitness"] if row["fitness"] is not None else float("-inf"))
        return functions, scores

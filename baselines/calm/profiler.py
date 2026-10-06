"""Profiler for CALM (w/o GRPO)."""

from __future__ import annotations

import json
import os

from baselines.profiler import ProfilerBase


class CALMProfiler(ProfilerBase):
    def __init__(self, run_dir=None, **kwargs):
        super().__init__(run_dir, **kwargs)
        if self._log_dir:
            self._event_dir = os.path.join(self._log_dir, 'calm')
            self._algo_dir = os.path.join(self._log_dir, 'algos')
            os.makedirs(self._event_dir, exist_ok=True)
            os.makedirs(self._algo_dir, exist_ok=True)

    def save_best_algo(self, *, step: int, sid: str, code: str) -> None:
        if not self._log_dir:
            return
        safe_sid = sid.replace('/', '_').replace(' ', '_')
        path = os.path.join(self._algo_dir, f'S{step}_{safe_sid}.py')
        with open(path, 'w', encoding='utf-8') as fp:
            fp.write(code)

    def save_trace(self, payload: dict) -> None:
        if not self._log_dir:
            return
        path = os.path.join(self._event_dir, 'trace.json')
        with open(path, 'w', encoding='utf-8') as fp:
            json.dump(payload, fp, indent=2)

    def append_log(self, message: str) -> None:
        if not self._log_dir:
            return
        path = os.path.join(self._event_dir, 'output.log')
        with open(path, 'a+', encoding='utf-8') as fp:
            fp.write(message + '\n')

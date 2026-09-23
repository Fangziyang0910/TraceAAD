"""Profiler for CALM (w/o GRPO)."""

from __future__ import annotations

import json
import os
from threading import Lock
from typing import Optional

from llm4ad.base import Function
from llm4ad.tools.profiler import ProfilerBase


class CALMProfiler(ProfilerBase):
    def __init__(
            self,
            log_dir: Optional[str] = None,
            *,
            initial_num_samples=0,
            log_style='complex',
            create_random_path=True,
            **kwargs,
    ):
        super().__init__(
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs,
        )
        self._event_lock = Lock()
        if self._log_dir:
            self._event_dir = os.path.join(self._log_dir, 'calm')
            self._algo_dir = os.path.join(self._log_dir, 'algos')
            os.makedirs(self._event_dir, exist_ok=True)
            os.makedirs(self._algo_dir, exist_ok=True)

    def register_function(self, function: Function, *, program: str = '') -> None:
        super().register_function(function, program=program)

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

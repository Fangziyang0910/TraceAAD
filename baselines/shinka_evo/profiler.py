from __future__ import annotations

import json
import os
from threading import Lock
from typing import Any

from baselines.profiler import ProfilerBase


class ShinkaEvoProfiler(ProfilerBase):
    def __init__(self, run_dir=None, **kwargs):
        super().__init__(run_dir, **kwargs)
        self._event_lock = Lock()
        if self._log_dir:
            self._event_dir = os.path.join(self._log_dir, "shinka_evo")
            os.makedirs(self._event_dir, exist_ok=True)

    def register_event(self, event_type: str, content: dict[str, Any]) -> None:
        self._append_event(f"{event_type}.jsonl", content)
        self.log_method_event(method="shinka_evo", event=event_type, **content)

    def _append_event(self, filename: str, content: dict[str, Any]) -> None:
        if not self._log_dir:
            return
        path = os.path.join(self._event_dir, filename)
        try:
            self._event_lock.acquire()
            with open(path, "a") as jsonl_file:
                jsonl_file.write(json.dumps(content, default=str) + "\n")
        finally:
            if self._event_lock.locked():
                self._event_lock.release()

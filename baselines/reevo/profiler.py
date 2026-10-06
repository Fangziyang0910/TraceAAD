from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

try:
    import wandb
except:
    pass

from .population import Population
from pathlib import Path
from traceaad.common.storage import append_jsonl
from core import Function
from baselines.profiler import ProfilerBase


class ReEvoProfiler(ProfilerBase):
    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """ReEvo Profiler
        Args:
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
        """
        super().__init__(log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._cur_gen = 0

    def register_population(self, pop: Population):
        if not self._log_dir:
            return
        with self._artifact_lock:
            if (self._num_samples == 0 or
                    pop.generation == self._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_json = {
                    'algorithm': f.algorithm,
                    'function_key': self._sources.add(str(f)),
                    'score': f.score
                }
                funcs_json.append(f_json)
            append_jsonl(Path(self._log_dir).parent / 'events.jsonl',
                         {'kind': 'population', 'generation': pop.generation, 'members': funcs_json})
            self._cur_gen += 1

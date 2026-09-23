from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from threading import Lock
from typing import List, Dict, Optional

try:
    import wandb
except:
    pass

from .population import Population
from llm4ad.base import Function
from baselines.profiler import TensorboardProfiler, ProfilerBase, WandBProfiler


class MAProfiler(ProfilerBase):

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """MCTS_AHD Profiler
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
        self._pop_lock = Lock()
        self._mcts_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)
            self._mcts_state_path = os.path.join(self._log_dir, 'mcts_state.jsonl')
            self._mcts_events_path = os.path.join(self._log_dir, 'mcts_events.jsonl')
            self._llm_calls_path = os.path.join(self._log_dir, 'llm_calls.jsonl')

    def register_population(self, pop: Population):
        try:
            self._pop_lock.acquire()
            if (self._num_samples == 0 or
                    pop.generation == self._cur_gen):
                return
            funcs = pop.population  # type: List[Function]
            funcs_json = []  # type: List[Dict]
            for f in funcs:
                f_json = {
                    'algorithm': f.algorithm,
                    'function': str(f),
                    'score': f.score
                }
                funcs_json.append(f_json)
            path = os.path.join(self._ckpt_dir, f'pop_{pop.generation}.json')
            with open(path, 'w') as json_file:
                json.dump(funcs_json, json_file, indent=4)
            self._cur_gen += 1
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def log_message(self, message: str):
        if self._log_dir and self._logger_txt.handlers:
            self._logger_txt.info(message)
        else:
            print(message)

    def log_mcts_state(self, *, phase: str, sample_order: int, max_sample_nums, mcts, selected_node=None):
        if not self._log_dir:
            return
        root_children = [self._node_summary(node) for node in mcts.root.children]
        payload = {
            'phase': phase,
            'sample_order': sample_order,
            'max_sample_nums': max_sample_nums,
            'q_min': mcts.q_min,
            'q_max': mcts.q_max,
            'rank_list': list(mcts.rank_list),
            'root_visits': mcts.root.visits,
            'root_q': mcts.root.Q,
            'root_children': root_children,
        }
        if selected_node is not None:
            payload['selected_node'] = self._node_summary(selected_node)
        self._append_jsonl(self._mcts_state_path, payload)
        self.log_method_state(method='mcts_ahd', **payload)

        best = max(mcts.rank_list) if mcts.rank_list else None
        subtree_sizes = [child['subtree_size'] for child in root_children]
        self.log_message(
            f"MCTS state {phase}: samples={sample_order}/{max_sample_nums}, "
            f"rank_count={len(mcts.rank_list)}, best={best}, root_subtree_sizes={subtree_sizes}"
        )

    def log_mcts_event(self, **payload):
        if not self._log_dir:
            return
        self._append_jsonl(self._mcts_events_path, payload)
        self.log_method_event(method='mcts_ahd', **payload)

        event = payload.get('event', 'event')
        status = payload.get('status')
        operator = payload.get('operator')
        sample_order = payload.get('sample_order')
        parent_score = payload.get('parent_score')
        child_score = payload.get('child_score')
        self.log_message(
            f"MCTS event {event}: status={status}, op={operator}, "
            f"samples={sample_order}, parent_score={parent_score}, child_score={child_score}"
        )

    def log_llm_call(self, **payload):
        super().log_llm_call(**payload)

    def _append_jsonl(self, path: str, payload: dict):
        try:
            self._mcts_lock.acquire()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(payload, ensure_ascii=False) + '\n')
        finally:
            if self._mcts_lock.locked():
                self._mcts_lock.release()

    @staticmethod
    def _node_summary(node):
        individual = getattr(node, 'individual', None)
        raw_info = getattr(node, 'raw_info', None)
        score = None
        if raw_info is not None:
            score = getattr(raw_info, 'score', None)
        if score is None and individual is not None:
            score = getattr(individual, 'score', None)
        return {
            'score': score,
            'q': getattr(node, 'Q', None),
            'depth': getattr(node, 'depth', None),
            'visits': getattr(node, 'visits', None),
            'children_count': len(getattr(node, 'children', [])),
            'subtree_size': len(getattr(node, 'subtree', [])),
            'is_root_child': (
                getattr(getattr(node, 'parent', None), 'is_root', False)
                or getattr(getattr(node, 'parent', None), 'code', None) == 'Root'
            ),
        }

    def _write_json(self, function: Function, program='', *, record_type='history', record_sep=200):
        """Write function data to a JSON file.
        Args:
            function   : The function object containing score and string representation.
            record_type: Type of record, 'history' or 'best'. Defaults to 'history'.
            record_sep : Separator for history records. Defaults to 200.
        """
        assert record_type in ['history', 'best']

        if not self._log_dir:
            return

        sample_order = self._num_samples
        content = {
            'sample_order': sample_order,
            'algorithm': function.algorithm,  # Added when recording
            'function': str(function),
            'score': function.score,
            'operator': getattr(function, 'operator', None),
            'program': program,
        }

        if record_type == 'history':
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            filename = f'samples_{lower_bound + 1}~{upper_bound}.json'
        else:
            filename = 'samples_best.json'

        path = os.path.join(self._samples_json_dir, filename)

        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []

        data.append(content)

        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)


class MATensorboardProfiler(TensorboardProfiler, MAProfiler):

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """MCTS_AHD Profiler for Tensorboard.
        Args:
            log_dir            : the directory of current run
            evaluation_name    : the name of the evaluation instance (the name of the problem to be solved).
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        MAProfiler.__init__(
            self, log_dir=log_dir,
            create_random_path=create_random_path,
            **kwargs
        )
        TensorboardProfiler.__init__(
            self,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )

    def finish(self):
        if self._log_dir:
            self._writer.close()


class MAWandbProfiler(WandBProfiler, MAProfiler):

    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples=0,
                 log_style='complex',
                 create_random_path=True,
                 **kwargs):
        """MCTS_AHD Profiler for Wandb.
        Args:
            wandb_project_name : the name of the wandb project
            log_dir            : the directory of current run
            initial_num_samples: the sample order start with `initial_num_samples`.
            create_random_path : create a random log_path according to evaluation_name, method_name, time, ...
            **kwargs           : kwargs for wandb
        """
        MAProfiler.__init__(
            self,
            log_dir=log_dir,
            create_random_path=create_random_path,
            **kwargs
        )
        WandBProfiler.__init__(
            self,
            wandb_project_name=wandb_project_name,
            log_dir=log_dir,
            initial_num_samples=initial_num_samples,
            log_style=log_style,
            create_random_path=create_random_path,
            **kwargs
        )
        self._pop_lock = Lock()
        if self._log_dir:
            self._ckpt_dir = os.path.join(self._log_dir, 'population')
            os.makedirs(self._ckpt_dir, exist_ok=True)

    def finish(self):
        wandb.finish()

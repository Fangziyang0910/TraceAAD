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

from zoneinfo import ZoneInfo

import numpy as np
import json
import logging
import math
from threading import RLock
from datetime import datetime

from pathlib import Path
from traceaad.common.storage import Programs, append_jsonl, read_json, seal_calls, write_json

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
    """Record one scalar-score research run in the canonical format."""

    def __init__(self, run_dir=None, *, initial_num_samples=0):
        self.run_dir = Path(run_dir) if run_dir else None
        self._log_dir = str(self.run_dir / 'logs') if self.run_dir else None
        self._num_samples = initial_num_samples
        self._process_start_time = datetime.now(ZoneInfo('Asia/Shanghai'))
        self._artifact_lock = RLock()
        self._sources = Programs(self.run_dir) if self.run_dir else None
        self._canonical_best = None
        self._evaluate_success_program_num = self._evaluate_failed_program_num = 0
        self._llm_call_count = self._method_event_count = self._method_state_count = self._error_count = 0
        self._finished = False
        self._llm = None
        self._recorded_transport = set()
        self._logger_txt = logging.getLogger('traceaad.' + str(self.run_dir))
        for field, name in (('_llm_calls_path', 'calls.jsonl'), ('_method_events_path', 'events.jsonl'),
                            ('_method_state_path', 'resume.json'), ('_run_summary_path', 'summary.json')):
            setattr(self, field, str(self.run_dir/name) if self.run_dir else None)
        self._errors_path = str(self.run_dir/'logs/errors.jsonl') if self.run_dir else None

    def record_parameters(self, llm, prob, method):
        self._llm = llm
        if not self.run_dir:
            return
        Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        for handler in self._logger_txt.handlers[:]:
            handler.close()
            self._logger_txt.removeHandler(handler)
        handler = logging.FileHandler(Path(self._log_dir)/'run_log.txt')
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s'))
        self._logger_txt.addHandler(handler)
        self._logger_txt.setLevel(logging.INFO)
        self._logger_txt.propagate = False
        for title, obj, fields in (('LLM', llm, _LLM_SAFE_FIELDS), ('Problem', prob, _EVAL_SAFE_FIELDS),
                                    ('Method', method, _METHOD_SAFE_FIELDS)):
            self.log_message(title + ': ' + type(obj).__name__)
            for field in sorted(fields):
                if hasattr(obj, field) and not callable(getattr(obj, field)):
                    self.log_message(f'  {field}: {str(getattr(obj, field))[:500]}')

    def register_function(self, function, program=''):
        with self._artifact_lock:
            self._num_samples += 1
            valid = function.score is not None and math.isfinite(float(function.score))
            self._evaluate_success_program_num += valid
            self._evaluate_failed_program_num += not valid
            if self.run_dir:
                self._write_json(function, program)
            best = self._canonical_best['fitness'] if self._canonical_best else None
            print(f'Sample {self._num_samples}: {function.operator} score={function.score} best={best}', flush=True)

    def _write_json(self, function, program):
        fitness = float(function.score) if function.score is not None and math.isfinite(float(function.score)) else None
        order, valid = self._num_samples, fitness is not None
        meta = None
        if program:
            config = read_json(self.run_dir/'run_config.json', {})
            meta = {'id': order, 'key': self._sources.add(program), 'fitness': fitness,
                    'score': -fitness if fitness is not None and config.get('objective') == 'min' else fitness,
                    'valid': valid, 'action': function.operator or 'unknown',
                    'idea': getattr(function, 'algorithm', '') or '', 'parent_id': None, 'depth': 0}
            if valid and (self._canonical_best is None or fitness > self._canonical_best['fitness']):
                self._canonical_best = meta
        attempt = {'id': order, 'action': function.operator or 'unknown',
                   'idea': getattr(function, 'algorithm', '') or '', 'program_id': order if meta else None,
                   'parent_id': None, 'repair_of': None, 'function_key': self._sources.add(str(function)),
                   'sample_time': function.sample_time, 'evaluate_time': function.evaluate_time,
                   'status': 'valid' if valid else 'invalid_output', 'call_ids': []}
        self._append_jsonl(str(self.run_dir/'events.jsonl'), {
            'kind': 'candidate', 'candidate_id': order, 'budget_used': order, 'x_label': '样本次数',
            'operator': attempt['action'], 'status': attempt['status'], 'fitness': fitness,
            'valid': valid, 'node_id': attempt['program_id'], 'attempt': attempt, 'program': meta,
            'ts': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'progress': {'attempts': order, 'phase': 'search', 'elapsed': self._elapsed(),
                         'started_at': self._process_start_time.isoformat()}})

    def register_population(self, pop):
        if not self.run_dir or not self._num_samples or pop.generation == self._cur_gen:
            return
        with self._artifact_lock:
            members = [{'algorithm': f.algorithm, 'function_key': self._sources.add(str(f)), 'score': f.score}
                       for f in pop.population]
            self._append_jsonl(str(self.run_dir/'events.jsonl'),
                {'kind': 'population', 'generation': pop.generation, 'members': members})
            self._cur_gen = pop.generation

    def log_llm_call(self, *, llm=None, transport_calls=None, **payload):
        if not self.run_dir:
            return
        calls = transport_calls if transport_calls is not None else getattr(llm or self._llm, 'last_calls', [])
        response_matches = calls and payload.get('response') == calls[-1]['response']
        if transport_calls is not None or response_matches or calls and payload.get('prompt') == calls[-1]['prompt']:
            for call in calls:
                if call['transport_id'] not in self._recorded_transport:
                    self._append_counted(self._llm_calls_path, {**payload, **call}, '_llm_call_count')
                    self._recorded_transport.add(call['transport_id'])
        else:
            self._append_counted(self._llm_calls_path, payload, '_llm_call_count')

    def log_method_event(self, event=None, **payload):
        if event is not None:
            payload['event'] = event
        self._append_counted(self._method_events_path, {'kind': 'method', 'data': self._common(payload)},
                             '_method_event_count')

    def log_method_state(self, phase=None, **payload):
        if not self.run_dir:
            return
        if phase is not None:
            payload['phase'] = phase
        with self._artifact_lock:
            write_json(self._method_state_path, {'method_state': self._plain(self._common(payload))})
            self._method_state_count += 1

    def log_error(self, stage, exc=None, **payload):
        payload['stage'] = stage
        if exc:
            payload.update(error_type=type(exc).__name__, error=str(exc)[:1000])
        self._append_counted(self._errors_path, payload, '_error_count')

    def _append_counted(self, path, payload, counter):
        if not self.run_dir:
            return
        with self._artifact_lock:
            if counter == '_llm_call_count':
                payload = {**payload, 'request_id': self._llm_call_count+1}
            self._append_jsonl(path, self._common(payload))
            setattr(self, counter, getattr(self, counter)+1)

    def _append_jsonl(self, path, payload):
        with self._artifact_lock:
            append_jsonl(path, self._plain(payload))

    def _common(self, payload):
        return {'timestamp': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'profiler_sample_order': self._num_samples, **payload}

    @staticmethod
    def _plain(payload):
        return json.loads(json.dumps(payload, default=lambda value: value.tolist()
            if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value), allow_nan=False))

    def _elapsed(self):
        return (datetime.now(ZoneInfo('Asia/Shanghai'))-self._process_start_time).total_seconds()

    def write_run_summary(self, **payload):
        if not self.run_dir or self._finished:
            return
        with self._artifact_lock:
            config = read_json(self.run_dir/'run_config.json', {})
            status = payload.pop('status', 'finished')
            write_json(self._run_summary_path, {'status': status,
                'phase': 'finished' if status == 'finished' else 'stopped',
                'method': config.get('method'), 'budget': config.get('budget', 0),
                'budget_axis': '样本次数', 'budget_used': self._num_samples,
                'started_at': self._process_start_time.isoformat(),
                'finished_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'seconds': self._elapsed(),
                'num_nodes': self._evaluate_success_program_num, 'candidate_count': self._num_samples,
                'valid_candidate_count': self._evaluate_success_program_num, 'best': self._canonical_best,
                'model_calls': self._llm_call_count,
                'evaluation_calls': self._evaluate_success_program_num+self._evaluate_failed_program_num,
                'error_count': self._error_count, **payload})
            self._finished = True

    def finish(self):
        self.write_run_summary(status='finished')
        if self.run_dir:
            seal_calls(self.run_dir)
        for handler in self._logger_txt.handlers[:]:
            handler.close()
            self._logger_txt.removeHandler(handler)

    def get_logger(self):
        return self._logger_txt

    def log_message(self, message):
        if self._logger_txt.handlers:
            self._logger_txt.info(message)
        else:
            print(message)

"""JSON snapshots at completed search boundaries; never serialize clients or executors."""
import dataclasses
import gzip
import json
import math
import os
from pathlib import Path
import random
import tempfile
from functools import lru_cache

import numpy as np


NAME = 'search_checkpoint.json.gz'
# shortcut: unfinished phases restart at a saved boundary; persist pending work for exact mid-phase replay.
FIELDS = {
    'funsearch': (),
    'clade_ahd': ('_phase', '_population', '_init_population', '_tree_state', '_iteration',
                  '_evaluation_count', '_upstream_evals'),
    'deltaevolve': ('_iteration', '_seeded', '_pending_evaluation', '_pending_node'),
    'openevolve': ('_seeded', '_pending_evaluation'),
    'hsevo': ('_phase', '_iteration', '_init_index', '_init_population', '_population', '_elitist',
              '_str_flash_memory', '_str_comprehensive_memory', '_lst_good_reflection', '_lst_bad_reflection'),
    'refineevo': ('_phase', '_generation', '_init_index', '_init_population', '_population', '_id_counter',
                  '_op_desc', '_strategy_history', '_improvement_history'),
    'calm': ('_algos', '_seed_algos', '_used_prompts', '_last_revisited_prompts',
             '_injected_components', '_seen_code_signatures', '_seen_performance_profiles',
             '_numeric_refine_attempted', '_age_stuck', '_train_epoch', '_log_step', '_messages', '_best_perf'),
    'pathwise': ('_outer_iteration', '_inner_step', '_node_counter', '_best_node',
                 '_all_nodes_archive', '_root_node_ids', '_discarded_nodes', '_policy_reflection',
                 '_world_model_reflection', '_policy_reflection_history', '_world_model_reflection_history',
                 '_initialization_index', '_initial_candidates', '_initialization_complete'),
    'shinka_evo': ('_generation', '_evaluated_since_meta', '_meta_summary', '_meta_scratch_pad',
                   '_meta_recommendations', '_meta_recommendations_history', '_best_function_found'),
}
GROUPS = {
    'funsearch': {'_database': ('_islands', '_best_score_per_island', '_best_program_per_island',
                               'num_resets', '_last_reset_time')},
    'clade_ahd': {},
    'deltaevolve': {'_population': ('nodes', 'islands', 'archive_ids', 'projection')},
    'openevolve': {'_database': ('programs', 'island_feature_maps', 'islands', 'current_island',
                                 'island_generations', 'last_migration_generation', 'archive',
                                 'best_program_id', 'island_best_programs', 'last_iteration',
                                 'feature_stats', 'diversity_by_code', 'diversity_reference_set')},
    'hsevo': {},
    'refineevo': {'_planner': ('operator_descriptions', 'history'), '_pool': ('data',)},
    'calm': {},
    'pathwise': {'_population': ('_nodes', '_generation')},
    'shinka_evo': {
        '_archive': ('programs', 'islands', 'archive_ids', 'best_program_id', 'initial_program_id', 'beam_parent_id'),
        '_bandit': ('n_submitted', 'n_completed', 'total_reward', 'last_probs', 'history'),
    },
}


@lru_cache(maxsize=1)
def _classes():
    from core import Function, Program as CoreProgram
    from baselines.funsearch.programs_database import Island, Cluster
    from baselines.calm.pool import HeuristicRecord
    from baselines.calm.prompt import Prompt
    from baselines.pathwise.graph import PathWiseNode, ParentInfo
    from baselines.shinka_evo.population import ShinkaProgram
    from baselines.shinka_evo.bandit import BanditUpdate
    from baselines.openevolve.upstream.database import Program
    from baselines.hsevo.hsevo import Individual
    classes = {cls.__name__: cls for cls in (Function, HeuristicRecord, Prompt, PathWiseNode,
                                            ParentInfo, ShinkaProgram, BanditUpdate, Program, Individual)}
    classes.update(CoreProgram=CoreProgram, Island=Island, Cluster=Cluster)
    return classes


def encode(value, memo=None):
    memo = {} if memo is None else memo
    if isinstance(value, np.generic):
        return encode(value.item(), memo)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {'type': 'float', 'value': str(value)}
    if isinstance(value, np.ndarray):
        return {'type': 'array', 'dtype': str(value.dtype), 'shape': value.shape,
                'value': encode(value.tolist(), memo)}
    if isinstance(value, (list, tuple, set)):
        return {'type': type(value).__name__, 'value': [encode(v, memo) for v in value]}
    if isinstance(value, dict):
        return {'type': 'dict', 'value': [[encode(k, memo), encode(v, memo)] for k, v in value.items()]}
    kind = 'CoreProgram' if type(value) is _classes().get('CoreProgram') else type(value).__name__
    if kind not in _classes() or type(value) is not _classes()[kind]:
        raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')
    if id(value) in memo:
        return {'ref': memo[id(value)]}
    identity = len(memo)
    memo[id(value)] = identity
    return {'type': kind, 'id': identity, 'value': encode(vars(value), memo)}


def decode(value, memo=None):
    memo = {} if memo is None else memo
    if not isinstance(value, dict):
        return value
    if 'ref' in value:
        return memo[value['ref']]
    kind, data = value['type'], value['value']
    if kind == 'float':
        return float(data)
    if kind == 'array':
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biuf':
            raise ValueError('Checkpoint array must be numeric')
        return np.asarray(decode(data, memo), dtype=dtype).reshape(value['shape'])
    if kind == 'dict':
        return {decode(k, memo): decode(v, memo) for k, v in data}
    if kind in ('list', 'tuple', 'set'):
        return {'list': list, 'tuple': tuple, 'set': set}[kind](decode(v, memo) for v in data)
    cls = _classes().get(kind)
    if cls is None:
        raise ValueError(f'Unknown checkpoint type: {kind}')
    fields = decode(data, memo)
    if dataclasses.is_dataclass(cls):
        obj = cls(**{k: fields[k] for k, f in cls.__dataclass_fields__.items() if f.init and k in fields})
    elif kind in ('Island', 'Cluster'):
        obj = cls.__new__(cls)
    else:
        obj = cls(fields['prompt'])
    vars(obj).update(fields)
    memo[value['id']] = obj
    return obj


def read_checkpoint(directory):
    path = Path(directory) / NAME
    if not path.exists():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def save_checkpoint(method, name, boundary):
    profiler = method._profiler
    if getattr(profiler, 'run_dir', None) is None:
        return
    with profiler._artifact_lock:
        directory = profiler.run_dir
        state = {k: getattr(method, k) for k in FIELDS[name]}
        # Completed CALM messages have already been consumed; next round must prepare anew.
        if name == 'calm' and boundary == 'round_complete':
            state['_messages'] = []
        groups = {group: {k: getattr(getattr(method, group), k) for k in fields}
                  for group, fields in GROUPS[name].items()}
        if name == 'funsearch':
            db = method._database
            groups['_database']['_last_reset_time'] = db._clock() - db._last_reset_time
        rng = {'python': random.getstate(), 'numpy': np.random.get_state()}
        if name == 'calm':
            rng['local'] = method._rs.get_state()
        if name == 'shinka_evo':
            rng.update(local=method._rng.getstate(), bandit=method._bandit.rng.getstate())
        if name in ('deltaevolve', 'openevolve'):
            rng['local'] = method._rng.getstate()
        payload = {'method': name, 'boundary': boundary,
                   'budget_used': method._tot_sample_nums, 'candidate_count': profiler._num_samples,
                   'files': {f: (directory / f).stat().st_size if (directory / f).exists() else 0
                             for f in ('events.jsonl', 'programs.jsonl')},
                   'state': encode({'fields': state, 'groups': groups, 'rng': rng})}
        if provenance := getattr(method, '_checkpoint_provenance', None):
            payload['provenance'] = provenance
        with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            os.replace(temporary, directory / NAME)
        finally:
            temporary.unlink(missing_ok=True)


def restore_checkpoint(method, directory, name):
    payload = read_checkpoint(directory)
    if payload is None:
        raise ValueError(f'{name} needs a search checkpoint; old logs do not contain its complete search state')
    if payload['method'] != name:
        raise ValueError('Checkpoint belongs to another method')
    for file, size in payload['files'].items():
        path = Path(directory) / file
        if file not in ('events.jsonl', 'programs.jsonl') or (path.stat().st_size if path.exists() else 0) < size:
            raise ValueError('Checkpoint journals are missing or incomplete')
    state = decode(payload['state'])
    if set(state['fields']) != set(FIELDS[name]) or set(state['groups']) != set(GROUPS[name]):
        raise ValueError('Checkpoint search fields differ from this implementation')
    for group, fields in GROUPS[name].items():
        if set(state['groups'][group]) != set(fields):
            raise ValueError('Checkpoint component fields differ from this implementation')
    for key, value in state['fields'].items():
        setattr(method, key, value)
    for group, fields in state['groups'].items():
        for key, value in fields.items():
            setattr(getattr(method, group), key, value)
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    if name == 'calm':
        method._rs.set_state(state['rng']['local'])
        method._prompts.injected_components = method._injected_components
    if name == 'shinka_evo':
        method._rng.setstate(state['rng']['local'])
        method._bandit.rng.setstate(state['rng']['bandit'])
    if name in ('deltaevolve', 'openevolve'):
        method._rng.setstate(state['rng']['local'])
    if name == 'clade_ahd':
        method._restore_tree()
    if name == 'funsearch':
        db = method._database
        db._last_reset_time = db._clock() - db._last_reset_time
    if provenance := payload.get('provenance'):
        method._checkpoint_provenance = provenance
    return payload

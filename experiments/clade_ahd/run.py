"""Run Clade-AHD under the shared task and generation-budget protocol."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from baselines.clade_ahd import CladeAHD
from baselines.shared.profiler import ProfilerBase
from experiments.infra.base import (
    DEFAULT_BACKEND, RESULTS_ROOT, build_llm_client, build_task, resolve_backend, set_random_seed,
)
from experiments.infra.runner import add_common_run_args, baseline_run_config, run_baseline_experiment
from traceaad.storage import write_json

UPSTREAM = 'Mriya0306/Clade-AHD@ffd566b70070331dbd2759a2cb828543e43da2ac'
FIELDS = ('init_size', 'pop_size', 'num_pseudo_evals', 'backprop_lambda', 'cool_down_enabled',
          'cool_down_beta', 'pruning_enabled', 'pruning_threshold', 'pruning_min_visits')


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    budget: int = 1000
    init_size: int = 4
    pop_size: int = 10
    num_pseudo_evals: int = 10
    backprop_lambda: float = 0.8
    cool_down_enabled: bool = True
    cool_down_beta: float = 1.0
    pruning_enabled: bool = True
    pruning_threshold: float = 0.1
    pruning_min_visits: int = 10
    thinking: bool | None = None
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self):
        return self.experiments_root / 'clade_ahd' / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = Path(self.experiments_root).resolve()
        for name in ('budget', 'init_size', 'pop_size', 'pruning_min_visits', 'output_tokens'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.num_pseudo_evals < 0:
            raise ValueError('num_pseudo_evals must be nonnegative')
        if not 0 <= self.backprop_lambda <= 1 or not 0 <= self.pruning_threshold <= 1:
            raise ValueError('backprop_lambda and pruning_threshold must be in [0, 1]')
        if not math.isfinite(self.cool_down_beta) or self.cool_down_beta < 0:
            raise ValueError('cool_down_beta must be finite and nonnegative')
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError('eval_workers must be positive')


def method_parameters(spec):
    return {name: getattr(spec, name) for name in FIELDS}


def build_method(spec, log_dir):
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(base_url=spec.base_url, model=spec.model, no_proxy=spec.no_proxy,
                           max_tokens=spec.output_tokens, enable_thinking=spec.thinking)
    return CladeAHD(llm, evaluation, ProfilerBase(log_dir.parent), max_sample_nums=spec.budget,
                    **method_parameters(spec))


def run_config(spec, run_dir, run_name):
    params = {**method_parameters(spec), 'max_sample_nums': spec.budget, 'thinking': spec.thinking,
              'upstream': UPSTREAM, 'execution': 'serial_generation_batch_tree_updates',
              'budget_basis': 'Every returned generator response, including parse and duplicate retries; '
                              'thought alignment and transport failures do not consume generation budget.',
              'normalization': 'minmax', 'optimization_type': 'min',
              'max_depth': 10, 'prior_alpha': 1.0, 'prior_beta': 1.0,
              'widening_alpha': 0.5, 'e1_max_parents': 5,
              'leaf_operators': list(CladeAHD.LEAF_OPERATORS),
              'cool_down_progress': 'returned_generations / budget',
              'pruning_schedule': 'upstream_evals % 5 == 0 at completed leaf batches',
              'implementation_notes': 'baselines/clade_ahd/README.md'}
    return baseline_run_config(spec, run_dir, run_name, 'clade_ahd', params,
                               llm_options=dict(enable_thinking=spec.thinking))


def write_run_config(spec, run_dir, run_name):
    write_json(run_dir / 'run_config.json', run_config(spec, run_dir, run_name))


def run_experiment(spec):
    return run_baseline_experiment(spec, write_config=write_run_config, build_method=build_method,
                                   description=f'clade_ahd budget={spec.budget}, pop={spec.pop_size}')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384)
    for name in FIELDS:
        default = RunSpec.__dataclass_fields__[name].default
        options = {'action': argparse.BooleanOptionalAction} if isinstance(default, bool) else {'type': type(default)}
        parser.add_argument('--' + name.replace('_', '-'), default=default, **options)
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    dry_run = args.pop('dry_run')
    spec = RunSpec(**args)
    if dry_run:
        name = spec.run_name or 'preview'
        print(json.dumps(run_config(spec, spec.experiment_root / name, name), ensure_ascii=False, indent=2))
        return
    run_experiment(spec)


if __name__ == '__main__':
    main()

"""Run the HSEvo baseline under the shared task protocol."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from baselines.hsevo import HSEvo
from baselines.hsevo.hsevo import HS_TRIES
from baselines.shared.profiler import ProfilerBase
from experiments.infra.base import (
    DEFAULT_BACKEND, RESULTS_ROOT, build_llm_client, build_task, resolve_backend, set_random_seed,
)
from experiments.infra.runner import add_common_run_args, baseline_run_config, run_baseline_experiment
from traceaad.storage import write_json

UPSTREAM = "datphamvn/HSEvo@f3afffabc2b458fddf70a42a8678758039671297"
# cfg/config.yaml: the GA loop and harmony search parameters.
FIELDS = ("pop_size", "init_pop_size", "mutation_rate", "hm_size", "hmcr", "par", "bandwidth", "max_iter")


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    budget: int = 1000
    pop_size: int = 10
    init_pop_size: int = 30
    mutation_rate: float = 0.5
    hm_size: int = 5
    hmcr: float = 0.7
    par: float = 0.5
    bandwidth: float = 0.2
    max_iter: int = 5
    thinking: bool | None = None
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self):
        return self.experiments_root / "hsevo" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = Path(self.experiments_root).resolve()
        for name in ("budget", "pop_size", "init_pop_size", "hm_size", "output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_iter < 0 or not all(0 <= getattr(self, n) <= 1 for n in ("mutation_rate", "hmcr", "par")):
            raise ValueError("Invalid harmony search iterations or rates")
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError("eval_workers must be positive")


def method_parameters(spec):
    return {key: getattr(spec, key) for key in FIELDS}


def build_method(spec, log_dir):
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(base_url=spec.base_url, model=spec.model, no_proxy=spec.no_proxy,
                           max_tokens=spec.output_tokens, enable_thinking=spec.thinking)
    return HSEvo(llm, evaluation, ProfilerBase(log_dir.parent), max_sample_nums=spec.budget,
                 **method_parameters(spec))


def run_config(spec, run_dir, run_name):
    params = {**method_parameters(spec), "max_sample_nums": spec.budget, "thinking": spec.thinking,
              "upstream": UPSTREAM, "harmony_search_tries": HS_TRIES, "execution": "serial",
              "budget_basis": ("Every function evaluation counts, including the seed, unparsed responses "
                               "and harmony-search variants; reflection and parameter-extraction calls do not."),
              "selection_fallback": "tied parent pairs only when every selectable score is tied",
              "implementation_notes": "baselines/hsevo/README.md"}
    return baseline_run_config(spec, run_dir, run_name, "hsevo", params,
                               llm_options=dict(enable_thinking=spec.thinking))


def write_run_config(spec, run_dir, run_name):
    write_json(run_dir / "run_config.json", run_config(spec, run_dir, run_name))


def run_experiment(spec):
    return run_baseline_experiment(spec, write_config=write_run_config, build_method=build_method,
                                   description=f"hsevo budget={spec.budget}, pop={spec.pop_size}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384)
    for name in FIELDS:
        default = RunSpec.__dataclass_fields__[name].default
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    dry_run = args.pop("dry_run")
    spec = RunSpec(**args)
    if dry_run:
        print(json.dumps(run_config(spec, spec.experiment_root / (spec.run_name or "preview"),
                                    spec.run_name or "preview"), ensure_ascii=False, indent=2))
        return
    run_experiment(spec)


if __name__ == "__main__":
    main()

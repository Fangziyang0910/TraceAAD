"""Run the OpenEvolve baseline under the shared task protocol."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from baselines.openevolve import OpenEvolve
from baselines.openevolve.openevolve import ENSEMBLE_WEIGHTS, MAX_CODE_LENGTH
from baselines.shared.profiler import ProfilerBase
from experiments.infra.base import (
    DEFAULT_BACKEND, RESULTS_ROOT, build_llm_client, build_task, llm_payload,
    resolve_backend, set_random_seed,
)
from experiments.infra.runner import add_common_run_args, baseline_run_config, run_baseline_experiment
from traceaad.storage import write_json

UPSTREAM = "algorithmicsuperintelligence/openevolve@34427bae6239063d46be0d38e0fabe11279d05aa"


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    budget: int = 1000
    num_islands: int = 5
    population_size: int = 1000
    archive_size: int = 100
    migration_interval: int = 50
    migration_rate: float = 0.1
    num_top_programs: int = 3
    num_diverse_programs: int = 2
    secondary_model: str | None = None
    thinking: bool | None = None
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self):
        return self.experiments_root / "openevolve" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = Path(self.experiments_root).resolve()
        for name in ("budget", "num_islands", "population_size", "archive_size",
                     "migration_interval", "num_top_programs", "output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_diverse_programs < 0 or not 0 <= self.migration_rate <= 1:
            raise ValueError("Invalid diverse program count or migration rate")
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError("eval_workers must be positive")


def method_parameters(spec):
    fields = ("num_islands", "population_size", "archive_size", "migration_interval",
              "migration_rate", "num_top_programs", "num_diverse_programs")
    return {key: getattr(spec, key) for key in fields}


def llm_options(spec):
    return dict(enable_thinking=spec.thinking)


def build_method(spec, log_dir):
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    options = dict(base_url=spec.base_url, no_proxy=spec.no_proxy,
                   max_tokens=spec.output_tokens, **llm_options(spec))
    primary = build_llm_client(model=spec.model, **options)
    secondary = build_llm_client(model=spec.secondary_model, **options) if spec.secondary_model else None
    return OpenEvolve(primary, evaluation, ProfilerBase(log_dir.parent), max_sample_nums=spec.budget,
                      secondary_llm=secondary, random_seed=spec.seed, **method_parameters(spec))


def run_config(spec, run_dir, run_name):
    params = {**method_parameters(spec), "max_sample_nums": spec.budget,
              "secondary_model": spec.secondary_model, "thinking": spec.thinking,
              "upstream": UPSTREAM,
              "parent_selection": {"exploration": 0.2, "exploitation": 0.7, "weighted": 0.1},
              "elite_selection_ratio": 0.1, "feature_dimensions": ["complexity", "diversity"],
              "feature_bins": 10, "diversity_reference_size": 20,
              "diff_based_evolution": True, "max_code_length": MAX_CODE_LENGTH,
              "island_order": "round_robin", "execution": "serial",
              "score_mapping": "combined_score = -raw_objective; failed evaluation = -inf",
              "budget_basis": "Seed evaluation and every returned generation count; transport failures do not.",
              "implementation_notes": "baselines/openevolve/README.md"}
    ensemble = [{"model": spec.model, "weight": ENSEMBLE_WEIGHTS[0] if spec.secondary_model else 1.0}]
    if spec.secondary_model:
        ensemble.append({"model": spec.secondary_model, "weight": ENSEMBLE_WEIGHTS[1],
                         "llm": llm_payload(base_url=spec.base_url, model=spec.secondary_model,
                             no_proxy=spec.no_proxy, max_tokens=spec.output_tokens, **llm_options(spec))})
    return baseline_run_config(spec, run_dir, run_name, "openevolve", params,
                               llm_options=llm_options(spec), extra={"llm_ensemble": ensemble})


def write_run_config(spec, run_dir, run_name):
    write_json(run_dir / "run_config.json", run_config(spec, run_dir, run_name))


def run_experiment(spec):
    return run_baseline_experiment(spec, write_config=write_run_config, build_method=build_method,
                                   description=f"openevolve budget={spec.budget}, islands={spec.num_islands}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384)
    for name in ("num_islands", "population_size", "archive_size", "migration_interval",
                 "num_top_programs", "num_diverse_programs"):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=RunSpec.__dataclass_fields__[name].default)
    parser.add_argument("--migration-rate", type=float, default=0.1)
    parser.add_argument("--secondary-model", help="Optional 20%% ensemble model on the same endpoint")
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

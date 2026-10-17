"""Run the paper-based DeltaEvolve baseline under the shared task protocol."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from baselines.deltaevolve import DeltaEvolve
from baselines.shared.profiler import ProfilerBase
from experiments.shinka_evo.embedding import embedding_settings
from .embedding import CHUNK_BYTES, CodeEmbedding
from experiments.infra.base import (
    DEFAULT_BACKEND, RESULTS_ROOT, build_llm_client, build_task, llm_payload,
    resolve_backend, set_random_seed,
)
from experiments.infra.runner import add_common_run_args, baseline_run_config, run_baseline_experiment
from traceaad.storage import write_json


@dataclass(slots=True)
class RunSpec:
    task: str
    backend: str = DEFAULT_BACKEND
    base_url: str | None = None
    model: str | None = None
    no_proxy: str | None = None
    budget: int = 1000
    population_size: int = 40
    archive_size: int = 20
    num_islands: int = 3
    top_k: int = 3
    diverse_m: int = 2
    recent_window: int = 10
    migration_interval: int = 10
    migration_rate: float = 0.1
    secondary_model: str | None = None
    thinking: bool | None = None
    eval_workers: int | None = None
    output_tokens: int = 8192
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self):
        return self.experiments_root / "deltaevolve" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = Path(self.experiments_root).resolve()
        for name in ("budget", "population_size", "archive_size", "num_islands", "recent_window",
                     "migration_interval", "output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if min(self.top_k, self.diverse_m) < 0 or not 0 <= self.migration_rate <= 1:
            raise ValueError("Invalid history count or migration rate")
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError("eval_workers must be positive")


def method_parameters(spec):
    fields = ("population_size", "archive_size", "num_islands", "top_k", "diverse_m",
              "recent_window", "migration_interval", "migration_rate")
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
    return DeltaEvolve(primary, evaluation, ProfilerBase(log_dir.parent),
                       max_sample_nums=spec.budget, embedding_fn=CodeEmbedding(),
                       secondary_llm=secondary, random_seed=spec.seed, **method_parameters(spec))


def run_config(spec, run_dir, run_name):
    params = {**method_parameters(spec), "max_sample_nums": spec.budget,
              "secondary_model": spec.secondary_model, "thinking": spec.thinking,
              "parent_selection": {"elite": 0.1, "exploration": 0.2, "exploitation": 0.7,
                                   "weights": "inverse_rank", "elite_pool_fraction": 0.1},
              "grid": {"bins": 10, "dimensions": 2, "projection_seed": 0,
                       "descriptors": "tanh(sqrt(d) * projection @ normalized_code_embedding)"},
              "embedding": {**embedding_settings(), "chunk_bytes": CHUNK_BYTES,
                            "aggregation": "utf8_byte_weighted_mean"},
              "history_feedback": "qualitative_parent_relative_change",
              "parent_feedback": "negative_raw_objective",
              "budget_basis": "Seed evaluation and every returned generation count; transport failures do not. Paper used 100 iterations; shared default is 1000 attempts.",
              "implementation_notes": "baselines/deltaevolve/README.md"}
    ensemble = [{"model": spec.model, "probability": 0.8 if spec.secondary_model else 1.0}]
    if spec.secondary_model:
        ensemble.append({"model": spec.secondary_model, "probability": 0.2,
                         "llm": llm_payload(base_url=spec.base_url, model=spec.secondary_model,
                             no_proxy=spec.no_proxy, max_tokens=spec.output_tokens, **llm_options(spec))})
    return baseline_run_config(spec, run_dir, run_name, "deltaevolve", params,
                               llm_options=llm_options(spec), extra={"llm_ensemble": ensemble})


def write_run_config(spec, run_dir, run_name):
    write_json(run_dir / "run_config.json", run_config(spec, run_dir, run_name))


def run_experiment(spec):
    return run_baseline_experiment(spec, write_config=write_run_config, build_method=build_method,
                                   description=f"deltaevolve budget={spec.budget}, recent_window={spec.recent_window}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=8192)
    for name in ("population_size", "archive_size", "num_islands", "top_k", "diverse_m",
                 "recent_window", "migration_interval"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=RunSpec.__dataclass_fields__[name].default)
    parser.add_argument("--migration-rate", type=float, default=0.1)
    parser.add_argument("--secondary-model", help="Optional 20%% reasoning model on the same endpoint")
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

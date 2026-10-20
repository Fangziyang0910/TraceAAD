"""Run the RefineEvo baseline under the shared task protocol."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from baselines.refineevo import RefineEvo
from baselines.shared.profiler import ProfilerBase
from experiments.infra.base import (
    DEFAULT_BACKEND, RESULTS_ROOT, build_llm_client, build_task, resolve_backend, set_random_seed,
)
from experiments.infra.runner import add_common_run_args, baseline_run_config, run_baseline_experiment
from experiments.shinka_evo.embedding import CodeEmbedding, embedding_settings
from traceaad.storage import write_json

UPSTREAM = "samwu-learn/RefineEvo@7de015d5230e32b8b85fa83a3ecc55d117489138"
# cfg/config.yaml population sizes and the paper's top-k retrieval.
FIELDS = ("pop_size", "init_pop_size", "top_k")


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
    top_k: int = 3
    thinking: bool | None = None
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = RESULTS_ROOT

    @property
    def experiment_root(self):
        return self.experiments_root / "refineevo" / self.task

    def __post_init__(self):
        profile = resolve_backend(self.backend, self.base_url, self.model, self.no_proxy)
        self.base_url, self.model, self.no_proxy = profile.base_url, profile.model, profile.no_proxy
        self.experiments_root = Path(self.experiments_root).resolve()
        for name in ("budget", "pop_size", "top_k", "output_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.init_pop_size < self.pop_size:
            raise ValueError("init_pop_size must be at least pop_size")
        if self.eval_workers is not None and self.eval_workers < 1:
            raise ValueError("eval_workers must be positive")


def method_parameters(spec):
    return {key: getattr(spec, key) for key in FIELDS}


def build_method(spec, log_dir):
    set_random_seed(spec.seed)
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(base_url=spec.base_url, model=spec.model, no_proxy=spec.no_proxy,
                           max_tokens=spec.output_tokens, enable_thinking=spec.thinking)
    return RefineEvo(llm, evaluation, CodeEmbedding(), ProfilerBase(log_dir.parent),
                     max_sample_nums=spec.budget, **method_parameters(spec))


def run_config(spec, run_dir, run_name):
    params = {**method_parameters(spec), "max_sample_nums": spec.budget, "thinking": spec.thinking,
              "upstream": UPSTREAM, "execution": "serial",
              "budget_basis": ("Every returned generator response counts, including unparsed and duplicate "
                               "responses that upstream retries; reflector, planner and embedding calls do not."),
              "batch_experience": ("slots of one operator batch retrieve from the pool as it stood at the batch "
                                   "start; experience writes and score updates follow after the batch"),
              "embedding": embedding_settings(),
              "implementation_notes": "baselines/refineevo/README.md"}
    return baseline_run_config(spec, run_dir, run_name, "refineevo", params,
                               llm_options=dict(enable_thinking=spec.thinking))


def write_run_config(spec, run_dir, run_name):
    write_json(run_dir / "run_config.json", run_config(spec, run_dir, run_name))


def run_experiment(spec):
    return run_baseline_experiment(spec, write_config=write_run_config, build_method=build_method,
                                   description=f"refineevo budget={spec.budget}, pop={spec.pop_size}")


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

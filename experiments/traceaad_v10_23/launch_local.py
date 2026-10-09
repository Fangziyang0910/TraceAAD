"""Launch V10.23 on the local host with the two server3 model endpoints."""

from experiments.infra.search_host_launch import plan_for as _plan_for, main as launch
from traceaad.v10_23 import Config
from traceaad.v10_23.config import EXPERIMENT
from benchmarks.tasks import CO_TASKS, INSTANCE_SECONDS

MODULE = "experiments.traceaad_v10_23.run"
BACKEND_NAMES = ('server3', 'server3b')


def plan_for(batch, *, eval_workers=1, eval_timeout_seconds=INSTANCE_SECONDS):
    Config(eval_workers=eval_workers, eval_timeout_seconds=eval_timeout_seconds)
    return _plan_for(batch, EXPERIMENT, MODULE, BACKEND_NAMES, tasks=CO_TASKS,
                     eval_workers=eval_workers, eval_timeout_seconds=eval_timeout_seconds)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1023", "local", BACKEND_NAMES, argv)


if __name__ == "__main__":
    main()

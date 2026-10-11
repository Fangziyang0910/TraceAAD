"""Plan or launch six tasks × three repetitions on this host."""

from benchmarks.tasks import CO_TASKS
from experiments.infra.search_host_launch import plan_for as _plan_for, main as launch
from traceaad.v10_25 import Config
from traceaad.v10_25.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_25.run"
BACKENDS = ("server3", "server3b")


def plan_for(batch, *, eval_workers=1, eval_timeout_seconds=None):
    Config(eval_workers=eval_workers, **({"eval_timeout_seconds": eval_timeout_seconds}
                                       if eval_timeout_seconds is not None else {}))
    return _plan_for(batch, EXPERIMENT, MODULE, BACKENDS, tasks=CO_TASKS,
                     eval_workers=eval_workers, eval_timeout_seconds=eval_timeout_seconds)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1025", "local", BACKENDS, argv)


if __name__ == "__main__":
    main()

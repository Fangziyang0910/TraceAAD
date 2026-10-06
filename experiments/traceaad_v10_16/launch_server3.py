"""Launch V10.16 on server3."""

from experiments.infra.search_host_launch import plan_for as _plan_for, main as launch
from traceaad.v10_16 import Config
from traceaad.v10_16.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_16.run"
BACKEND_NAMES = ('server3', 'server3b')


def plan_for(batch):
    return _plan_for(batch, EXPERIMENT, MODULE, BACKEND_NAMES)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1016", "server3", BACKEND_NAMES, argv)


if __name__ == "__main__":
    main()

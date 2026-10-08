"""Launch V10.21 on the local host with the two server3 model endpoints."""

from experiments.infra.search_host_launch import plan_for as _plan_for, main as launch
from traceaad.v10_21 import Config
from traceaad.v10_21.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_21.run"
BACKEND_NAMES = ('server3', 'server3b')


def plan_for(batch):
    return _plan_for(batch, EXPERIMENT, MODULE, BACKEND_NAMES)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1021", "local", BACKEND_NAMES, argv)


if __name__ == "__main__":
    main()

"""Launch V10.20 on local: the comparison baseline is V10.18/V10.19 on the same host."""

from experiments.infra.search_host_launch import plan_for as _plan_for, main as launch
from traceaad.v10_20 import Config
from traceaad.v10_20.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_20.run"
BACKEND_NAMES = ('server3', 'server3b')


def plan_for(batch):
    return _plan_for(batch, EXPERIMENT, MODULE, BACKEND_NAMES)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1020", "local", BACKEND_NAMES, argv)


if __name__ == "__main__":
    main()

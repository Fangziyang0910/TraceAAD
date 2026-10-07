"""Prepare V10.20 runs using the routes in an existing batch."""

from experiments.infra.search_launch import build_plan as _build_plan, main as launch
from traceaad.v10_20 import Config
from traceaad.v10_20.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_20.run"


def build_plan(previous, batch, experiment=EXPERIMENT):
    return _build_plan(previous, batch, experiment, MODULE)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1020", argv)


if __name__ == "__main__":
    main()

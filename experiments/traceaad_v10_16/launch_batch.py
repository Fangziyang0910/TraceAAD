"""Prepare V10.16 runs using the routes in an existing batch."""

from experiments.infra.search_launch import (evaluation_limits, served_models,
    build_plan as _build_plan, main as launch)
from traceaad.v10_16 import Config
from traceaad.v10_16.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_16.run"


def build_plan(previous, batch, experiment=EXPERIMENT):
    return _build_plan(previous, batch, experiment, MODULE)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1016", argv)


if __name__ == "__main__":
    main()

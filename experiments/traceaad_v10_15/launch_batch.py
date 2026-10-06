"""Prepare V10.15 runs using the routes in an existing batch."""

from experiments.infra.search_launch import (evaluation_limits, served_models,
    build_plan as _build_plan, main as launch)
from traceaad.v10_15 import Config
from traceaad.v10_15.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_15.run"


def build_plan(previous, batch, experiment=EXPERIMENT):
    return _build_plan(previous, batch, experiment, MODULE)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1015", argv)


if __name__ == "__main__":
    main()

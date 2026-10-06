"""Prepare V10.19 runs using the routes in an existing batch."""

from experiments.infra.search_launch import build_plan as _build_plan, main as launch
from traceaad.v10_19 import Config
from traceaad.v10_19.config import EXPERIMENT

MODULE = "experiments.traceaad_v10_19.run"


def build_plan(previous, batch, experiment=EXPERIMENT):
    return _build_plan(previous, batch, experiment, MODULE)


def main(argv=None):
    return launch(Config, EXPERIMENT, MODULE, "v1019", argv)


if __name__ == "__main__":
    main()

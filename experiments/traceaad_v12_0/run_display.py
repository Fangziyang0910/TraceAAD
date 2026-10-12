"""Run the V12.0 display diagnostic: a found earlier program is shown with probability 0.5."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v12_0 import DisplayConfig
from traceaad.v12_0.config import EXPERIMENT
from .run import LinkedSearch


def build_parser():
    return _build_parser(EXPERIMENT, DisplayConfig)


def main(argv=None):
    return run(LinkedSearch, DisplayConfig, EXPERIMENT, "V12.0 display diagnostic: earlier program shown at random", argv)


if __name__ == "__main__":
    main()

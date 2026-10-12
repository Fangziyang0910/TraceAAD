"""Run the V12.0 base arm: the same search with linking off (V10.17 under the current protocol)."""

from experiments.infra.search_run import build_parser as _build_parser, main as run
from traceaad.v12_0 import BaseConfig, TraceAADV120
from traceaad.v12_0.config import EXPERIMENT


def build_parser():
    return _build_parser(EXPERIMENT, BaseConfig)


def main(argv=None):
    return run(TraceAADV120, BaseConfig, EXPERIMENT, "V12.0 base arm: linking off", argv)


if __name__ == "__main__":
    main()

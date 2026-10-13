"""Run the V12.0 base arm: the same search with linking off."""

from experiments.infra.search_run import main as run
from traceaad.v12_0 import BaseConfig, TraceAADV120
from traceaad.v12_0.config import EXPERIMENT


def main(argv=None):
    return run(TraceAADV120, BaseConfig, EXPERIMENT, "V12.0 base arm: linking off", argv)


if __name__ == "__main__":
    main()

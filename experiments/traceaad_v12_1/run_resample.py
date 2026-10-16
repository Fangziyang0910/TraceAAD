"""Run the V12.1 resample arm: completions are independent implementations from the start."""

from experiments.infra.search_run import main as run
from traceaad.v12_1 import ResampleConfig
from traceaad.v12_1.config import EXPERIMENT
from .run import VerifiedSearch


def main(argv=None):
    return run(
        VerifiedSearch,
        ResampleConfig,
        EXPERIMENT,
        "V12.1 resample arm: independent implementations",
        argv,
    )


if __name__ == "__main__":
    main()

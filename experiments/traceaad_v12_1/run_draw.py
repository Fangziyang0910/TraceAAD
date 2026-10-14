"""Run the V12.1 randomized arm: an exploration that needs completion is completed with probability 0.5."""

from experiments.infra.search_run import main as run
from traceaad.v12_1 import DrawConfig
from traceaad.v12_1.config import EXPERIMENT
from .run import VerifiedSearch


def main(argv=None):
    return run(VerifiedSearch, DrawConfig, EXPERIMENT, "V12.1 randomized arm: completion at random", argv)


if __name__ == "__main__":
    main()

"""Run unchanged V10.17 search under the current six-task evaluation protocol."""

from experiments.infra.search_run import main as run
from traceaad.v10_17.config import EXPERIMENT
from .current_protocol import Config, CurrentProtocolV1017


def main(argv=None):
    return run(CurrentProtocolV1017, Config, EXPERIMENT, __doc__, argv)


if __name__ == "__main__":
    main()

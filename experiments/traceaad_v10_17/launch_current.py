"""Launch V10.17 with current tasks, evaluation and final selection."""

from experiments.infra.search_host_launch import main as launch
from traceaad.v10_17.config import EXPERIMENT
from .current_protocol import Config


def main(argv=None):
    return launch(Config, EXPERIMENT, "experiments.traceaad_v10_17.run_current", "v1017",
                  "local", ("server3", "server3b"), argv)


if __name__ == "__main__":
    main()

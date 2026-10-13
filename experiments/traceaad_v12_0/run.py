"""Run V12.0 (re-proposal linking) under the shared six-task protocol."""

from experiments.infra.search_run import main as run
from experiments.shinka_evo.embedding import CodeEmbedding
from traceaad.v12_0 import Config, TraceAADV120
from traceaad.v12_0.config import EXPERIMENT


class LinkedSearch(TraceAADV120):
    """V12.0 with the text-embedding service used to rank earlier Designs."""

    def __init__(self, **kwargs):
        super().__init__(embed=CodeEmbedding(), **kwargs)


def main(argv=None):
    return run(LinkedSearch, Config, EXPERIMENT, "V12.0: search with re-proposal linking", argv)


if __name__ == "__main__":
    main()

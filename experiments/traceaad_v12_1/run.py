"""Run V12.1 (design verification) under the shared six-task protocol."""

from experiments.infra.search_run import main as run
from experiments.shinka_evo.embedding import CodeEmbedding
from traceaad.v12_1 import Config, TraceAADV121
from traceaad.v12_1.config import EXPERIMENT


class VerifiedSearch(TraceAADV121):
    """V12.1 with the text-embedding service used to rank earlier Designs."""

    def __init__(self, **kwargs):
        super().__init__(embed=CodeEmbedding(), **kwargs)


def main(argv=None):
    return run(VerifiedSearch, Config, EXPERIMENT, "V12.1: search with design verification", argv)


if __name__ == "__main__":
    main()

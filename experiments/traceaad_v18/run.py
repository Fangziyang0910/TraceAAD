"""Run V18 with shared task evaluation and request-based search budgets."""

from experiments.infra.search_run import main as run


def main(argv=None):
    return run(argv, version="v18")


if __name__ == "__main__":
    main()

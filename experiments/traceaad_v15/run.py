"""Run V15 with shared evaluation and resource accounting."""
from experiments.infra.search_run import main as run


def main(argv=None):
    return run(argv, version="v15")


if __name__ == "__main__":
    main()

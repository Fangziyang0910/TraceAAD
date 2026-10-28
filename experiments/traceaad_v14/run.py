"""Run V14 with shared evaluation and resource accounting."""
from experiments.infra.search_run import main as run


def main(argv=None):
    return run(argv, version="v14")


if __name__ == "__main__":
    main()

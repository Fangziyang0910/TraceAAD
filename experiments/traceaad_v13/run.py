"""Run V13 with the shared training/evaluation and result format."""
from experiments.infra.search_run import main as run


def main(argv=None):
    return run(argv, version='v13')

if __name__ == '__main__':
    main()

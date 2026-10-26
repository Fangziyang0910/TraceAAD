"""Run V13.1 with shared evaluation and resource accounting."""
from experiments.infra.search_run import main as run


def main(argv=None):
    return run(argv,version='v13_1')


if __name__=='__main__': main()

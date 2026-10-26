"""Preview or launch a V13 batch; preview is the default."""
from experiments.infra.search_host_launch import main as launch


def main(argv=None):
    return launch(argv, version='v13')

if __name__ == '__main__':
    main()

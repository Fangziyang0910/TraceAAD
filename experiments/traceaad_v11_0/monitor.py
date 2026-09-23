"""V11.0 compatibility entry point for the shared training monitor."""

from core.training_monitor import *  # noqa: F401,F403
from core.training_monitor import main

if __name__ == "__main__":
    main()

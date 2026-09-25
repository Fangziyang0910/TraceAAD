"""V10.13 training monitor."""

from __future__ import annotations

import sys
from pathlib import Path

from core.training_monitor import main


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "results"
    if "--results-dir" not in sys.argv:
        sys.argv[1:1] = ["--results-dir", str(root)]
    main()

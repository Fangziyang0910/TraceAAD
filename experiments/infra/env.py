"""Load repository-root ``.env`` into ``os.environ`` (without overriding)."""

from __future__ import annotations

import os
from pathlib import Path

_LOADED = False
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_repo_dotenv(*, override: bool = False) -> Path | None:
    """Parse ``<repo>/.env`` into the process environment.

    Returns the path loaded, or ``None`` if missing. Existing environment
    variables win unless ``override=True``.
    """
    global _LOADED
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return None
    if _LOADED and not override:
        return env_path

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
    _LOADED = True
    return env_path


def resolve_llm_api_key(*, base_url: str | None = None, default: str = "EMPTY") -> str:
    """Resolve API key: ``LLM_API_KEY``, else host-specific key, else default."""
    load_repo_dotenv()
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    # Both B3 vLLM services (server3 :8000/:8001 and server1 :8080) share one key.
    if base_url and ("222.201.145.6" in base_url or "222.201.145.8" in base_url):
        return os.environ.get("SERVER3_API_KEY", default)
    if base_url and "x5m5x.com" in base_url:
        return os.environ.get("X5M5X_API_KEY", default)
    return default

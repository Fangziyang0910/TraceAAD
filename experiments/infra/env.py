"""Load local model credentials without overwriting the process environment."""

import os
from pathlib import Path

_LOADED = False
_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_repo_dotenv():
    global _LOADED
    path = _REPO_ROOT / ".env"
    if _LOADED or not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip():
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    _LOADED = True


def resolve_llm_api_key(*, base_url=None):
    load_repo_dotenv()
    if key := os.environ.get("LLM_API_KEY"):
        return key
    for host, variable in (("222.201.145.6", "SERVER3_API_KEY"),
                           ("222.201.145.8", "SERVER3_API_KEY"),
                           ("183.36.243.124", "DEEPSEEK_API_KEY")):
        if base_url and host in base_url:
            return os.environ.get(variable, "EMPTY")
    return "EMPTY"

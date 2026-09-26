"""Shared backend checks and tmux launch operations."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from experiments.infra.base import (
    BACKENDS,
    REPO_ROOT,
    BackendName,
)
from .env import resolve_llm_api_key

def check_backends(backends: Iterable[BackendName]) -> None:
    """Read each backend's /v1/models once; abort if any is unreachable."""
    unique_backends = {b: BACKENDS[b] for b in backends}
    for name, profile in unique_backends.items():
        url = profile.base_url.rstrip("/") + "/models"
        api_key = resolve_llm_api_key(base_url=profile.base_url)
        headers = (
            {}
            if not api_key or api_key == "EMPTY"
            else {"Authorization": f"Bearer {api_key}"}
        )
        request = Request(url, headers=headers)
        opener = build_opener(ProxyHandler({}))
        try:
            with opener.open(request, timeout=15) as response:
                ok = response.status == 200
        except Exception as exc:
            raise RuntimeError(f"backend {name} unreachable at {url}: {exc}") from exc
        if not ok:
            raise RuntimeError(f"backend {name} returned {response.status} at {url}")
        print(f"backend {name} reachable", flush=True)


def get_summary_status(run_dir: Path) -> str | None:
    """Return status from logs/run_summary.json or logs/summary.json."""
    for filename in ("run_summary.json", "summary.json"):
        path = run_dir / "logs" / filename
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                status = payload.get("status")
                if isinstance(status, str):
                    return status
            except json.JSONDecodeError:
                pass
    return None


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def launch_command(session: str, command: Sequence[str]) -> None:
    if is_session_alive(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT),
         shlex.join(command)], cwd=REPO_ROOT, check=True,
    )


def is_session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0



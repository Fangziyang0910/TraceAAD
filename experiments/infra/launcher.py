"""Standard batch launcher and watchdog.

Provides multi-backend rotation, backend reachability checks, tmux session
management, and automatic in-place relaunch (resume) until all runs finish.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from experiments.infra.base import (
    BACKENDS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    BackendName,
    LaunchItem,
    launch_items,
)
from .env import resolve_llm_api_key

DEFAULT_MAX_ATTEMPTS = 5


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


def is_session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def live_session_name(base_session: str, attempt: int) -> str:
    return base_session if attempt == 1 else f"{base_session}_r{attempt}"


def is_any_attempt_alive(item: LaunchItem, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> bool:
    if is_session_alive(item.session):
        return True
    return any(
        is_session_alive(live_session_name(item.session, retry))
        for retry in range(2, max_attempts + 1)
    )


def tmux_launch(item: LaunchItem, session: str) -> None:
    printable = shlex.join(item.command())
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-c",
            str(REPO_ROOT),
            printable,
        ],
        check=True,
    )


def ensure_launchable(items: list[LaunchItem], max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> None:
    """Guard against duplicate tmux sessions."""
    sessions = [item.session for item in items]
    if len(sessions) != len(set(sessions)):
        raise ValueError("tmux session names must be unique")
    alive = [item.session for item in items if is_any_attempt_alive(item, max_attempts)]
    if alive:
        raise RuntimeError(f"tmux sessions already exist: {alive}")


def build_batch_parser(
    description: str = "Launch a batch experiment.",
    default_session_prefix: str = "batch",
    default_watch_interval: int = 120,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--session-prefix", default=default_session_prefix)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=int, default=default_watch_interval)
    return parser



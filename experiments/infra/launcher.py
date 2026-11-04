"""Read a model endpoint once and execute a concrete tmux batch plan."""

from dataclasses import dataclass
import shlex
import subprocess
from pathlib import Path

from .base import REPO_ROOT


def is_session_alive(session):
    return (
        subprocess.run(
            ["tmux", "has-session", "-t", "=" + session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )


def launch_command(session, command):
    if is_session_alive(session):
        raise RuntimeError(f"tmux session already exists: {session}")
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT), shlex.join(command)],
        cwd=REPO_ROOT,
        check=True,
    )


@dataclass(frozen=True, slots=True)
class LaunchItem:
    task: str
    repeat: int
    backend: str
    session: str
    run_name: str
    run_dir: Path
    seed: int
    module: str
    extra_args: tuple[str, ...] = ()

    def command(self) -> tuple[str, ...]:
        return (
            str(REPO_ROOT / ".venv/bin/python"),
            "-m",
            self.module,
            "--task",
            self.task,
            "--backend",
            self.backend,
            "--repeat",
            str(self.repeat),
            "--seed",
            str(self.seed),
            "--run-name",
            self.run_name,
            *self.extra_args,
        )


def pending_items(plan):
    return [item for item in plan if not item.run_dir.exists() and not is_session_alive(item.session)]


def validate_new_items(items):
    if len({item.session for item in items}) != len(items) or len({item.run_dir for item in items}) != len(items):
        raise ValueError("run directories and tmux sessions must be unique")
    for item in items:
        if item.run_dir.exists() or is_session_alive(item.session):
            raise ValueError(f"existing run or session: {item.run_name}")

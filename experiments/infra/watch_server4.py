"""Sync and check the server4 batches that still have primary runs to complete.

No batch or version is named here. A run is complete when its local summary.json
has a terminal status and, if it finished, its local heldout.json exists. A batch
is active when its manifest is on server4 and one of its runs in
experiments_result/primary_runs.json is not complete here. So a batch is synced
and checked once its runs are in the primary list, and a replaced batch is
dropped with them.

check (default; traceaad-watch.timer): failed user services, the local model and
its tunnel to server4, and for each active batch the remote run state, its
search sessions and whether the local copy keeps up. New problems are sent as
desktop notifications; the last check is saved in ~/.cache/traceaad/server4-watch.json.

sync (traceaad-results-sync.service): copy each active batch here every minute.
"""

import argparse
from collections import defaultdict
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from experiments.infra import remote
from traceaad.storage import read_json, read_object

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments_result"
HOST = "B3-server4"
REMOTE = "/home/fzy/code/LLM4AD/TraceAAD"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", HOST]
STATE = Path.home() / ".cache/traceaad/server4-watch.json"
# Candidates a local run may trail its remote run before the sync counts as behind.
LAG = 20


def command(args, timeout=40):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout).stdout


def complete(run):
    status = read_object(run / "summary.json").get("status")
    return status in remote.TERMINAL and (status != "finished" or (run / "heldout.json").exists())


def remote_manifests():
    manifests = defaultdict(list)
    for path in command(SSH + [f"cd {REMOTE}/experiments_result && ls -1 */batch_*.json"]).split():
        experiment, name = path.split("/")
        manifests[experiment].append(name.removeprefix("batch_").removesuffix(".json"))
    return manifests


def active_batches(manifests):
    """Sorted (experiment, batch) pairs; a run belongs to the batch whose name is the longest prefix of its name."""
    found = set()
    for experiment, runs in read_json(RESULTS / "primary_runs.json", {}).get("methods", {}).items():
        for relative in runs:
            name = relative.split("/")[-1]
            batches = [b for b in manifests.get(experiment, ()) if name.startswith(b + "_")]
            if batches and not complete(RESULTS / experiment / relative):
                found.add((experiment, max(batches, key=len)))
    return sorted(found)


def sync():
    remote.HOST, remote.REMOTE = HOST, REMOTE
    while True:
        try:
            for experiment, batch in active_batches(remote_manifests()):
                remote.pull(experiment, batch)
                plan = read_object(RESULTS / experiment / f"batch_{batch}.json").get("plan", [])
                done = sum(complete(RESULTS / experiment / row["task"] / row["run_name"]) for row in plan)
                print(f"{experiment}/{batch} complete={done}/{len(plan)}", flush=True)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(type(exc).__name__, str(exc)[:300], flush=True)
        time.sleep(60)


def batch_issues(payload, plan, sessions):
    sessions_by_run = {row["run_name"]: row["session"] for row in plan}
    issues = []
    for row in payload["runs"]:
        name, status, heldout = row["name"], row["status"], row["heldout"]
        session = sessions_by_run.get(name)
        if not session:
            issues.append(f"{name}: 批次清单缺少会话")
        elif sessions is not None and status in {"recorded", "stale"} and session not in sessions:
            issues.append(f"{name}: 搜索会话已退出")
        if status in {"blocked", "unknown", "absent", "stopped"}:
            issues.append(f"{name}: 状态 {status}")
        if status == "stale":
            issues.append(f"{name}: 长时间没有新进度")
        if heldout["failed"] or heldout["mismatched"] or heldout["unverified"]:
            issues.append(f"{name}: 主测试失败或无法核验")
        if status == "finished" and not row["frozen"]:
            issues.append(f"{name}: 最终程序未核验")
        if status == "finished" and heldout["missing"] and sessions is not None and session not in sessions:
            issues.append(f"{name}: 主测试缺失且会话已退出")
    return issues


def lag_issues(there, here):
    local = {row["name"]: row for row in here["runs"]}
    issues = []
    for row in there["runs"]:
        mine = local.get(row["name"], {})
        if ((row["budget_used"] or 0) - (mine.get("budget_used") or 0) >= LAG
                or row["status"] == "finished" and mine.get("status") != "finished"):
            issues.append(f"{row['name']}: 本机结果落后于 server4")
    return issues


def check():
    from experiments.infra.batch_status import collect

    issues, summaries = [], {}
    try:
        failed = json.loads(command(["systemctl", "--user", "list-units", "--failed", "-o", "json"]))
        issues += [f"本机服务失败: {unit['unit']}" for unit in failed]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        issues.append("无法查询本机服务")
    try:
        with build_opener(ProxyHandler({})).open("http://127.0.0.1:8001/health", timeout=5) as response:
            if json.load(response).get("status") != "ok":
                raise ValueError("unhealthy")
    except (OSError, ValueError):
        issues.append("本机模型健康检查失败")
    try:
        command(SSH + ["curl --noproxy '*' -fsS --connect-timeout 5 --max-time 10 http://127.0.0.1:18001/health"])
    except (OSError, subprocess.SubprocessError):
        issues.append("server4 无法访问本机模型")
    try:
        sessions = set(command(SSH + ["tmux list-sessions -F '#{session_name}'"]).splitlines())
    except (OSError, subprocess.SubprocessError):
        sessions = None
        issues.append("无法查询 server4 实验会话")
    try:
        batches = active_batches(remote_manifests())
    except (OSError, ValueError, subprocess.SubprocessError):
        batches = []
        issues.append("无法查询 server4 上的批次")

    for experiment, batch in batches:
        key, manifest = f"{experiment}/{batch}", f"experiments_result/{experiment}/batch_{batch}.json"
        plan = read_object(ROOT / manifest).get("plan")
        if not plan:
            issues.append(f"{key}: 本机没有批次清单，同步可能未运行")
            continue
        try:
            there = json.loads(command([str(ROOT / ".venv/bin/python"), "-m", "experiments.infra.batch_status",
                                        "--ssh", HOST, "--repo", REMOTE, "--manifest", f"{REMOTE}/{manifest}",
                                        "--json"], timeout=75))
            summaries[key] = there["summary"]
            issues.extend(batch_issues(there, plan, sessions))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            issues.append(f"{key}: 无法查询批次状态")
            continue
        try:
            issues.extend(lag_issues(there, collect(ROOT / manifest)))
        except (OSError, ValueError, KeyError):
            issues.append(f"{key}: 无法查询本机批次状态")

    issues = sorted(set(issues))
    try:
        previous = json.loads(STATE.read_text()).get("notified", [])
    except (OSError, ValueError):
        previous = []
    notified = set(previous) & set(issues)
    new = sorted(set(issues) - notified)
    if new:
        try:
            subprocess.run(["notify-send", "-u", "critical", "TraceAAD 实验异常", "\n".join(new[:5])],
                           check=True, timeout=10)
            notified.update(new)
        except (OSError, subprocess.SubprocessError):
            pass
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "issues": issues,
              "notified": sorted(notified), "batches": summaries}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(STATE)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", choices=("check", "sync"), default="check")
    return sync() if parser.parse_args(argv).action == "sync" else check()


if __name__ == "__main__":
    main()

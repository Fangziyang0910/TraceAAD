"""Create a reviewed runtime containing both B/C ablation arms."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def freeze(batch, prefix="bc", arm=None):
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", value) for value in (batch, prefix)):
        raise ValueError("invalid batch or prefix")
    if arm not in ("B", "C"):
        raise ValueError("arm must be B or C")
    root = Path(__file__).resolve().parents[2]
    results = root / "experiments/traceaad_bc/results"
    runtime = results / f"runtime_{batch}"
    runtime.mkdir(parents=True, exist_ok=False)
    sources = [
        root / "llm4ad/base",
        root / "llm4ad/tools",
        root / "llm4ad/method/traceaad_bc",
        root / "llm4ad/method/traceaad_v10_11",
        root / "llm4ad/method/traceaad_v11_0",
        root / "experiments/infra",
        root / "experiments/traceaad_bc",
    ]
    sources += [root / "llm4ad/task/optimization" / task for task in (
        "tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing", "vrptw_construct"
    )]
    files = [
        root / "experiments/__init__.py",
        root / "llm4ad/__init__.py",
        root / "llm4ad/method/__init__.py",
        root / "llm4ad/task/__init__.py",
        root / "llm4ad/task/optimization/__init__.py",
        root / "llm4ad/task/optimization/generated_data_config.py",
    ]
    for directory in sources:
        for folder, dirs, names in os.walk(directory):
            dirs[:] = sorted(directory_name for directory_name in dirs
                             if directory_name not in ("results", "__pycache__", "data"))
            files.extend(
                Path(folder) / name for name in sorted(names)
                if Path(name).suffix in (".py", ".yaml", ".model")
            )
    hashes = {}
    for path in files:
        relative = path.relative_to(root)
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    if (root / ".env").is_file():
        (runtime / ".env").symlink_to(root / ".env")
    (runtime / "experiments/traceaad_bc/results").symlink_to(results, target_is_directory=True)
    command = [
        sys.executable,
        "-m",
        "experiments.traceaad_bc.launch",
        "--batch",
        batch,
        "--session-prefix",
        prefix,
        "--arm",
        arm,
        "--watch",
    ]
    payload = dict(
        batch=batch,
        runtime=str(runtime),
        python=sys.executable,
        python_version=sys.version,
        git_base=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        files=hashes,
        launch_command=command,
    )
    (runtime / "runtime_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--session-prefix", default="bc")
    parser.add_argument("--arm", choices=("B", "C"), required=True)
    args = parser.parse_args()
    result = freeze(args.batch, args.session_prefix, args.arm)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()

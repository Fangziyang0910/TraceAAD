"""Write checked launch plans for V19.4; preparing never starts searches."""

import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
from zoneinfo import ZoneInfo

from benchmarks.tasks import CO_TASKS
from traceaad_v19_4.search import METHOD
from traceaad_v19_4.storage import write_json
from traceaad_v19_4.tokens import TokenCounter
from .launch_host import main as launch_preview


ROOT = Path(__file__).resolve().parents[2]
CELLS = {"main": "trajectory", "neighborhood": "neighborhood", "rank": "rank", "direct": "direct", "preflight": "trajectory"}


def source_files():
    paths = set()
    for directory in ("traceaad_v19_4", "experiments/traceaad_v19_4", "benchmarks", "core",
                      "tests/independent_v19_4"):
        paths.update((ROOT / directory).rglob("*.py"))
    paths.update(ROOT / name for name in ("experiments/__init__.py", "traceaad_v19_4/pyproject.toml",
        "traceaad_v19_4/README.md", "experiments/traceaad_v19_4/README.md",
        "docs/机制设计/V19.4.md", "docs/机制设计/V19.4-上下文与提示词.md", "experiments/PROTOCOL.md"))
    return sorted(path for path in paths if path.is_file())


def verify(directory):
    directory = Path(directory).resolve()
    setup = json.loads((directory / "preparation.json").read_text())
    if setup["source_root"] != str(ROOT) or setup["method"] != METHOD:
        raise ValueError("Preparation belongs to another source root or method")
    expected = json.loads((directory / "source_hashes.json").read_text())
    changed = [name for name, digest in expected.items() if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
    tokenizer = directory / "assets/tokenizer.json"
    if hashlib.sha256(tokenizer.read_bytes()).hexdigest() != setup["tokenizer_sha256"]:
        changed.append("assets/tokenizer.json")
    for cell in setup["cells"]:
        path = directory / "plans" / (cell + ".json")
        plan = json.loads(path.read_text())
        if (plan["method"] != METHOD
                or hashlib.sha256(path.read_bytes()).hexdigest() != setup["plan_hashes"][cell]):
            changed.append("plans/" + cell + ".json")
    if changed:
        raise ValueError("Prepared source changed: " + ", ".join(changed))
    return {"verified_source_files": len(expected), "cells": setup["cells"], "source_root": str(ROOT)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--batch-prefix")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--scheduler-socket")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=CO_TASKS, default=list(CO_TASKS))
    parser.add_argument("--cells", nargs="+", choices=CELLS, default=list(CELLS))
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    directory = args.output.resolve()
    if args.verify:
        print(json.dumps(verify(directory), ensure_ascii=False))
        return
    if not args.batch_prefix or not args.tokenizer:
        parser.error("Preparation needs --batch-prefix and --tokenizer")
    if bool(args.scheduler_socket) == bool(args.base_url) or (args.base_url and not args.model):
        parser.error("Choose a scheduler or a direct endpoint with model")
    if len(set(args.cells)) != len(args.cells) or len(set(args.tasks)) != len(args.tasks):
        parser.error("Cells and tasks must not be repeated")
    if directory.exists():
        raise FileExistsError("Use a new preparation directory")
    tokenizer = args.tokenizer.expanduser()
    if tokenizer.is_dir():
        tokenizer = tokenizer / "tokenizer.json"
    tokenizer_bytes = tokenizer.read_bytes()
    TokenCounter(str(tokenizer))
    plans, commands = {}, {}
    copied_tokenizer = directory / "assets/tokenizer.json"
    for cell in args.cells:
        preview_args = ["--batch", args.batch_prefix + "_" + cell, "--experiment", METHOD if cell == "main" else METHOD + "_" + cell,
            "--tasks", *args.tasks, "--budget", str(5 if cell == "preflight" else args.budget),
            "--repeats", str(1 if cell == "preflight" else args.repeats),
            "--initial-size", "1" if cell == "preflight" else "5", "--selection-mode", CELLS[cell],
            "--reflection-mode", "trajectory", "--tokenizer", str(copied_tokenizer)]
        for name in ("scheduler_socket", "base_url", "model", "env_file"):
            if getattr(args, name):
                preview_args.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        if cell == "preflight":
            preview_args.append("--skip-heldout")
        with redirect_stdout(io.StringIO()):
            plans[cell] = launch_preview(preview_args)
        commands[cell] = [sys.executable, "-m", "experiments.traceaad_v19_4.launch_host", *preview_args, "--launch"]
    directory.mkdir(parents=True)
    copied_tokenizer.parent.mkdir()
    shutil.copyfile(tokenizer, copied_tokenizer)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files()}
    setup = {"method": METHOD, "source_root": str(ROOT),
        "prepared_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "cells": args.cells,
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
        "runs": {cell: len(plan["plan"]) for cell, plan in plans.items()},
        "commands": commands, "status": "prepared_not_launched"}
    write_json(directory / "source_hashes.json", hashes)
    setup["plan_hashes"] = {}
    for cell, plan in plans.items():
        write_json(directory / "plans" / (cell + ".json"), plan)
        setup["plan_hashes"][cell] = hashlib.sha256((directory / "plans" / (cell + ".json")).read_bytes()).hexdigest()
        check = [sys.executable, "-m", "experiments.traceaad_v19_4.prepare", "--output", str(directory), "--verify"]
        script = "#!/bin/sh\nset -eu\ncd " + shlex.quote(str(ROOT)) + "\n"
        script += shlex.join(check) + "\nexec " + shlex.join(commands[cell]) + "\n"
        path = directory / ("launch_" + cell + ".sh")
        path.write_text(script)
        path.chmod(0o755)
    write_json(directory / "preparation.json", setup)
    print(json.dumps({"directory": str(directory), **verify(directory), "runs": setup["runs"],
                      "status": setup["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

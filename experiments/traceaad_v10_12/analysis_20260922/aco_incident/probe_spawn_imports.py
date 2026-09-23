"""Reproduce ACO worker bootstrap failures without evaluating or changing a run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "experiments/traceaad_v10_12/results/runtime_20260921_v1012"
OUTPUT = Path(__file__).resolve().parent / "spawn_probe.json"

# Match multiprocessing.spawn's re-import of the search entry point. The parent
# imports the intact frozen modules first, modeling an already-running process.
# Only the child's source search path changes between cases.
PROBE = r'''
import __main__
import importlib
import multiprocessing
import os
import sys
entry = importlib.import_module('experiments.traceaad_v10_12.run')
print('parent_entry=' + entry.__file__, flush=True)
__main__.__spec__ = entry.__spec__
sys.path.insert(0, sys.argv[1])
child = multiprocessing.get_context('spawn').Process(target=os.getpid)
child.start()
child.join(12)
if child.is_alive():
    child.terminate()
    child.join()
    raise RuntimeError('Diagnostic child bootstrap exceeded 12 seconds')
print('child_exitcode=' + str(child.exitcode), flush=True)
sys.exit(0 if child.exitcode == 0 else 1)
'''


def main():
    manifest = json.loads((RUNTIME / "runtime_manifest.json").read_text())
    mismatches = [
        name for name, digest in manifest["files"].items()
        if hashlib.sha256((RUNTIME / name).read_bytes()).hexdigest() != digest
    ]
    assert not mismatches, mismatches
    env = dict(os.environ)
    env["PYTHONPATH"] = str(RUNTIME)
    # Bound diagnostic import overhead, identically in every case. This probe
    # does not alter the environment of any experiment or evaluate candidates.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "1"
    results = []
    with tempfile.TemporaryDirectory(prefix="v1012-spawn-probe-") as directory:
        missing = Path(directory)
        for name in manifest["files"]:
            target = missing / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(RUNTIME / name, target)
        shutil.rmtree(missing / "llm4ad/method/traceaad_v10_12_rand_ctx")
        cases = [
            ("intact_frozen_runtime", RUNTIME, 0, None),
            ("only_rand_ctx_package_removed", missing, 1, "ModuleNotFoundError"),
            ("current_working_tree", ROOT, 1, "IndentationError"),
        ]
        for label, child_root, expected, error in cases:
            result = subprocess.run(
                [sys.executable, "-c", PROBE, str(child_root)],
                cwd=RUNTIME, env=env, text=True, capture_output=True, timeout=20,
            )
            record = dict(case=label, returncode=result.returncode,
                          stdout=result.stdout, stderr=result.stderr)
            results.append(record)
            assert result.returncode == expected, record
            assert error is None or error in result.stderr, record
    OUTPUT.write_text(json.dumps({
        "captured_at": datetime.now().astimezone().isoformat(),
        "frozen_files_verified": len(manifest["files"]),
        "candidate_evaluations": 0,
        "description": "Same already-imported parent; change only source path used by spawned child.",
        "cases": results,
    }, ensure_ascii=False, indent=2) + "\n")
    for result in results:
        print(result["case"], "returncode", result["returncode"])
    print(OUTPUT)


if __name__ == "__main__":
    main()

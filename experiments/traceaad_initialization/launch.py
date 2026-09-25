"""Run a preregistered balanced schedule, six requests maximum on server3."""
import concurrent.futures
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BATCH = '20260925_init_v1'
TASKS = ['tsp_construct','online_bin_packing','cvrp_aco','op_aco','vrptw_construct']

def main():
    jobs = []
    rng = random.Random(20260925)
    for rep in range(1,5):
        tasks = TASKS.copy()
        rng.shuffle(tasks)
        for task in tasks:
            modes = ['independent','sequential','hybrid']
            rng.shuffle(modes)
            for mode in modes:
                jobs.append(dict(task=task, mode=mode, rep=rep, seed=92500+rep,
                    run_name=f'{BATCH}_{mode}_r{rep}',
                    init_only=task not in ['tsp_construct','online_bin_packing']))
    manifest = HERE / 'schedule.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == jobs
    else:
        manifest.write_text(json.dumps(jobs, indent=2))
    def execute(job):
        run = HERE/'results'/job['task']/job['run_name']
        summary = run/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('status') == 'finished':
            return job, 0
        cmd = [sys.executable,'-u','-m','experiments.traceaad_initialization.run',
            '--task',job['task'],'--backend','server3','--init-mode',job['mode'],
            '--seed',str(job['seed']),'--repeat',str(job['rep']),
            '--run-name',job['run_name'],'--budget','24','--eval-workers','2']
        if job['init_only']:
            cmd.append('--init-only')
        log = HERE/'logs'/f"{job['task']}_{job['run_name']}.log"
        log.parent.mkdir(exist_ok=True)
        with log.open('a') as f:
            code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        return job, code
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for job, code in pool.map(execute,jobs):
            print(json.dumps(dict(**job, returncode=code)), flush=True)

if __name__ == '__main__':
    main()

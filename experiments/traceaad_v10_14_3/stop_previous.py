"""Authorized one-time stop of the exact previous V10.14-2 batch."""
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from datetime import datetime

ROOT = Path('/home/fang/code/LLM4AD/TraceAAD')
manifest_path = ROOT / 'experiments_result/traceaad_v10_14_2/batch_20260928_v1014_2.json'
report_path = manifest_path.parent / 'stop_20260928_for_v1014_3.json'
manifest = json.loads(manifest_path.read_text())
names = {r['run_name'] for r in manifest['plan']}


def stamp():
    return datetime.now().astimezone().isoformat()


def processes():
    found = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            argv = (path / 'cmdline').read_bytes().decode().strip('\0').split('\0')
            if not argv or 'python' not in Path(argv[0]).name:
                continue
            if 'experiments.traceaad_v10_14_2.run' not in argv or '--run-name' not in argv:
                continue
            name = argv[argv.index('--run-name')+1]
            if name not in names:
                continue
            stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            found[int(path.name)] = {'pid': int(path.name), 'ppid': int(stat[1]),
                'start_ticks': stat[19], 'run_name': name, 'argv': argv}
        except (OSError, ValueError, IndexError):
            continue
    return found


def save(data):
    tmp = report_path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(report_path)


def tail_facts(path):
    size = path.stat().st_size
    with path.open('rb') as handle:
        start = max(0, size - 8_000_000)
        handle.seek(start)
        lines = handle.read().splitlines()
    if start:
        lines = lines[1:]
    state, completed = None, None
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if state is None and 'state' in row:
            state = row['state']
        if completed is None and row.get('kind') == 'attempt':
            completed = row['data']['id']
        if state is not None and completed is not None:
            break
    return {'bytes': size, 'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest(),
            'last_completed_attempt': completed, 'phase': state.get('phase') if state else None,
            'pending_external_operation': state.get('pending') if state else None,
            'ledger': state.get('ledger') if state else None}


if report_path.exists():
    raise RuntimeError('stop report already exists; inspect instead of rerunning')
before = processes()
mains = [p for p in before.values() if p['ppid'] not in before]
report = {'reason': 'user requested replacement by V10.14-3 after code verification',
          'requested_at': stamp(), 'signal': 'SIGINT', 'status': 'stopping',
          'main_processes': mains, 'observed_children': [p for p in before.values() if p not in mains]}
save(report)
for process in mains:
    current = processes().get(process['pid'])
    if current and current['start_ticks'] == process['start_ticks']:
        os.kill(process['pid'], signal.SIGINT)
        print('SIGINT', process['run_name'], process['pid'], flush=True)
deadline = time.monotonic() + 35
while processes() and time.monotonic() < deadline:
    time.sleep(1)
remaining = processes()
report['term_processes'] = list(remaining.values())
for p in remaining.values():
    current = processes().get(p['pid'])
    if current and current['start_ticks'] == p['start_ticks']:
        os.kill(p['pid'], signal.SIGTERM)
deadline = time.monotonic() + 10
while processes() and time.monotonic() < deadline:
    time.sleep(.5)
report['remaining'] = list(processes().values())
report['stopped_at'] = stamp()
report['runs'] = []
for item in manifest['plan']:
    directory = Path(item['run_dir'])
    summary = json.loads((directory / 'logs/run_summary.json').read_text())
    row = {k: item[k] for k in ('task', 'repeat', 'backend', 'run_name', 'run_dir')}
    row.update(summary_status=summary['status'], summary_budget_used=summary['budget_used'],
               journal=tail_facts(directory / 'search.jsonl'))
    report['runs'].append(row)
    item['status'] = 'finished' if summary['status'] == 'finished' else 'stopped_for_v1014_3'
    item['stopped_at'] = report['stopped_at']
report['status'] = 'stopped' if not report['remaining'] else 'requires_inspection'
save(report)
if report['remaining']:
    raise RuntimeError('some exact batch processes remain; inspect report before launch')
manifest.update(status='stopped_replaced_by_v1014_3', stopped_at=report['stopped_at'],
                stop_report=str(report_path), replacement_batch='20260928_v1014_3')
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'stopped_main_processes': len(mains), 'finished_runs': sum(r['summary_status']=='finished' for r in report['runs']),
                  'remaining': len(report['remaining']), 'report': str(report_path)}, ensure_ascii=False), flush=True)

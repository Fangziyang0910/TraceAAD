"""Count prior experience in the exact search prefixes already frozen for the paper.

This reads archived prompts; it neither refreshes snapshot.json nor runs search.
"""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'paper/evidence'
SNAPSHOT = EVIDENCE / 'snapshot.json'
MARKER = '\n\n[Earlier Algorithm-Design Experience]\n'


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


snapshot = json.loads(SNAPSHOT.read_text())
runs = []
for run in snapshot['runs']:
    lines = (ROOT / run['source']).read_bytes().splitlines(keepends=True)[:run['attempts']]
    prefix_hash = hashlib.sha256(b''.join(lines)).hexdigest()
    assert prefix_hash == run['sha256'], run['source']
    counts = {'edge': 0, 'node': 0}
    inherited = {'edge': 0, 'node': 0}
    matched = 0
    previous_outputs = []
    for line in lines:
        row = json.loads(line)
        calls = row['attempt'].get('record', {}).get('consolidation', {}).get('reflection_calls', [])
        update_outputs = []
        for call in calls:
            kind = {'variation': 'edge', 'selection': 'node', 'edge': 'edge', 'node': 'node'}[call['kind']]
            counts[kind] += 1
            prompt = call['prompt']
            if MARKER in prompt:
                assert prompt.count(MARKER) == 1
                block = prompt.split(MARKER, 1)[1].split('\n\n[Comparison Catalog]', 1)[0]
                assert block.strip()
                inherited[kind] += 1
                block = normalized(block)
                matched += any(note and note in block for note in previous_outputs)
            update_outputs.append(normalized(call['experience']))
        previous_outputs.extend(update_outputs)
    assert sum(counts.values()) == run['reflection_calls']
    assert sum(inherited.values()) == matched, (run['task'], run['seed'], inherited, matched)
    horizon = snapshot['common_budget']
    series = run['series'][:horizon]
    improvements = [i+1 for i in range(1, len(series))
                    if series[i]['best'] is not None and series[i-1]['best'] is not None
                    and series[i]['best'] < series[i-1]['best']]
    runs.append({
        'task': run['task'], 'seed': run['seed'], 'source': run['source'],
        'source_prefix_sha256': prefix_hash, 'prefix_attempts': run['attempts'],
        'reflection_calls': counts, 'with_prior_experience': inherited,
        'containing_output_from_an_earlier_update': matched,
        'last_improvement_at_or_before_common_horizon': max(improvements),
    })

totals = {key: {kind: sum(r[key][kind] for r in runs) for kind in ['edge', 'node']}
          for key in ['reflection_calls', 'with_prior_experience']}
total_calls = sum(totals['reflection_calls'].values())
total_inherited = sum(totals['with_prior_experience'].values())
report = {
    'snapshot_sha256': hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
    'scope': 'Same frozen prefixes as snapshot.json; prior experience is counted in reflection inputs, not as a measure of summary quality or search benefit.',
    'definition': 'A nonempty Earlier Algorithm-Design Experience block containing at least one reflection output from a strictly earlier population update, matched after whitespace normalization.',
    'common_horizon': snapshot['common_budget'], 'totals': totals,
    'reflection_calls': total_calls, 'with_prior_experience': total_inherited,
    'with_prior_experience_percent': 100*total_inherited/total_calls,
    'runs': runs,
}
(EVIDENCE / 'experience_reuse.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, ensure_ascii=False, indent=2))

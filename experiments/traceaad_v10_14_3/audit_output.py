"""Audit a fixed prefix of real requests and responses without further model calls."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import json
from pathlib import Path
import re

from experiments.infra.launcher import write_json_atomic
from traceaad.v10_14_3.edits import parse_response


LAYOUT = re.compile(r"\AIdea:[ \t]*(.*?)\n[ \t]*code:[ \t]*\n\s*```python[ \t]*\n(.*?)\n```\s*\Z", re.S)
TEMPLATE = "Idea:\n<Explanation of the proposed algorithm or change>\n\ncode:\n```python\n<Complete executable Python code>\n```"


def read_prefix(path, limit):
    requests, calls, attempts = {}, {}, []
    digest, size = hashlib.sha256(), 0
    if path.exists():
        with path.open('rb') as stream:
            for line in stream:
                if not line.endswith(b'\n'):
                    break
                digest.update(line)
                size += len(line)
                kind = re.match(rb'\{"kind":\s*"([^"]+)"', line)
                if not kind or kind[1] not in (b'request', b'call', b'attempt'):
                    continue
                row = json.loads(line)
                if row['kind'] == 'request':
                    requests[row['data']['id']] = row['data']
                elif row['kind'] == 'call':
                    calls[row['request_id']] = row
                else:
                    attempts.append(row['data'])
                    if len(attempts) == limit:
                        break
    return requests, calls, attempts, {'bytes': size, 'sha256': digest.hexdigest()}


def audit(manifest_path, limit=5):
    manifest = json.loads(manifest_path.read_text())
    rows, totals, backends = [], Counter(), defaultdict(Counter)
    for item in manifest['plan']:
        requests, calls, attempts, boundary = read_prefix(Path(item['run_dir']) / 'search.jsonl', limit)
        template = import_module(f"benchmarks.{item['task']}.template").template_program
        counts, records = Counter(), []
        for attempt in attempts:
            rid = attempt.get('request_id')
            request, call = requests.get(rid, {}), calls.get(rid, {})
            text = call.get('response', '')
            counts['attempts'] += 1
            counts[attempt['status']] += 1
            template_sent = (request.get('output_mode') == 'full'
                             and TEMPLATE in request.get('prompt', '').split('# Delivery\n')[-1]
                             and not request.get('response_format') and not request.get('structured_outputs'))
            counts['template_sent'] += template_sent
            record = {'candidate': attempt['id'], 'status': attempt['status'],
                      'template_sent': template_sent, 'finish_reason': call.get('finish_reason'),
                      'idea_sources': attempt.get('delivery', {}).get('idea_sources', []),
                      'error': attempt.get('error')}
            if call.get('error') or not call:
                counts['no_model_response'] += 1
                records.append(record)
                continue
            counts['responses'] += 1
            layout = LAYOUT.fullmatch(text.strip())
            canonical = bool(layout and layout[1].strip() and layout[2].strip())
            counts['canonical_layout'] += canonical
            metadata = {}
            parse_error = None
            try:
                code, idea = parse_response(text, call.get('finish_reason'), template, metadata=metadata)
            except (ValueError, SyntaxError, TypeError) as exc:
                code, idea, parse_error = getattr(exc, 'code', None), metadata.get('idea_raw', ''), str(exc)
            matches = (code == attempt.get('code') and idea == attempt.get('idea', '')
                       and bool(parse_error) == (attempt['status'] == 'delivery_failed'))
            # This independent check reads the two fields from the requested
            # layout, rather than using the production parser to define them.
            layout_idea_matches = not canonical or idea == layout[1].strip()
            counts['storage_matches_parser'] += matches
            counts['canonical_idea_mismatch'] += not layout_idea_matches
            counts['idea_present'] += bool(attempt.get('idea', '').strip())
            counts['missing_idea'] += not bool(attempt.get('idea', '').strip())
            if not parse_error:
                counts['parsed'] += 1
                counts['fallback_parsed'] += not canonical
            record.update(canonical_layout=canonical, storage_matches_parser=matches,
                          canonical_idea_matches=layout_idea_matches, parse_error=parse_error,
                          response_excerpt=text[:350], idea_excerpt=idea[:250])
            records.append(record)
        row = {k: item[k] for k in ('task', 'repeat', 'backend', 'run_name', 'run_dir')}
        row.update(window_complete=len(attempts) == limit, counts=dict(counts), boundary=boundary, records=records)
        rows.append(row)
        totals.update(counts)
        backends[item['backend']].update(counts)
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'batch': manifest['batch'],
            'attempts_per_run': limit, 'window_complete': len(rows) == 20 and all(r['window_complete'] for r in rows),
            'counts': dict(totals), 'backends': {k: dict(v) for k, v in backends.items()}, 'runs': rows,
            'engineering_checks_passed': (totals['attempts'] > 0 and totals['template_sent'] == totals['attempts']
                                          and totals['storage_matches_parser'] == totals['responses']
                                          and totals['canonical_idea_mismatch'] == 0),
            'scope': 'Fixed first completed attempts per run, including failures; no resampling or evaluation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    report = audit(args.manifest, args.limit)
    write_json_atomic(args.output, report)
    print(json.dumps({k: report[k] for k in ('checked_at', 'window_complete', 'engineering_checks_passed', 'counts', 'backends')}))


if __name__ == '__main__':
    main()

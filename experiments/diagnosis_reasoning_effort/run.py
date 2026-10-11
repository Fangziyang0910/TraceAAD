"""Replay fixed EoH/ReEvo prompts in high, low, then non-thinking mode without a client output cap."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import argparse
import hashlib
import json
from pathlib import Path
import time

from baselines.eoh.sampler import trim_braced_thought
from baselines.shared.profiler import ProfilerBase
from baselines.shared.sampling import SampleTrimmer
from core.llm import OpenAIAPI, generate, ModelCallError
from experiments.infra.base import BACKENDS, set_no_proxy
from experiments.infra.env import resolve_llm_api_key
from traceaad.common.storage import write_json


def replay(case, mode, root):
    directory = root / mode / case['id']
    directory.mkdir(parents=True, exist_ok=False)
    profiler = ProfilerBase(directory)
    profile = BACKENDS['deepseek']
    thinking = mode != 'off'
    client = OpenAIAPI(base_url=profile.base_url, api_key=resolve_llm_api_key(base_url=profile.base_url),
        model=profile.model, timeout=1800, max_tokens=None, temperature=case['sampling']['temperature'],
        top_p=case['sampling'].get('top_p'), enable_thinking=thinking,
        reasoning_effort=mode if thinking else None)
    row = {'id': case['id'], 'method': case['method'], 'task': case['task'], 'mode': mode,
           'prompt_sha256': case['prompt_sha256'], 'max_tokens_sent': False,
           'temperature': client.temperature, 'top_p': client.top_p, 'request_timeout_seconds': 1800}
    started = time.monotonic()
    try:
        result = generate(client, case['prompt'])
        profiler.log_llm_call(transport_calls=result['calls'], stage='initialization', operator='replay')
        usage = result.get('usage') or {}
        reasoning = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
        completion = usage.get('completion_tokens')
        row.update(model=result.get('model'), finish_reason=result.get('finish_reason'),
            prompt_tokens=usage.get('prompt_tokens'), completion_tokens=completion,
            reasoning_tokens=reasoning, total_tokens=usage.get('total_tokens'),
            answer_tokens=completion-reasoning if isinstance(completion, int) and isinstance(reasoning, int) else None,
            response_empty=not bool(result['content']), reasoning_received=bool(result.get('reasoning')))
        try:
            program = SampleTrimmer.sample_to_program(result['content'], case['template_program'])
            row['code_parse_success'] = program is not None
            row['thought_parse_success'] = trim_braced_thought(result['content']) is not None if case['method'] == 'eoh' else None
            if program is not None:
                (directory / 'candidate.py').write_text(str(program))
        except Exception as error:
            row.update(code_parse_success=False, parse_error=f'{type(error).__name__}: {error}')
    except ModelCallError as error:
        profiler.log_llm_call(transport_calls=error.calls, stage='initialization', operator='replay')
        row.update(error=str(error), response_empty=True, code_parse_success=False)
    except Exception as error:
        row.update(error=f'{type(error).__name__}: {error}', code_parse_success=False)
    finally:
        row['seconds'] = time.monotonic()-started
        row['token_usage'] = dict(profiler._token_usage)
        write_json(directory/'result.json', row)
        profiler.write_run_summary(status='finished' if not row.get('error') else 'aborted')
        client.close()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    cases = json.loads(args.inputs.read_text())['cases']
    assert len(cases) == 12 and len({c['id'] for c in cases}) == 12
    assert args.workers > 0
    for case in cases:
        assert hashlib.sha256(json.dumps(case['prompt'], ensure_ascii=False, sort_keys=True).encode()).hexdigest() == case['prompt_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    profile = BACKENDS['deepseek']
    set_no_proxy(profile.no_proxy)
    manifest = {'created_at': datetime.now().astimezone().isoformat(), 'status': 'running',
        'design': '12 fixed prompts; phases high, low, off; one draw per prompt/mode; no client output cap',
        'model': profile.model, 'workers': args.workers, 'cases': [{k:c[k] for k in ('id','method','task','prompt_sha256')} for c in cases],
        'inputs_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(), 'results': []}
    write_json(args.output/'manifest.json', manifest)
    for mode in ('high', 'low', 'off'):
        manifest['phase'] = mode
        write_json(args.output/'manifest.json', manifest)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(replay, case, mode, args.output) for case in cases]
            for future in as_completed(futures):
                row = future.result()
                manifest['results'].append(row)
                write_json(args.output/'manifest.json', manifest)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    manifest['status'] = 'finished'
    write_json(args.output/'manifest.json', manifest)


if __name__ == '__main__':
    main()

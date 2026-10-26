"""Small online-bank experiment; cached calls replay the same adaptive decisions."""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import random

from benchmarks.tasks import heldout_task, training_task
import core.llm as llm_api
from experiments.infra.base import BACKENDS, build_llm_client
from traceaad.config import Config
from traceaad.delivery import validate_source
from traceaad.history import code_diff
from traceaad.instance_evaluation import InstanceProgramEvaluator
from traceaad.program import Program
from traceaad.prompts import PromptBuilder
from traceaad.selection import better, choose_reference, sample_parent
from traceaad.storage import read_json, write_json
from .run import canonical, digest

OUTPUT_LIMIT = 8192


def reply_object(details):
    if details.get('error') or details.get('finish_reason') != 'stop':
        raise ValueError(details.get('error') or 'incomplete completion')
    content = details['content'].strip()
    # Accept only a single JSON fence, identically for every arm and backend.
    if content.startswith('```json\n') and content.endswith('\n```'):
        content = content[8:-4]
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError('expected JSON object')
    return value


def request(out, index, prompt, llm, *, kind):
    target = out / 'calls' / f'{index:03}.json'
    fingerprint = digest(prompt)
    saved = read_json(target)
    if saved:
        if saved['prompt_sha256'] != fingerprint or saved['kind'] != kind:
            raise ValueError(f'resume request changed: {target}')
        return saved['details']
    # This isolated experiment counts failed transports and makes no free retries.
    llm_api.SERVICE_WAIT_SECONDS = 0
    try:
        details = llm_api.generate(llm, prompt, max_tokens=OUTPUT_LIMIT,
                                   response_format={'type': 'json_object'})
    except llm_api.ModelCallError as exc:
        details = dict(error=str(exc), calls=exc.calls, content='', finish_reason=None)
    assert len(details['calls']) == 1
    write_json(target, dict(prompt_sha256=fingerprint, kind=kind, details=details))
    return details


def evaluation(out, name, code, evaluator):
    target = out / 'evaluations' / f'{name}.json'
    saved = read_json(target)
    if saved:
        if saved['code_sha256'] != digest(code):
            raise ValueError(f'resume source changed: {target}')
        return saved
    result = evaluator.evaluate(code, name)
    write_json(target, dict(code_sha256=digest(code), **result))
    return read_json(target)


def evaluator(task, role):
    return InstanceProgramEvaluator(task, role=role, n_workers=8, timeout_seconds=120,
                                    scheduler_socket=os.environ.get('TRACEAAD_SCHEDULER_SOCKET'))


def client(backend):
    settings = BACKENDS['server3' if backend == 'qwen' else backend]
    return build_llm_client(base_url=settings.base_url, model=settings.model,
                            no_proxy=settings.no_proxy, max_tokens=OUTPUT_LIMIT)


def bank_prompt(base, bank):
    # Keep the original history and branches; replace only the final action/output request.
    base = base[:base.index('[Your Task:')]
    return base + '\n[Available Transformations]\n' + json.dumps(bank, ensure_ascii=False) + '''
[Your Task: Choose and Improve]
Use the current algorithm's actual computations, its formation history and failed branches to
choose one applicable transformation. Bind its roles to the recipient, preserve unrelated
computations, and implement its operation with permitted adaptations. Check your code against
its checks, including excluding the candidate itself when finding another neighbor. Outcomes
are evidence about previous applications, not proof that the transformation always helps.
If none is suitable, choose null and make an open modification instead. Do not copy a different
algorithm wholesale. Return only JSON with analysis, design, code (complete Python source),
transformation_id (one listed id or null), bindings (string), checks (string).
The checks field reports your own reasoning; it is not an independent verification.
'''


def induction_prompt(parent, child):
    return '''Extract one reusable transformation from this evaluated improvement. Lower scores
are better. Ground it in executable differences. The improvement belongs to the whole change;
do not claim each part caused it. Specify required computational roles rather than variable
names. Preserve the recipient's unrelated logic. Separate original parameter values from allowed
adaptations. A nearest OTHER neighbor excludes self; select_next_node returns a full-graph node
ID. State dependencies and how to check them. Return only JSON with six nonempty strings:
name, applicability, operation, preserve, parameters, checks.
''' + json.dumps(dict(before=parent.code, after=child.code, diff=code_diff(parent.code, child.code),
                     before_score=parent.fitness, after_score=child.fitness))


def update_bank(bank, selected, parent, child, error=None):
    if selected is None:
        return
    item = next(item for item in bank if item['id'] == selected)
    improved = child is not None and child.valid and better(child.fitness, parent.fitness)
    item['attempts'] += 1
    item['improvements'] += int(improved)
    item['recent_outcomes'] = (item['recent_outcomes'] + [dict(
        parent_id=parent.id, parent_score=parent.fitness,
        child_score=child.fitness if child is not None else None,
        error=error or (child.failure['kind'] if child is not None and child.failure else None),
        improved=improved)])[-3:]


def decode(reply, template, bank, offered):
    selected = reply.get('transformation_id')
    if selected is not None and (not offered or selected not in [item['id'] for item in bank]):
        raise ValueError('unknown transformation_id')
    for key in ('analysis', 'design', 'code'):
        if not isinstance(reply.get(key), str) or not reply[key].strip():
            raise ValueError(f'missing {key}')
    if selected is not None and any(not isinstance(reply.get(k), str) or not reply[k].strip()
                                    for k in ('bindings', 'checks')):
        raise ValueError('missing transformation binding/checks')
    validate_source(reply['code'], template)
    return selected


def run(out, inputs, arm, seed, backend, budget):
    config = dict(arm=arm, seed=seed, backend=backend, budget=budget,
                  inputs_sha256=digest(json.dumps(inputs, sort_keys=True)),
                  driver_sha256=digest(Path(__file__).read_text()))
    if read_json(out / 'run_config.json', config) != config:
        raise ValueError('resume configuration changed')
    write_json(out / 'run_config.json', config)
    created = {p['id']: p['created_at'] for p in read_json(out / 'state.json', {}).get('programs', [])}
    task, _ = training_task('tsp_construct')
    train = evaluator(task, 'search')
    programs = {}
    # Three paired blocks: weak, middle, strong historical starts, two roots per block.
    roots = [p for p in inputs['parents'] if p['tier'] == 2 - seed]
    assert len(roots) == 2
    for index, root in enumerate(roots, 1):
        result = evaluation(out, f'root_{index}', root['code'], train)
        assert result['fitness'] is not None
        programs[index] = Program(id=index, code=root['code'], idea=root['idea'], analysis=None,
                                 action='Init', generation=0, status='valid', fitness=result['fitness'])
    llm = client(backend)
    prompts = PromptBuilder(llm, 'tsp_construct', task, programs, Config())
    rng = {name: random.Random(f'{seed}:{name}') for name in ('parent', 'action', 'reference')}
    bank, curve, events = [], [], []
    used = generated = 0
    pending = None
    try:
        while used < budget:
            if pending is not None:
                parent, child = pending
                details = request(out, used + 1, induction_prompt(parent, child), llm, kind='induce')
                used += 1
                error = None
                try:
                    spec = reply_object(details)
                    for key in ('name', 'applicability', 'operation', 'preserve', 'parameters', 'checks'):
                        if not isinstance(spec.get(key), str) or not spec[key].strip():
                            raise ValueError(f'invalid specification: {key}')
                    bank.append(dict(id=f't{child.id}', spec=spec, source_parent=parent.id,
                                     source_child=child.id, attempts=0, improvements=0, recent_outcomes=[]))
                except (ValueError, TypeError) as exc:
                    error = str(exc)
                events.append(dict(call=used, kind='induce', source=child.id, error=error))
                pending = None
            else:
                valid = {p.id: p for p in programs.values() if p.valid}
                parent, selection = sample_parent(list(valid.values()), programs, rng['parent'])
                action = rng['action'].choices(['Refine', 'Explore', 'Crossover'], [.45, .30, .25])[0]
                reference = None
                if action == 'Crossover':
                    reference, _ = choose_reference(parent, valid, rng['reference'])
                    if reference is None:
                        action = 'Refine'
                offered = arm == 'bank' and bool(bank) and generated % 2 == 1
                # Bank turns always expose this parent's formation path and direct attempts.
                request_data = prompts.build('Refine' if offered else action, parent, reference=reference)
                prompt = bank_prompt(request_data['prompt'], bank) if offered else request_data['prompt']
                details = request(out, used + 1, prompt, llm, kind='bank' if offered else action)
                used += 1
                generated += 1
                child = selected = error = None
                try:
                    reply = reply_object(details)
                    # Keep a valid claimed id even when the code fails delivery validation.
                    selected = reply.get('transformation_id')
                    if selected not in [item['id'] for item in bank] or not offered:
                        selected = None
                    decode(reply, task.template_program, bank, offered)
                    child = Program(id=len(programs) + 1, code=reply['code'], idea=reply['design'],
                                    analysis=reply['analysis'], action='Refine' if offered else action,
                                    generation=used, parent_id=parent.id, depth=parent.depth + 1,
                                    reference_id=reference.id if reference and not offered else None,
                                    request=dict(transformation_id=selected, offered=offered, selection=selection))
                    result = evaluation(out, f'candidate_{child.id}', child.code, train)
                    child.fitness, child.failure = result['fitness'], result['failure']
                    child.status = 'valid' if child.failure is None else child.failure['kind']
                    programs[child.id] = child
                    if (arm == 'bank' and selected is None and child.valid and len(bank) < 6
                            and better(child.fitness, parent.fitness) and used + 1 < budget
                            and canonical(child.code) != canonical(parent.code)):
                        pending = (parent, child)
                except (ValueError, SyntaxError, TypeError) as exc:
                    error = str(exc)
                update_bank(bank, selected, parent, child, error)
                events.append(dict(call=used, kind='candidate', parent=parent.id,
                                   child=child.id if child else None, offered=offered,
                                   transformation_id=selected, error=error))
            best = min((p for p in programs.values() if p.valid), key=lambda p: (p.fitness, p.id))
            curve.append(dict(calls=used, best=best.fitness, selected=best.id, bank_size=len(bank)))
            for p in programs.values():
                p.created_at = created.setdefault(p.id, p.created_at)
            write_json(out / 'state.json', dict(programs=[asdict(p) for p in programs.values()],
                                               bank=bank, curve=curve, events=events))
            print(out.name, used, events[-1]['kind'], 'best', round(best.fitness, 6), 'bank', len(bank), flush=True)
        # Selection is committed before any held-out result is read.
        write_json(out / 'selection.json', dict(id=best.id, train=best.fitness, code_sha256=digest(best.code)))
        return finish(out, config)
    finally:
        llm.close()


def finish(out, config=None):
    """Evaluate the committed winner; usable after interruption during final evaluation."""
    config = config or read_json(out / 'run_config.json')
    state = read_json(out / 'state.json')
    selected = read_json(out / 'selection.json')
    assert state['curve'][-1]['calls'] == config['budget']
    best = next(p for p in state['programs'] if p['id'] == selected['id'])
    assert digest(best['code']) == selected['code_sha256']
    heldout = heldout_task('tsp_construct', 'eval_50')
    result = evaluation(out, 'winner_heldout', best['code'], evaluator(heldout, 'heldout'))
    calls = [read_json(p)['details']['calls'][0] for p in sorted((out / 'calls').glob('*.json'))]
    assert len(calls) == config['budget']
    summary = dict(**config, winner=best['id'], train=best['fitness'], heldout=result['fitness'],
                   initial_best=min(p['fitness'] for p in state['programs'] if p['generation'] == 0),
                   generations=sum(e['kind'] == 'candidate' for e in state['events']),
                   bank_size=len(state['bank']), calls=len(calls),
                   tokens=sum(c['usage'].get('total_tokens', 0) for c in calls),
                   missing_usage=sum(not c['usage'].get('total_tokens') for c in calls),
                   transport_errors=sum(bool(c['error']) for c in calls))
    write_json(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--arm', choices=['open', 'bank'], required=True)
    parser.add_argument('--backend', choices=['qwen', 'deepseek'], default='qwen')
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--budget', type=int, default=32)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.out, read_json(args.inputs), args.arm, args.seed, args.backend, args.budget)


if __name__ == '__main__':
    main()

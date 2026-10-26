"""Matched model execution checks with frozen, unambiguous operations and behavioral oracles."""

import argparse
import ast
import fcntl
import json
from pathlib import Path

import numpy as np

from benchmarks.tasks import training_task
from core.evaluate import Evaluation
from traceaad.storage import read_json, write_json
from .loop import client, evaluation, evaluator, reply_object, request
from .run import decode_candidate, digest

OPERATIONS = {
    't42': 'Replace the mean distance to other unvisited nodes by the minimum distance to another unvisited node, excluding self by setting the diagonal to infinity. In the final score, change the destination-distance coefficient from 0.6 to 0.3 and the connectivity coefficient from 0.2 to 0.5. Preserve the current-distance coefficient and all guards and return mapping.',
    't89': 'Let scale be the mean distance from unvisited nodes to destination. If scale < 1e-9, use 1.0. Change the second-neighbor bridge coefficient from 0.3 to 0.3/scale. Change the savings coefficient from 0.5*urgency to 0.6*urgency/scale. Preserve every other term and all small-set guards.',
    't284': 'For each candidate u, choose its nearest OTHER unvisited node v, excluding u. Add 0.5 times (distance(current,u) + distance(u,v) + dest_weight*distance(v,destination)) to the existing score. Change the existing nearest-neighbor-distance coefficient from 0.3 to 0.2. Preserve the existing current-distance and destination-distance terms, the dest_weight formula, small-set guards and node-ID mapping.',
}


class Rename(ast.NodeTransformer):
    def __init__(self, mapping):
        self.mapping = mapping
    def visit_Name(self, node):
        return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)


def renamed(before, after):
    names = sorted({n.id for code in (before, after) for n in ast.walk(ast.parse(code))
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)})
    mapping = {name: f'local_{i}' for i, name in enumerate(names)}
    return tuple(ast.unparse(Rename(mapping).visit(ast.parse(code))) + '\n' for code in (before, after))


class BehaviorCheck(Evaluation):
    def __init__(self, reference, template):
        super().__init__(template)
        self.reference = reference

    def evaluate_program(self, program_str, callable_func):
        namespace = {}
        exec(self.reference, namespace)
        expected = namespace['select_next_node']
        rng = np.random.default_rng(20261025)
        mismatches = 0
        for index in range(300):
            size = int(rng.integers(5, 20))
            points = rng.random((size, 2)) * (1e-12, 1e-6, .1, 1, 10)[index % 5]
            distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            ids = rng.permutation(size)
            current, destination = int(ids[0]), int(ids[1])
            remaining = ids[2:2 + int(rng.integers(1, size - 1))]
            correct = expected(current, destination, remaining.copy(), distances.copy())
            actual = callable_func(current, destination, remaining.copy(), distances.copy())
            if not isinstance(actual, (int, np.integer)) or isinstance(actual, (bool, np.bool_)):
                return None
            mismatches += int(actual != correct)
        return mismatches / 300


def prepare(out, pilot):
    cases = []
    for source in pilot['sources']:
        before, after = source['parent']['code'], source['child']['code']
        for form in ('original', 'renamed'):
            b, a = (before, after) if form == 'original' else renamed(before, after)
            cases.append(dict(id=source['id'] + '_' + form, code=b, reference=a,
                              operation=OPERATIONS[source['id']]))
    payload = dict(cases=cases, draws=2, backends=['qwen', 'deepseek'])
    if read_json(out / 'inputs.json', payload) != payload:
        raise ValueError('frozen execution cases changed')
    write_json(out / 'inputs.json', payload)
    return payload


def run(out, backend):
    task, _ = training_task('tsp_construct')
    inputs = read_json(out.parent / 'inputs.json')
    config = dict(backend=backend, inputs_sha256=digest(json.dumps(inputs, sort_keys=True)),
                  driver_sha256=digest(Path(__file__).read_text()))
    if read_json(out / 'config.json', config) != config:
        raise ValueError('execution configuration changed')
    write_json(out / 'config.json', config)
    llm = client(backend)
    results = []
    try:
        for case in inputs['cases']:
            probe = evaluator(BehaviorCheck(case['reference'], task.template_program), 'behavior')
            control = evaluation(out, case['id'] + '_oracle', case['reference'], probe)
            assert control['fitness'] == 0
            parent = evaluation(out, case['id'] + '_parent', case['code'], probe)
            assert parent['fitness'] > 0
            prompt = (task.task_description + '\nINTERFACE:\n' + task.template_program
                      + '\nPARENT:\n' + case['code'] + '\nREQUIRED OPERATION:\n' + case['operation'] + '''
This operation is applicable to this parent. Apply precisely this operation, preserving unrelated
computations. Return only JSON with applicable (boolean), bindings (string), design (string), and
edits (list of objects with old and new string fields). Each old block must match exactly once
in the current source after earlier edits. Do not return full code. Check dependencies and roles
against the recipient rather than variable names. Do not make other improvements.
''')
            for draw in range(inputs['draws']):
                details = request(out, len(results) + 1, prompt, llm, kind=case['id'])
                result = dict(case=case['id'], draw=draw, valid=False, matched=False)
                try:
                    reply = reply_object(details)
                    code, status = decode_candidate({'arm': 'constrained'}, case, reply, task.template_program)
                    test = evaluation(out, f'{case["id"]}_{draw}', code, probe)
                    result.update(valid=test['fitness'] is not None, matched=test['fitness'] == 0,
                                  mismatch_fraction=test['fitness'], failure=test['failure'], status=status,
                                  code=code, reply=reply)
                except (ValueError, SyntaxError, TypeError) as exc:
                    result['error'] = str(exc)
                results.append(result)
                write_json(out / 'results.json', results)
                print(backend, case['id'], draw, result['matched'], flush=True)
        write_json(out / 'summary.json', dict(backend=backend, attempts=len(results),
                                              valid=sum(r['valid'] for r in results),
                                              matched=sum(r['matched'] for r in results)))
    finally:
        llm.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', choices=['qwen', 'deepseek'])
    parser.add_argument('--prepare', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            prepare(args.out, read_json(args.prepare))
        else:
            if not args.backend:
                parser.error('--backend is required for execution')
            run(args.out, args.backend)


if __name__ == '__main__':
    main()

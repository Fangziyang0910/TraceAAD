"""Resumable V13 search: every model stage and evaluated program is checkpointed."""

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from core.llm import generate, ModelCallError
from core.scheduling import SchedulerError
from traceaad.delivery import validate_source
from traceaad.history import code_diff
from traceaad.program import Program
from traceaad.selection import better, temperature
from traceaad.storage import read_json, write_json
from traceaad.traceaad import TraceAAD
from .config import Config
from .knowledge import (ancestors, apply_edits, bindings, changed_regions, digest, distribution,
                        eligible, evidence_rate, specification, used_on_path)
from .prompts import node_view, prompt, trajectory


class CallBudgetExhausted(Exception):
    pass


class TraceAADV13(TraceAAD):
    METHOD = 'traceaad_v13'

    def __init__(self, *, config=None, **kwargs):
        super().__init__(config=config or Config(), **kwargs)
        self.task = kwargs.get('task')
        implementation = digest(''.join(p.read_text() for p in sorted(Path(__file__).parent.glob('*.py'))))
        signature = digest(json.dumps(dict(config=asdict(self.config), task=self.task,
                                          template=self.template, implementation=implementation), sort_keys=True))
        saved = self.facts.state or {}
        if saved and ('v13' not in saved or saved['v13']['config_signature'] != signature):
            raise ValueError('V13 resume requires the same version and configuration')
        self.state = deepcopy(saved.get('v13') or dict(
            config_signature=signature, library={}, routes={}, attempts=[], work=None,
            reply=None, inflight=None, requests=0, interrupted=[], stages={},
            missing_usage=0, measured_input_tokens=0, measured_output_tokens=0,
            next_transform=1, stop_reason=None))
        if not saved and self.config.library_path:
            self._import_library(Path(self.config.library_path))

    def _save(self, **record):
        self.facts.commit({**self.progress.snapshot(), 'v13': self.state}, **record)

    def _ask(self, stage, required, optional=()):
        text = prompt(self.prompts.common, stage, required, optional)
        optional = list(optional)
        while self.llm.count_prompt_tokens(text) > self.config.prompt_tokens:
            if not optional:
                raise ValueError('required program context exceeds V13 prompt limit')
            optional.pop()
            text = prompt(self.prompts.common, stage, required, optional)
        key = digest(text)
        if self.state['reply'] is not None:
            saved = self.state['reply']
            if saved['key'] != key or saved['stage'] != stage:
                raise RuntimeError('checkpointed response does not match the resumed stage')
            return self._json(saved)
        if self.state['inflight']:
            self.state['interrupted'].append(self.state['inflight'])
            self.state['inflight'] = None
            self._save()
        if self.state['requests'] >= self.config.max_model_calls:
            raise CallBudgetExhausted()
        self.state['requests'] += 1
        self.state['inflight'] = dict(id=self.state['requests'], stage=stage, key=key, prompt=text)
        self._save()
        request = dict(action=stage, input_tokens=self.llm.count_prompt_tokens(text))
        try:
            result = generate(self.llm, text, max_tokens=self.config.output_tokens,
                              response_format={'type': 'json_object'}, retry_limit=0)
        except ModelCallError as exc:
            self._record_calls(exc.calls, request)
            self.state['inflight'] = None
            self._save(calls=exc.calls)
            raise
        self._record_calls(result['calls'], request)
        self.state['inflight'] = None
        self.state['reply'] = dict(key=key, stage=stage, content=result.get('content'),
                                   finish_reason=result.get('finish_reason'))
        self._save(calls=result['calls'])
        return self._json(self.state['reply'])

    @staticmethod
    def _json(reply):
        if reply['finish_reason'] != 'stop' or not isinstance(reply['content'], str):
            raise ValueError('incomplete completion')
        text = reply['content'].strip()
        if text.startswith('```json\n') and text.endswith('\n```'):
            text = text[8:-4]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError('expected one JSON object')
        return value

    def _record_calls(self, calls, request):
        self.progress.record_calls(calls, request, self.config.output_tokens)
        for call in calls:
            stage = call['stage']
            self.state['stages'][stage] = self.state['stages'].get(stage, 0) + 1
            usage = call['usage']
            self.state['missing_usage'] += int(any(usage.get(k) is None for k in ('prompt_tokens', 'completion_tokens')))
            self.state['measured_input_tokens'] += usage.get('prompt_tokens', 0) or 0
            self.state['measured_output_tokens'] += usage.get('completion_tokens', 0) or 0

    def _advance(self, stage, **record):
        self.state['reply'] = None
        self.state['work']['stage'] = stage
        self._save(**record)

    def _catalog(self, ids=None):
        return [dict(id=tid, version=t['version'], spec=t['spec'],
                     rate=evidence_rate(t['records']), uses=len(t['records']),
                     recent=[{k: r.get(k) for k in ('outcome', 'route', 'failure', 'features', 'diagnosis')} for r in t['records'][-3:]])
                for tid, t in self.state['library'].items() if ids is None or tid in ids]

    def _shortlist(self, parent, candidates, *, explore=True):
        words = set(re.findall(r'\w+', parent.code.lower() + ' ' + parent.idea.lower()))
        def overlap(tid):
            spec = self.state['library'][tid]['spec']
            terms = set(re.findall(r'\w+', ' '.join(
                [spec['name'], spec['applicability'], spec['operation']]).lower()))
            return len(words & terms) / max(1, len(terms))
        ranked = sorted(candidates, key=lambda t: (-overlap(t), t))
        size = self.config.retrieval_size
        head = ranked[:max(1, size-1)]
        tail = [x for x in ranked if x not in head]
        if tail and len(head) < size:
            head.append(self.progress.rng['reference'].choice(tail) if explore else tail[0])
        return head

    def _start(self):
        if self.budget_used >= self.config.budget or self.state['requests'] >= self.config.max_model_calls:
            self.state['stop_reason'] = 'candidate_budget' if self.budget_used >= self.config.budget else 'model_call_budget'
            self.progress.phase = 'freeze'
            self._save()
            return
        roots = [p for p in self.archive.values() if p.parent_id is None]
        if self.phase == 'roots' and (len(roots) >= self.config.roots or self.progress.init_attempts >= self.config.init_attempt_limit):
            self.progress.phase = 'search' if roots else 'freeze'
        work = dict(attempt=self.budget_used + 1, parent=None, reference=None, route=None,
                    mode='Init', stage='generate', child=None, transform=None, bindings=[])
        if self.phase == 'freeze':
            self._save()
            return
        if self.phase == 'search':
            routes = sorted({p.request['route'] for p in self.archive.values()})
            counts = Counter(a['route'] for a in self.state['attempts'] if a['route'] in routes)
            quota = math.floor(self.config.route_floor * (sum(counts.values()) + 1) / len(routes))
            behind = [r for r in routes if counts[r] < quota]
            route = self.progress.rng['parent'].choice(behind) if behind else self.progress.rng['parent'].choices(routes, distribution(
                routes, self.state['attempts'], 'route', self.config.recent_window, self.config.route_floor))[0]
            nodes = [p for p in self.archive.values() if p.request['route'] == route]
            beta, _ = temperature([p.fitness for p in nodes])
            best = min(p.fitness for p in nodes)
            parent = self.progress.rng['parent'].choices(nodes, [math.exp(-beta*(p.fitness-best)) for p in nodes])[0]
            candidates = eligible(parent, self.programs, self.state['library'], self.state['attempts'])
            modes = ['Open'] + (['Transform'] if candidates else []) + (['Crossover'] if len(self.archive) > 1 else [])
            weights = distribution(modes, self.state['attempts'], 'mode', self.config.recent_window)
            weights = [(1-self.config.open_floor)*w + (self.config.open_floor if m == 'Open' else 0)
                       for m, w in zip(modes, weights)]
            mode = self.progress.rng['action'].choices(modes, weights)[0]
            previous = [a for a in self.state['attempts'] if a['mode'] != 'Init']
            if sum(a['mode'] == 'Open' for a in previous) < math.floor(self.config.open_floor * (len(previous)+1)):
                mode = 'Open'
            reference = None
            if mode == 'Crossover':
                refs = [p for p in self.archive.values() if p.id != parent.id]
                other = [p for p in refs if p.request['route'] != route]
                reference = self.progress.rng['reference'].choice(other or refs).id
            work.update(parent=parent.id, reference=reference, route=route, mode=mode,
                        mode_probabilities=dict(zip(modes, weights)))
            if mode == 'Transform':
                work.update(stage='match', candidates=self._shortlist(parent, candidates))
        else:
            self.progress.init_attempts += 1
        self.progress.generations += 1
        self.state['work'] = work
        self._save()

    def _context(self, work):
        if work['parent'] is None:
            return [dict(existing_roots=[node_view(p) for p in self.archive.values()])]
        parent = self.programs[work['parent']]
        contexts = [dict(parent_trajectory=trajectory(parent, self.programs, self.state['attempts'], self.config.history_depth))]
        if work['reference'] is not None:
            ref = self.programs[work['reference']]
            common = ancestors(parent.id, self.programs) & ancestors(ref.id, self.programs)
            merged = sorted(ancestors(parent.id, self.programs) | ancestors(ref.id, self.programs))
            contexts = [dict(common_ancestor_ids=sorted(common),
                             merged_history=[dict(id=i, parent=self.programs[i].parent_id,
                                 reference=self.programs[i].reference_id, trace=self.programs[i].request.get('trace', {}),
                                 score=self.programs[i].fitness) for i in merged[-self.config.history_depth:]],
                             parent_trials=trajectory(parent, self.programs, self.state['attempts'], self.config.history_depth)['trials'],
                             reference_trials=trajectory(ref, self.programs, self.state['attempts'], self.config.history_depth)['trials'])]
        return contexts

    def _generate_candidate(self, work):
        required = dict(mode=work['mode'])
        if work['parent'] is not None:
            required['CURRENT_PARENT'] = node_view(self.programs[work['parent']])
        if work['reference'] is not None:
            required['REFERENCE_PARENT'] = node_view(self.programs[work['reference']])
        contexts = self._context(work)
        if self.state['library'] and work['parent'] is not None:
            ids = self._shortlist(self.programs[work['parent']], list(self.state['library']), explore=False)
            contexts.append(dict(known_operations=self._catalog(ids)))
        reply = self._ask('generate', required, contexts)
        if not isinstance(reply.get('code'), str) or not reply['code'].strip():
            raise ValueError('missing complete program')
        self._new_program(reply['code'], reply)

    def _new_program(self, code, reply):
        work = self.state['work']
        parent = self.programs.get(work['parent'])
        p = Program(id=len(self.programs)+1, code=code, idea=str(reply.get('design', '')),
                    analysis=str(reply.get('analysis', '')), action=work['mode'], generation=self.budget_used,
                    parent_id=work['parent'], reference_id=work['reference'], depth=parent.depth+1 if parent else 0,
                    request_id=self.progress.model_calls,
                    request=dict(mode=work['mode'], route=work['route'], transformation=work['transform'],
                                 version=work.get('version'), bindings=work['bindings']))
        self.programs[p.id] = p
        work['child'] = p.id
        if work['mode'] == 'Transform':
            try:
                validate_source(code, self.template)
            except (ValueError, SyntaxError) as exc:
                p.status = 'implementation_failed'
                p.failure = dict(kind='invalid_source', error=str(exc))
                work['outcome'] = 'implementation_failed'
                self._advance('describe', program=p, new=True)
                return
        self._advance('audit' if work['mode'] == 'Transform' else 'evaluate', program=p, new=True, candidate=False)

    def _match(self, work):
        parent = self.programs[work['parent']]
        reply = self._ask('match', dict(CURRENT_PARENT=node_view(parent), candidates=self._catalog(work['candidates'])), self._context(work))
        matches = reply.get('matches')
        if not isinstance(matches, list):
            raise ValueError('matches must be a list')
        grounded = {}
        for match in matches:
            if not isinstance(match, dict) or match.get('id') not in work['candidates'] or match['id'] in grounded:
                raise ValueError('unknown or duplicate match')
            grounded[match['id']] = bindings(match.get('bindings'), parent.code)
        for tid in work['candidates']:
            if tid not in grounded:
                self._record_operator(tid, 'not_applicable', work, regions=[])
        if not grounded:
            work['outcome'] = 'not_applicable'
            self._advance('revise')
            return
        ids = list(grounded)
        tid = self.progress.rng['action'].choices(ids, [evidence_rate(
            self.state['library'][t]['records'], route=work['route']) * self.state['library'][t].get('priority', 1)
            for t in ids])[0]
        work.update(transform=tid, version=self.state['library'][tid]['version'],
                    spec=deepcopy(self.state['library'][tid]['spec']), bindings=grounded[tid])
        self._advance('edit')

    def _edit(self, work):
        parent = self.programs[work['parent']]
        t = self.state['library'][work['transform']]
        examples = [self._evidence(i) for i in t['examples'][-2:] if i in self.programs] or t.get('source_examples', [])[-2:]
        reply = self._ask('edit', dict(CURRENT_PARENT=node_view(parent), spec=work['spec'], bindings=work['bindings'],
                          use_records=self._catalog([work['transform']])[0]['recent']),
                          self._context(work) + [dict(source_examples=examples)])
        code = apply_edits(parent.code, reply.get('edits'), work['bindings'])
        work['edits'] = reply['edits']
        self._new_program(code, reply)

    def _audit(self, work):
        parent, child = self.programs[work['parent']], self.programs[work['child']]
        reply = self._ask('audit', dict(CURRENT_PARENT=dict(id=parent.id, code=parent.code),
            candidate=child.code, actual_diff=code_diff(parent.code, child.code), spec=work['spec'], bindings=work['bindings']))
        keys = ('operation_ok', 'dependencies_ok', 'preservation_ok', 'parameters_ok')
        if any(type(reply.get(k)) is not bool for k in keys) or not isinstance(reply.get('summary'), str):
            raise ValueError('audit needs explicit checks and actual-change summary')
        work['audit'] = reply
        child.request['audit'] = reply
        if all(reply[k] for k in keys):
            self._advance('evaluate', program=child)
        else:
            work['outcome'] = 'implementation_failed'
            child.status = 'implementation_failed'
            child.failure = dict(kind='implementation_failed', error=str(reply.get('reason', 'audit rejected')))
            self._advance('describe', program=child, candidate=True)

    def _evaluate(self, work):
        p = self.programs[work['child']]
        receipt = self.run_dir / 'evaluation_receipts' / f'{p.id}.json'
        saved = read_json(receipt)
        if saved:
            if saved['source_sha256'] != digest(p.code):
                raise ValueError('evaluation receipt source mismatch')
            p.__dict__.update(saved['program'])
            rows = saved['evaluations']
        else:
            rows = p.evaluate(self.training, self.template)
            write_json(receipt, dict(source_sha256=digest(p.code), program=asdict(p), evaluations=rows))
        parent = self.programs.get(work['parent'])
        work['outcome'] = ('root' if parent is None else 'improved' if p.valid and better(p.fitness, parent.fitness)
                           else 'no_improvement')
        work['failure'] = p.failure['kind'] if p.failure else None
        if parent is not None and work['failure'] in ('invalid_source', 'runtime_error', 'invalid_output'):
            work['outcome'] = 'implementation_failed'
        self._advance('describe', program=p, evaluations=rows, candidate=True)

    def _describe(self, work):
        p = self.programs[work['child']]
        parent = self.programs.get(work['parent'])
        known = sorted({n.request['trace']['key'] for n in self.programs.values()
                        if n.parent_id is not None and 'trace' in n.request})
        required = dict(candidate=node_view(p), parent=node_view(parent) if parent else None,
                        actual_diff=code_diff(parent.code, p.code) if parent else '',
                        routes=self.state['routes'], existing_categories=known[-40:])
        reply = self._ask('describe', required, self._context(work) if parent else [])
        if any(not isinstance(reply.get(k), str) or not reply[k].strip() for k in ('change', 'key', 'lineage')):
            raise ValueError('missing actual-change or lineage summary')
        route = reply.get('route')
        if route is not None and route not in self.state['routes']:
            raise ValueError('unknown route')
        if route is None:
            name = str(reply.get('route_name') or 'unclassified')[:160]
            route = next((k for k, v in self.state['routes'].items() if v['name'].casefold() == name.casefold()), None)
            if route is None and len(self.state['routes']) < self.config.max_routes:
                route = f'r{len(self.state["routes"])+1}'
                self.state['routes'][route] = dict(name=name)
            route = route or work['route'] or next(iter(self.state['routes']))
        p.request.update(route=route, trace=dict(change=reply['change'][:2000], key=reply['key'][:120],
                         lineage=reply['lineage'][:2000], features=[str(x)[:120] for x in reply.get('features', [])][:8]))
        self._advance('learn' if parent is not None and work['mode'] in ('Open', 'Crossover') else 'feedback', program=p)

    def _evidence(self, node_id):
        p = self.programs[node_id]
        parent = self.programs.get(p.parent_id)
        return dict(parent=node_view(parent) if parent else None, child=node_view(p),
                    diff=code_diff(parent.code, p.code) if parent else '',
                    improves=bool(parent and p.valid and parent.valid and better(p.fitness, parent.fitness)))

    def _learn(self, work):
        p = self.programs[work['child']]
        key = p.request.get('trace', {}).get('key')
        similar = [n for n in self.programs.values() if n.parent_id is not None and n.id != p.id
                   and key and n.request.get('trace', {}).get('key') == key]
        cross_route = any(n.request.get('route') != p.request['route'] for n in similar)
        if work['outcome'] != 'improved' and not cross_route and not self.state['library']:
            self._advance('feedback')
            return
        examples = similar[-3:] + [p]
        ids = list(self.state['library'])
        catalog = [dict(id=tid, **{k: t['spec'][k] for k in ('name', 'applicability', 'operation')})
                   for tid, t in self.state['library'].items()]
        reply = self._ask('learn', dict(evaluated_edits=[self._evidence(n.id) for n in examples],
                          library=catalog, can_add=len(self.state['library']) < self.config.library_capacity))
        kind = reply.get('kind')
        work['learning_decision'] = dict(kind=kind, reason=reply.get('reason'))
        regions = changed_regions(self.programs[work['parent']].code, p.code) if kind in ('new', 'attach') else []
        if kind == 'new':
            if not any(self._evidence(n.id)['improves'] for n in examples):
                raise ValueError('new operation has no improving evidence')
            spec = specification(reply.get('spec'), self.state['library'])
            if spec['requires']:
                raise ValueError('initial induction cannot establish transformation prerequisites')
            duplicate = next((tid for tid, t in self.state['library'].items()
                              if t['spec']['operation'].casefold() == spec['operation'].casefold()), None)
            if duplicate:
                tid = duplicate
            elif len(self.state['library']) < self.config.library_capacity:
                tid = self._add_transform(spec, [n.id for n in examples])
            else:
                self._advance('feedback')
                return
        elif kind == 'attach':
            tid = reply.get('id')
            if tid not in ids:
                raise ValueError('unlisted existing transformation')
        elif kind == 'none':
            self._advance('feedback')
            return
        else:
            raise ValueError('unknown induction decision')
        work.update(transform=tid, version=self.state['library'][tid]['version'],
                    bindings=regions)
        p.request.update(transformation=tid, version=work['version'], inferred_from_open_edit=True)
        self._advance('feedback', program=p)

    def _add_transform(self, spec, examples, *, family=None, components=()):
        tid = f't{self.state["next_transform"]}'
        self.state['next_transform'] += 1
        self.state['library'][tid] = dict(id=tid, family=family or tid, version=1, spec=spec,
            versions=[dict(version=1, spec=deepcopy(spec), reason='induction')], examples=examples,
            components=list(components), records=[], reviewed=0, priority=1.0)
        return tid

    def _record_operator(self, tid, outcome, work, *, regions=None):
        record = dict(attempt=work['attempt'], parent=work['parent'], child=work['child'],
                      route=work['route'], outcome=outcome, mode=work['mode'], failure=work.get('failure'),
                      bindings=work['bindings'] if regions is None else regions,
                      version=work.get('version', self.state['library'][tid]['version']),
                      already_used=sorted(used_on_path(work['parent'], self.programs, self.state['attempts'], self.state['library'])),
                      features=self.programs[work['parent']].request.get('trace', {}).get('features', []),
                      diagnosis=work.get('audit', {}).get('reason'))
        self.state['library'][tid]['records'].append(record)

    def _feedback(self, work):
        tid = work['transform']
        if tid:
            self._record_operator(tid, work['outcome'], work)
        self._advance('revise')

    def _revise(self, work):
        pending = [tid for tid, t in self.state['library'].items()
                   if len(t['records']) - t['reviewed'] >= self.config.revision_interval]
        if not pending:
            self._advance('finish')
            return
        tid = pending[0]
        transform = self.state['library'][tid]
        records = transform['records']
        sample = []
        for outcome in ('improved', 'no_improvement', 'implementation_failed', 'not_applicable'):
            sample.extend([r for r in records if r['outcome'] == outcome][-2:])
        examples = [dict(record=r, parent=node_view(self.programs[r['parent']]),
                         child=node_view(self.programs[r['child']]) if r.get('child') else None)
                    for r in sample if r.get('parent') in self.programs and not r.get('external')]
        reply = self._ask('revise', dict(transform=self._catalog([tid])[0], examples=examples,
                                        related_operations=self._catalog(self._shortlist(self.programs[work['parent']], list(self.state['library']), explore=False))
                                        if work['parent'] else self._catalog([tid])))
        self._apply_revision(tid, reply)
        self.state['library'][tid]['reviewed'] = len(records)
        self._advance('revise')

    def _apply_revision(self, tid, reply):
        kind, specs = reply.get('kind'), reply.get('specs', [])
        library = self.state['library']
        transform = library[tid]
        counts = {'keep': 0, 'revise': 1, 'split': 2, 'compose': 1}
        if kind not in counts or not isinstance(specs, list) or len(specs) != counts[kind]:
            raise ValueError('invalid library revision')
        parsed = [specification(s, library, own_id=tid if kind == 'revise' else None) for s in specs]
        for spec in parsed:
            for dep in set(spec['requires']) - set(transform['spec']['requires']):
                positives = [r for r in transform['records'] if r['outcome'] == 'improved']
                negatives = [r for r in transform['records'] if r['outcome'] == 'no_improvement']
                if not (positives and all(dep in r['already_used'] for r in positives)
                        and any(dep not in r['already_used'] for r in negatives)):
                    raise ValueError('new prerequisite lacks contrasting observations')
        components = reply.get('components', [])
        if kind == 'compose' and (not isinstance(components, list) or tid not in components
                or len(set(components)) < 2 or any(x not in library for x in components)):
            raise ValueError('composition needs the reviewed operation and another known operation')
        if kind in ('split', 'compose') and len(library) + len(parsed) > self.config.library_capacity:
            raise ValueError('library capacity reached; retain existing operation')
        if kind == 'revise':
            transform['version'] += 1
            transform['spec'] = parsed[0]
            transform['versions'].append(dict(version=transform['version'], spec=deepcopy(parsed[0]), reason=str(reply.get('reason', ''))))
        elif kind in ('split', 'compose'):
            children = [self._add_transform(spec, transform['examples'][:],
                        family=transform['family'] if kind == 'split' else None,
                        components=components if kind == 'compose' else []) for spec in parsed]
            transform['versions'].append(dict(version=transform['version'], kind=kind, children=children,
                                              reason=str(reply.get('reason', ''))))
            transform['priority'] *= 0.5
        transform['last_review'] = deepcopy(reply)

    def _finish_attempt(self, work):
        self.state['reply'] = None
        self.state['attempts'].append(deepcopy({k: v for k, v in work.items() if k not in ('stage', 'spec', 'edits')}))
        self.state['work'] = None
        self._save()

    def _stage_error(self, exc):
        work = self.state['work']
        work.setdefault('stage_errors', []).append(dict(stage=work['stage'], error=str(exc)))
        stage = work['stage']
        p = self.programs.get(work['child'])
        if stage == 'describe':
            route = work['route'] or next(iter(self.state['routes']), 'r1')
            self.state['routes'].setdefault(route, dict(name='unclassified'))
            p.request.update(route=route, trace=dict(change='summary unavailable; use actual diff',
                             key='unclassified', lineage='', features=[]))
            self._advance('feedback', program=p)
        elif stage == 'learn':
            self._advance('feedback')
        elif stage == 'revise':
            # Do not spend the entire remaining budget repeatedly reviewing one malformed reply.
            for t in self.state['library'].values():
                if len(t['records'])-t['reviewed'] >= self.config.revision_interval:
                    t['reviewed'] = len(t['records'])
                    break
            self._advance('revise')
        else:
            work['outcome'] = 'implementation_failed' if work['mode'] == 'Transform' else 'delivery_failed'
            if p is not None:
                p.status = 'implementation_failed'
                p.failure = dict(kind='implementation_failed', error=str(exc))
            self._advance('describe' if p else 'feedback', **({'program': p, 'candidate': True} if p else {}))

    def _import_library(self, path):
        payload = read_json(path)
        if not isinstance(payload, dict) or payload.get('schema') != 13 or payload.get('task') != self.task:
            raise ValueError('library must be a V13 export for the same task')
        library = deepcopy(payload['library'])
        if not isinstance(library, dict):
            raise ValueError('library must be an object')
        if len(library) > self.config.library_capacity:
            raise ValueError('imported library exceeds capacity')
        for tid, t in library.items():
            if not isinstance(t, dict) or tid != t.get('id') or not re.fullmatch(r't[1-9][0-9]*', tid):
                raise ValueError('invalid library identifier')
            if (t.get('family') not in library or not isinstance(t.get('components'), list)
                    or any(c not in library for c in t['components'])
                    or type(t.get('version')) is not int or t['version'] < 1
                    or not isinstance(t.get('versions'), list) or not isinstance(t.get('records'), list)
                    or not isinstance(t.get('priority'), (int, float)) or not 0 < t['priority'] <= 1
                    or type(t.get('reviewed')) is not int or not 0 <= t['reviewed'] <= len(t['records'])):
                raise ValueError('invalid library metadata')
            if any(not isinstance(r, dict) or r.get('outcome') not in
                   ('improved', 'no_improvement', 'implementation_failed', 'not_applicable')
                   or not isinstance(r.get('already_used'), list) for r in t['records']):
                raise ValueError('invalid library use record')
        for tid, t in library.items():
            t['spec'] = specification(t['spec'], library, own_id=tid)
            t['examples'] = []
            for record in t['records']:
                record.update(external=True, route=None)
        self.state['library'] = library
        self.state['next_transform'] = max((int(k[1:]) for k in library), default=0)+1
        self.state['import'] = dict(path=str(path), sha256=digest(path.read_text()), source=payload.get('run'))

    def _export_library(self):
        payload = dict(schema=13, task=self.task, run=str(self.run_dir), library=deepcopy(self.state['library']))
        for t in payload['library'].values():
            t['source_examples'] = ([self._evidence(i) for i in t['examples'][-3:] if i in self.programs]
                                    or t.get('source_examples', []))
        write_json(self.run_dir / 'transformations.json', payload)

    def _summary(self, status, error=None):
        result = super()._summary(status, error)
        result.update(method=self.METHOD, model_call_limit=self.config.max_model_calls,
                      reserved_model_calls=self.state['requests'], stages=self.state['stages'],
                      uncertain_calls=len(self.state['interrupted']), missing_usage=self.state['missing_usage'],
                      measured_input_tokens=self.state['measured_input_tokens'], measured_output_tokens=self.state['measured_output_tokens'],
                      transformations=len(self.state['library']), routes=len(self.state['routes']), stop_reason=self.state['stop_reason'])
        self.facts.save_summary(result)
        self._export_library()
        return result

    def run(self):
        from experiments.infra.runner import baseline_run_lock
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with baseline_run_lock(self.run_dir):
            if self.phase in ('finished', 'no_valid_root'):
                saved = self.facts.load_summary()
                if saved and 'reserved_model_calls' in saved and (self.run_dir / 'transformations.json').exists():
                    return saved
                return self._summary(self.phase)
            self._save()
            try:
                while self.phase in ('roots', 'search', 'freeze'):
                    if self.phase == 'freeze':
                        self._freeze()
                    elif self.state['work'] is None:
                        self._start()
                    else:
                        stage = self.state['work']['stage']
                        operation = dict(generate=self._generate_candidate, match=self._match, edit=self._edit,
                            audit=self._audit, evaluate=self._evaluate, describe=self._describe,
                            learn=self._learn, feedback=self._feedback, revise=self._revise, finish=self._finish_attempt)[stage]
                        try:
                            operation(self.state['work'])
                        except (ValueError, SyntaxError, TypeError, KeyError) as exc:
                            if stage in ('evaluate', 'feedback', 'finish'):
                                raise
                            self._stage_error(exc)
            except CallBudgetExhausted:
                self.state['stop_reason'] = 'model_call_budget'
                self.state['unfinished_work'] = self.state['work']
                self.state['work'] = None
                self.progress.phase = 'freeze'
                self._save()
                self._freeze()
            except (ModelCallError, SchedulerError) as exc:
                return self._summary('service_unavailable', str(exc))
            return self._summary(self.phase)

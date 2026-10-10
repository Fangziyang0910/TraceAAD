"""Shared finite development over the existing evaluator and candidate archive."""

from collections import Counter
import re

from traceaad.common.delivery import DeliveryError
from traceaad.common.canonical import canonical, key
from traceaad.common.history import code_diff
from traceaad.common.prompts import ContextTooLong
from traceaad.common.search import Search
from traceaad.v10_21.traceaad import TraceAADV1021
from .config import Config
from .policy import SOURCES, declarations, frontier, score_vector, stable_key, values
from .prompts import BaseContextTooLong, PromptBuilder
from .delivery import clean_response, edit_design, mode, parse_candidate


class TraceAADV1024(TraceAADV1021):
    METHOD = 'v1024'
    Config = Config
    PromptBuilder = PromptBuilder
    AUTO_REPAIR = False
    RECORD_EXPLORATIONS = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity['delivery_protocol'] = 'v1024-atomic-edit-1'
        self.identity['development_protocol'] = 'v1024-shared-development-3'
        saved = self.facts.state or {}
        if saved and (saved.get('identity') != self.identity or 'v1024' not in saved):
            raise ValueError('V10.24 resume requires the same configuration, task and protocol')
        self.policy = saved.get('v1024', {
            'units': [], 'blocks': [], 'active': None, 'next_decision': 'new', 'excluded': [],
        })
        self.source_ids = {self._source_identity(p['code']): p['id'] for p in self.programs.values()}

    @staticmethod
    def _source_identity(code):
        try:
            return key(canonical(code))
        except (SyntaxError, ValueError):
            return key(code.rstrip() + '\n')

    def _normalize_source(self, code):
        # Keep untouched source text; normalize only the identity used for deduplication.
        code = code.replace('\r\n', '\n')
        return code if code.endswith('\n') else code + '\n'

    def _known_program(self, code, source_key):
        return self.programs.get(self.source_ids.get(self._source_identity(code)))

    def _parse_candidate(self, request, details, parent):
        if request.get('edit_base_mismatch'):
            raise DeliveryError('edit: Base conflicts with the host-bound program; edit the bound base or return complete Code')
        return parse_candidate(details.get('content', ''), details.get('finish_reason'), self.template,
                               parent['code'] if parent else None)

    def _state(self):
        return {**super()._state(), 'v1024': self.policy}

    @property
    def unit(self):
        return self.policy['units'][self.block['unit_id'] - 1]

    @property
    def block(self):
        return self.policy['blocks'][self.policy['active'] - 1]

    def _save(self, **record):
        attempt = record.get('attempt')
        if attempt:
            if 'submitted_edit_design' in attempt:
                attempt['idea'] = attempt.pop('submitted_edit_design')
                if record.get('program'):
                    record['program']['idea'] = attempt['idea']
            if attempt['status'] != 'delivery_failed':
                attempt.pop('failed_edit_submission', None)
            calls = attempt['calls']
            attempt['delivery_cost'] = {'model_calls': len(calls),
                **{name: sum(c['usage'][field] for c in calls)
                   if all(isinstance(c['usage'].get(field), int) for c in calls) else None
                   for name, field in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]}}
            attempt['entered_evaluation'] = bool(record.get('evaluations'))
        if attempt and attempt.get('unit_id'):
            self._record_trial(attempt, record.get('program'), record.get('evaluations', []))
        # Event deltas suffice to reconstruct the policy; resume.json is only its checkpoint.
        changed = self.policy['blocks'][-1] if self.policy['blocks'] else None
        record['development'] = {
            'scheduler': {k: v for k, v in self.policy.items() if k not in ('units', 'blocks')},
            'unit': self.policy['units'][changed['unit_id'] - 1] if changed else None,
            'block': changed,
        }
        super()._save(**record)
        if record.get('program'):
            p = record['program']
            self.source_ids[self._source_identity(p['code'])] = p['id']

    def _frontier(self):
        roots = [p['id'] for p in self.archive.values() if self._is_root(p)]
        return frontier(self.programs, self.facts.evaluations, roots, self.policy['units'],
                        self.config.frontier_groups, self.config.seed, self.policy['excluded'])

    def _parent(self, groups):
        weights = [1 / g['rank'] for g in groups]
        group = self.parent_rng.choices(groups, weights)[0]
        counts = Counter(u['anchor_id'] for u in self.policy['units'])
        lowest = min(counts[pid] for pid in group['members'])
        members = [pid for pid in group['members'] if counts[pid] == lowest]
        pid = self.parent_rng.choice(members)
        return pid, {'groups': [{'key': g['key'], 'members': g['members'], 'rank': g['rank']} for g in groups],
                     'group_key': group['key'], 'group_probability': (1 / group['rank']) / sum(weights),
                     'eligible_members': members, 'member_probability': 1 / len(members)}

    def _donor(self, anchor):
        base = values(score_vector(self.programs[anchor], self.facts.evaluations))
        candidates = [p for p in self.archive.values() if p['key'] != self.programs[anchor]['key']]
        if not candidates:
            return None, None
        complement = {}
        for p in candidates:
            scores = values(score_vector(p, self.facts.evaluations))
            if len(scores) != len(base):
                raise ValueError('donor and anchor must use the same training instances')
            complement[p['id']] = sum(max(a - b, 0) for a, b in zip(base, scores)) / len(base)
        ranked = sorted(complement, key=lambda i: (-complement[i], stable_key(self.config.seed, i)))
        cutoff = complement[ranked[min(3, len(ranked)) - 1]]
        top = [i for i in ranked if complement[i] >= cutoff]
        rest = [i for i in ranked if complement[i] < cutoff]
        pool = self.reference_rng.choice([top, rest]) if rest else top
        pid = self.reference_rng.choice(pool)
        return pid, {'complementarity': {str(i): c for i, c in complement.items()},
                     'pool': pool, 'pool_probability': .5 if rest else 1.,
                     'member_probability': 1 / len(pool)}

    def _allocate(self):
        remaining = self.config.budget - self.attempts
        if remaining <= 0:
            self.progress.phase = 'freeze'
            self._save()
            return
        groups = self._frontier()
        waiting = [u for u in self.policy['units'] if u['status'] == 'waiting']
        # A local opener that cannot fit blocks only this source/base pair, not the program.
        blocked = {(u['source'], u['anchor_id']) for u in self.policy['units']
                   if not u['trial_ids'] and u['closure_reason'] == 'local_context_too_long'}
        source = None
        for offset in range(len(SOURCES)):
            candidate = SOURCES[(len(self.policy['units']) + offset) % len(SOURCES)]
            available = [{**g, 'members': [pid for pid in g['members'] if (candidate, pid) not in blocked]}
                         for g in groups]
            available = [g for g in available if g['members']]
            if available:
                source, groups = candidate, available
                break
        if source is None:
            groups = []
        if not groups and not waiting:
            self.progress.phase = 'freeze'
            self._save()
            return
        preferred = self.policy['next_decision']
        continuing = bool(waiting) and (preferred == 'continuation' or not groups)
        purpose = 'continuation' if continuing else 'new'
        if continuing:
            unit = min(waiting, key=lambda u: (u['waiting_since'], u['id']))
            reason = 'oldest_waiting'
        else:
            anchor, selection = self._parent(groups)
            donor, donor_selection = self._donor(anchor) if source == 'Crossover' else (None, None)
            # The producer precedes other units that merely used the same program.
            linked = [u for u in self.policy['units'] if any(
                self.attempts_table[i]['program_id'] == anchor for i in u['trial_ids'])
                or u['anchor_id'] == anchor]
            linked.sort(key=lambda u: (self.programs[anchor]['attempt_id'] not in u['trial_ids'], -u['id']))
            unit = {'id': len(self.policy['units']) + 1, 'source': source,
                    'request_key': f'{self.task}:unit:{len(self.policy["units"]) + 1}',
                    'question': None,
                    'anchor_id': anchor, 'proposal_id': None, 'champion_id': None, 'worktip_id': anchor,
                    'pending_failure': None, 'donor_id': donor, 'trial_ids': [], 'block_ids': [],
                    'linked_unit_ids': [u['id'] for u in linked], 'status': 'active',
                    'selection': selection, 'donor_selection': donor_selection,
                    'closure_reason': None, 'waiting_since': None, 'events': {}}
            self.policy['units'].append(unit)
            reason = 'source_cycle' if source != 'Crossover' or donor else 'no_distinct_reference'
        self.policy['next_decision'] = 'new' if continuing else 'continuation'
        unit['status'] = 'active'
        block = {'id': len(self.policy['blocks']) + 1, 'unit_id': unit['id'],
                 'purpose': purpose, 'preferred': preferred, 'allocation_reason': reason,
                 'limit': min(self.config.block_size, remaining), 'spent': 0,
                 'start_frontier': min(p['fitness'] for p in self.archive.values()), 'best_new_id': None,
                 'start_after': self.attempts, 'closed_after': None, 'frontier_gain': None,
                 'eligible_units': [u['id'] for u in waiting]}
        self.policy['blocks'].append(block)
        self.policy['active'] = block['id']
        unit['block_ids'].append(block['id'])
        self._save()

    def _prepare_attempt(self, request, details, parent):
        request['delivery_mode'] = mode(details.get('content'))
        if request['delivery_mode'] == 'edit':
            request['failed_edit_submission'] = clean_response(details.get('content'))
            request['submitted_edit_design'] = edit_design(details.get('content'))
        if not request.get('unit_id'):
            request['edit_base_id'] = parent['id'] if parent else None
            return parent
        fields = declarations(details.get('content'))
        declared = fields.get('base', '')
        pid = int(declared) if re.fullmatch(r'\d+', declared) else None
        chosen = pid if pid in request['available_bases'] else request['system_default_base_id']
        if request['delivery_mode'] == 'edit':
            request['edit_base_mismatch'] = pid is not None and pid != request['system_default_base_id']
            chosen = request['system_default_base_id']
        request['edit_base_id'] = chosen if request['delivery_mode'] == 'edit' else None
        request.update(declared_base_id=pid, base_fallback_reason=None if chosen == pid else 'missing_or_unavailable_base',
                       control=fields, recorded_diff_base_id=chosen)
        return self.programs[chosen]

    def _record_trial(self, a, new, evaluations):
        unit, block = self.unit, self.block
        assert a['unit_id'] == unit['id'] and block['spent'] < block['limit']
        unit['trial_ids'].append(a['id'])
        block['spent'] += 1
        p = new or self.programs.get(a['program_id'])
        fields = a['control']
        if not unit['question']:
            unit['question'] = fields.get('question') or fields.get('effect')
        if unit['proposal_id'] is None:
            if p:
                unit['proposal_id'] = p['id']
                unit['events']['first_proposal'] = a['id']
        if p:
            base = self.programs[a['parent_id']]
            a['actual_diff'] = code_diff(base['code'], p['code'])
            if p['valid']:
                unit['worktip_id'] = p['id']
                champion = self.programs.get(unit['champion_id'])
                if champion is None or p['fitness'] < champion['fitness']:
                    unit['champion_id'] = p['id']
                best = self.programs.get(block['best_new_id'])
                if new and (best is None or p['fitness'] < best['fitness']):
                    block['best_new_id'] = p['id']
                unit['events'].setdefault('first_valid', a['id'])
                if p['fitness'] < self.programs[unit['anchor_id']]['fitness']:
                    unit['events'].setdefault('first_beat_anchor', a['id'])
                if p['fitness'] < block['start_frontier']:
                    unit['events'].setdefault('first_global_gain', a['id'])
                roots = [p['id'] for p in self.archive.values() if self._is_root(p)]
                groups = frontier({**self.programs, p['id']: p}, self.facts.evaluations + evaluations,
                                  roots, self.policy['units'], self.config.frontier_groups,
                                  self.config.seed, self.policy['excluded'])
                if any(unit['champion_id'] in g['members'] for g in groups):
                    unit['events'].setdefault('first_frontier', a['id'])
                unit['pending_failure'] = None
            else:
                unit['pending_failure'] = p['id']
            if not new:
                a['duplicate_of'] = p['id']
                related = [u['id'] for u in self.policy['units'] if u['id'] != unit['id'] and p['id'] in
                           (u['anchor_id'], u['proposal_id'], u['champion_id'], u['worktip_id'])]
                unit['linked_unit_ids'] = sorted(set(unit['linked_unit_ids'] + related))
        if re.match(r'^change_request\b', fields.get('status', ''), re.I):
            unit.update(status='closed', closure_reason='change_request')

    def _close_block(self, reason):
        block, unit = self.block, self.unit
        best = self.programs.get(block['best_new_id'])
        gain = max(0., block['start_frontier'] - best['fitness']) if best else 0.
        block.update(closed_after=self.attempts, completion_reason=reason, frontier_gain=gain,
                     gain_per_attempt=gain / block['spent'] if block['spent'] else None)
        champion = self.programs.get(unit['champion_id'])
        unit['anchor_delta'] = self.programs[unit['anchor_id']]['fitness'] - champion['fitness'] if champion else None
        admitted = {pid for g in self._frontier() for pid in g['members']}
        if unit['champion_id'] in admitted:
            unit['events'].setdefault('first_frontier', self.attempts)
        terminal = unit['status'] == 'closed' or reason != 'block_complete' or len(unit['block_ids']) >= 2
        if terminal:
            unit.update(status='closed', closure_reason=unit['closure_reason'] or
                        ('unit_limit' if len(unit['block_ids']) >= 2 and reason == 'block_complete' else reason))
        else:
            unit.update(status='waiting', waiting_since=self.attempts)
        self.policy['active'] = None
        self._save()

    def _search(self):
        if self.policy['active'] is not None:
            if self.unit['status'] == 'closed':
                self._close_block(self.unit['closure_reason'])
                return
            if self.block['spent'] == self.block['limit']:
                self._close_block('block_complete')
                return
        if self.attempts >= self.config.budget:
            self.progress.phase = 'freeze'
            self._save()
            return
        if self.policy['active'] is None:
            self._allocate()
            return
        unit, block = self.unit, self.block
        failed = unit['pending_failure']
        first = not unit['trial_ids']
        action = ('Repair' if failed else
                  unit['source'] if first and (unit['source'] != 'Crossover' or unit['donor_id']) else
                  'Refine' if first else 'Develop')
        if unit['trial_ids'] and self.attempts_table[unit['trial_ids'][-1]]['status'] == 'delivery_failed' and not failed:
            action = self.attempts_table[unit['trial_ids'][-1]]['action']
        default = failed or unit['worktip_id']
        linked = [self.policy['units'][i - 1] for i in unit['linked_unit_ids']]
        linked.sort(key=lambda u: (self.programs[unit['anchor_id']]['attempt_id'] not in u['trial_ids'], -u['id']))
        paired = ([values(score_vector(self.programs[pid], self.facts.evaluations))
                   for pid in (unit['anchor_id'], unit['donor_id'])] if unit['donor_id'] else None)
        try:
            request = self.prompts.request(unit, block, action, default, linked, paired)
        except BaseContextTooLong:
            if default not in self.policy['excluded']:
                self.policy['excluded'].append(default)
            self._close_block('base_context_too_long')
            return
        except ContextTooLong:
            self._close_block('local_context_too_long')
            return
        request.update(unit_id=unit['id'], block_id=block['id'], resource_purpose=block['purpose'],
                       request_key=unit['request_key'], request_statement=unit['question'])
        self._attempt(request, parent=self.programs[default],
                      reference=self.programs.get(unit['donor_id']) if unit['donor_id'] in request['material_ids'] else None,
                      repair_of=failed)

    def _summary(self, status, error=None):
        summary = Search._summary(self, status, error)
        costs = Counter()
        for b in self.policy['blocks']:
            costs[b['purpose']] += b['spent']
        summary['development'] = {
            'units': len(self.policy['units']), 'blocks': len(self.policy['blocks']),
            'candidate_cost_by_purpose': dict(costs),
            'remaining_candidates': self.config.budget - self.attempts,
            'frontier_gain': sum(b['frontier_gain'] or 0 for b in self.policy['blocks']),
            'new_question_proposals': sum(u['source'] == 'Explore' and bool(u['trial_ids']) for u in self.policy['units']),
        }
        self.facts.save_summary(summary)
        return summary

"""V10.8: individual opportunities and verified formation-conditioned generation."""

import json
import math
from pathlib import Path
import time
import traceback

from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103, calibrate_beta
from llm4ad.method.traceaad_v10_5.traceaad import atomic_json, ess, UnknownEvaluation
from llm4ad.method.traceaad_v10_7 import traceaad as v107
from llm4ad.method.traceaad_v10_7.sampling import (
    MAX_FIT_ATTEMPTS, _task_base_weights, _weighted_order,
)
from . import trajectory
from .trajectory import digest

SELECTION_POLICY = 'individual_quality_count_v1'
DEDUP_POLICY = 'exact_parent_donor_raw_and_view_v1'
ALLOCATION_ARMS = ('A', 'B', 'C', 'D')


class TraceAADV108(v107.TraceAADV107):
    METHOD = 'v108'
    DISPLAY_NAME = 'V10.8'
    STATE_VERSION = 1081
    OPERATOR_PROBABILITIES = v107.OPERATOR_PROBABILITIES
    TEMPLATE_HASH = trajectory.TEMPLATE_HASH
    CONTEXT_POLICY = trajectory.CONTEXT_POLICY

    def __init__(self, *, history_tokens=8192, allocation_arm='A', **kwargs):
        if allocation_arm not in ALLOCATION_ARMS:
            raise ValueError(f'unknown allocation arm: {allocation_arm}')
        self.allocation_arm = allocation_arm
        if 'max_context_programs' in kwargs:
            raise TypeError('V10.8 uses a formation suffix and one optional Fuse donor')
        super().__init__(history_tokens=history_tokens, **kwargs)
        self.builder = trajectory.TrajectoryBuilder(
            self.llm, self.task_contract, max_tokens=self.builder.max_tokens,
            history_tokens=history_tokens, max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            log_count=lambda record: self._append_record(
                self.run_dir / 'tokenizer_calls.jsonl', record),
        )
        self.mechanism['inherited_unused'] = {'donor_topk': self.donor_topk}
        self.mechanism.pop('max_context_programs')
        self.mechanism.update(
            history_tokens=history_tokens, traj_gens=self.traj_gens,
            context_policy=trajectory.CONTEXT_POLICY,
            selection_policy=SELECTION_POLICY, dedup_policy=DEDUP_POLICY,
            seed=kwargs.get('seed', 0),
            allocation_arm=allocation_arm,
            allocation_policy='quality_concentration_ablation_v1',
            count_exponent=0.5 if allocation_arm == 'A' else 0.0,
            quality_ess_schedule={'A': 'max(ess_minimum,ess_fraction*N)', 'B': 'max(ess_minimum,ess_fraction*N)',
                                  'C': '8', 'D': '32**(1-r)*8**r'}[allocation_arm],
        )
        # Public JSON configuration covers task sizes, panel seeds and evaluator
        # execution limits; large private dataset arrays stay out of checkpoints.
        config = {}
        for key, value in vars(self.evaluation).items():
            if key.startswith('_'):
                continue
            try:
                config[key] = json.loads(json.dumps(value, allow_nan=False))
            except (TypeError, ValueError):
                continue
        self.mechanism['evaluation_config'] = config
        for source in (Path(__file__), Path(trajectory.__file__), Path(v107.__file__)):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())

    def node_distribution(self, nodes, operator):
        scores = [node.fitness for node in nodes]
        progress = min(1.0, self.budget_used / self.budget)
        fraction, minimum = self.ess_fraction, self.ess_minimum
        if self.allocation_arm in ('C', 'D'):
            fraction = 0.0
            minimum = 8.0 if self.allocation_arm == 'C' else 32.0 ** (1-progress) * 8.0 ** progress
        beta, attainable_target, quality_ess = calibrate_beta(
            scores, fraction, minimum)
        maximum = max(scores)
        weights = [math.exp(beta * (node.fitness - maximum)) /
                   (math.sqrt(1 + self.parent_selection_counts.get(node.id, 0))
                    if self.allocation_arm == 'A' else 1.0)
                   for node in nodes]
        total = sum(weights)
        p0 = [w / total for w in weights]
        probabilities = ([0.5 * p + 0.5 / len(nodes) for p in p0]
                         if operator == 'Pivot' else p0)
        return probabilities, {
            'eligible_nodes': len(nodes), 'beta': beta,
            'ess_target': min(len(nodes), max(fraction * len(nodes), minimum)),
            'allocation_arm': self.allocation_arm, 'budget_progress': progress,
            'attainable_ess_target': attainable_target, 'quality_ess': quality_ess,
            'corrected_ess': ess(p0), 'conditional_ess': ess(probabilities),
        }

    def select_donor(self, parent):
        nodes = [n for n in self.tree.all_nodes() if n.id != parent.id]
        weights = _task_base_weights(nodes, {}, 'Fuse')
        attempts = []
        for donor in _weighted_order(nodes, weights, self.rng, MAX_FIT_ATTEMPTS):
            fits = self.builder.fits(parent, 'Fuse', donor)
            attempts.append({'node_id': donor.id, 'fitness': donor.fitness, 'fits': fits})
            if fits:
                return donor, attempts
        return None, attempts

    def _duplicate_inputs(self, code):
        code_hash = digest(code)
        matches = []
        for role in ('parent', 'donor'):
            node_id = self.pending[role + '_id']
            if node_id is None:
                continue
            node = self.tree.nodes[node_id]
            for view, text in (('raw', node.code), ('prompt', self.builder.code_view(node)[0])):
                if code_hash == digest(text.strip()) and code == text.strip():
                    matches.append({'role': role, 'node_id': node_id, 'view': view})
        return matches

    def _schedule(self):
        started = time.monotonic()
        counts_before = len(self.builder._counts)
        parent = donor = None
        requested = operator = 'Init'
        selection, donor_attempts = {}, []
        if len(self.tree.roots) >= self.n_roots:
            requested = self.rng.choices(
                list(self.OPERATOR_PROBABILITIES),
                weights=list(self.OPERATOR_PROBABILITIES.values()))[0]
            operator = requested
            nodes = self.eligible_nodes()
            probabilities, selection = self.node_distribution(nodes, requested)
            index = self.rng.choices(range(len(nodes)), weights=probabilities)[0]
            parent = nodes[index]
            selection.update(
                parent_route='operator_then_node',
                parent_probability=probabilities[index],
                parent_count_before=self.parent_selection_counts.get(parent.id, 0),
            )
            if requested == 'Fuse':
                donor, donor_attempts = self.select_donor(parent)
                if donor is None:
                    operator = 'Refine'
                    selection['fallback_reason'] = (
                        'donor_fit_attempt_limit' if len(donor_attempts) == MAX_FIT_ATTEMPTS
                        else 'no_fitting_other_node_donor')
        text, context = self.builder.build(parent, operator, donor)
        tokens = self.builder.count(text, chat=True)
        # Count only a complete selected request. Tokenizer failures before this
        # point are not generation opportunities; pending recovery restores once.
        if parent is not None:
            self.parent_selection_counts[parent.id] = selection['parent_count_before'] + 1
        return {
            'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
            'requested_operator': requested, 'operator': operator,
            'parent_id': parent.id if parent else None, 'donor_id': donor.id if donor else None,
            'parent_fitness': parent.fitness if parent else None,
            'donor_fitness': donor.fitness if donor else None,
            'best_before': self.tree.best().fitness if self.tree.nodes else None,
            'operator_probabilities': self.OPERATOR_PROBABILITIES, 'selection': selection,
            'donor_attempts': donor_attempts, 'prompt': text, 'prompt_tokens': tokens,
            'prompt_hash': digest(text), 'template_hash': self.TEMPLATE_HASH,
            'context_policy': self.CONTEXT_POLICY, **context,
            'scheduling_seconds': time.monotonic() - started,
            'tokenizer_requests': len(self.builder._counts) - counts_before,
            'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
        }

    def _append_record(self, path, record):
        if path == self.events_path:
            best = self.tree.best().fitness if self.tree.nodes else None
            record['best_so_far'] = best
        super()._append_record(path, record)

    def _save_state(self):
        atomic_json(self.state_path, {
            'version': self.STATE_VERSION, 'mechanism': self.mechanism, 'started_at': self.started_at,
            'nodes': self.tree.to_state(), 'rng_state': list(self.rng.getstate()),
            'parent_selection_counts': self.parent_selection_counts,
            'step_counter': self.step_counter, 'batch_counter': self.step_counter,
            'budget_used': self.budget_used, 'completed_attempts': self.completed_attempts,
            'invalid_streak': self._invalid_streak,
        })

    def _load_state(self):
        state = json.loads(self.state_path.read_text())
        if state.get('version') != self.STATE_VERSION or state.get('mechanism') != self.mechanism:
            raise ValueError(f'checkpoint mechanism/source/backend differs from this {self.DISPLAY_NAME} configuration')
        TraceAADV103._load_state(self)
        self.completed_attempts = state['completed_attempts']
        self._invalid_streak = state['invalid_streak']
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if pending['candidate_id'] <= self.completed_attempts:
                self.pending_path.unlink()
                return
            if pending['candidate_id'] != self.completed_attempts + 1:
                raise ValueError('pending candidate is not the next attempt')
            self.pending = pending
            rng = pending['rng_state']
            self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
            if pending['parent_id'] is not None:
                self.parent_selection_counts[pending['parent_id']] = (
                    pending['selection']['parent_count_before'] + 1)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Reject a foreign configuration before writing any run artifacts.
        if self.state_path.exists():
            self._load_state()
        else:
            self._save_state()
        try:
            while self.budget_used < self.budget:
                self._advance()
                best = self.tree.best().fitness if self.tree.nodes else None
                print(f'{self.METHOD}: budget={self.budget_used}/{self.budget} '
                      f'nodes={len(self.tree.nodes)} best={best}', flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError(f'evaluation budget exhausted before {self.n_roots} requested valid roots')
            self._write_summary('finished')
        except UnknownEvaluation:
            self._write_summary('blocked', error=traceback.format_exc())
            raise
        except KeyboardInterrupt:
            self._write_summary('interrupted')
            raise
        except Exception:
            self._write_summary('error', error=traceback.format_exc())
            raise

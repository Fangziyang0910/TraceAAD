"""V10.7 organizes archived evidence by the requested design task."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback
import time

from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103
from llm4ad.method.traceaad_v10_5.traceaad import (
    TraceAADV105, UnknownEvaluation, atomic_json, ess,
)
from llm4ad.method.traceaad_v10_6 import prompts as v106_prompts
from llm4ad.method.traceaad_v10_6 import traceaad as v106_traceaad
from llm4ad.method.traceaad_v10_6.traceaad import (
    OPERATOR_PROBABILITIES, TraceAADV106, joint_parent_distribution,
)
from . import prompts, sampling


class TraceAADV107(TraceAADV106):
    METHOD = 'v107r'

    def __init__(self, *, history_tokens=8192, task_name=None,
                 max_context_programs=2, **kwargs):
        # Only 1 or 2: the sampler serves at most one reference, so 3 has no
        # distinct behavior and must not become a phantom configuration axis.
        if not isinstance(max_context_programs, int) or not 1 <= max_context_programs <= 2:
            raise ValueError('max_context_programs must be 1 or 2')
        self.max_context_programs = max_context_programs
        TraceAADV105.__init__(self, history_tokens=history_tokens, **kwargs)
        self.task_contract = prompts.build_task_contract(self.evaluation)
        self.builder = prompts.TrajectoryBuilder(
            self.llm, self.task_contract, max_tokens=self.builder.max_tokens,
            history_tokens=history_tokens, max_events=self.traj_gens,
            log_count=lambda record: self._append_record(
                self.run_dir / 'tokenizer_calls.jsonl', record
            ),
        )
        self.mechanism.update(
            task_name=task_name, generation=prompts.GENERATION,
            context_policy=sampling.CONTEXT_POLICY,
            max_context_programs=max_context_programs,
            reference_fit_attempts=sampling.MAX_FIT_ATTEMPTS,
            task_contract_hash=hashlib.sha256(self.task_contract.encode()).hexdigest(),
        )
        # donor_topk/traj_gens/history_tokens only satisfy the inherited
        # constructor and feed history rendering the V10.7R path never calls
        # (TrajectoryBuilder overrides trajectory/fits without reading
        # history_tokens). They are not mechanism parameters: relabel them so
        # future analysis cannot mistake them for live controls.
        self.mechanism['inherited_unused'] = {
            key: self.mechanism.pop(key)
            for key in ('donor_topk', 'traj_gens', 'history_tokens')
            if key in self.mechanism
        }
        for source in [
            Path(__file__), Path(prompts.__file__), Path(sampling.__file__),
            Path(v106_traceaad.__file__), Path(v106_prompts.__file__),
        ]:
            self.mechanism['source_hashes'][str(source.resolve())] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()

    def _log_call(self, record):
        TraceAADV105._log_call(self, record)

    @staticmethod
    def _roles(parent, operator, donor=None, extra=None):
        roles = {parent.id: ('comparison_baseline' if operator == 'Pivot' else 'design_base')}
        roles.update({int(node_id): role for node_id, role in (extra or {}).items()})
        if donor is not None:
            roles[donor.id] = 'transfer_source'
        return roles

    def _schedule(self):
        scheduling_started = time.monotonic()
        counts_before = len(self.builder._counts)
        parent = donor = None
        selection = {}
        requested = operator = 'Init'
        if len(self.tree.roots) >= self.n_roots:
            nodes = self.eligible_nodes()
            p0, _, selection = self.parent_distribution(nodes, 'Refine')
            marginal, conditional = joint_parent_distribution(p0)
            index = self.rng.choices(range(len(nodes)), weights=marginal)[0]
            requested = self.rng.choices(
                list(conditional[index]), weights=list(conditional[index].values())
            )[0]
            operator = requested
            selection.update(
                parent_marginal_ess=ess(marginal),
                operator_conditional=conditional[index],
            )
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(
                parent_route='joint_marginal', parent_probability=marginal[index],
                parent_count_before=count,
            )
        references, context = [], {}
        roles = {}
        sampling_started = time.monotonic()
        if parent is not None:
            def fits(refs, proposed_donor, extra_roles, relations):
                prompt_roles = self._roles(
                    parent, requested, proposed_donor, extra_roles,
                )
                return self.builder.fits_references(
                    parent, refs, requested, proposed_donor, prompt_roles, relations,
                )
            references, donor, context = sampling.sample_task_evidence(
                self.tree.all_nodes(), parent, self.rng, operator=requested,
                limit=self.max_context_programs - 1, fits=fits,
            )
            roles = self._roles(
                parent, requested, donor, context.get('reference_roles'),
            )
        prompt_text, programs, donor, operator, blocks, view_omissions = self.builder.trajectory(
            parent, references, requested, donor, roles,
            context.get('evidence_relations'),
        )
        prompt_tokens = self.builder.count(prompt_text, chat=True)
        if prompt_tokens > self.builder.max_tokens:
            raise ValueError('minimum complete prompt exceeds the model context budget')
        if requested == 'Fuse' and operator == 'Refine':
            selection['fallback_reason'] = 'no fitting distinct-code reference'
        context.update(
            sampling_seconds=time.monotonic() - sampling_started,
            context_node_ids=[node.id for node in programs],
            context_program_tokens=[self.builder.count(block) for block in blocks],
            context_program_roles=[roles[node.id] for node in programs],
            context_view_omissions=view_omissions,
        )
        template_hash = prompts.TRAJECTORY_TEMPLATE_HASH
        context.update(
            context_best_fitness=max(
                (node.fitness for node in programs), default=None,
            ),
        )
        context.update(scheduling_seconds=time.monotonic() - scheduling_started,
                       tokenizer_requests=len(self.builder._counts) - counts_before)
        prompt_hash = hashlib.sha256(prompt_text.encode()).hexdigest()
        return {
            'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
            'requested_operator': requested, 'operator': operator,
            'parent_id': parent.id if parent else None,
            'donor_id': donor.id if donor else None,
            'operator_probabilities': OPERATOR_PROBABILITIES, 'selection': selection,
            'parent_fitness': parent.fitness if parent else None,
            'donor_fitness': donor.fitness if donor else None,
            'best_before': self.tree.best().fitness if self.tree.nodes else None,
            'prompt': prompt_text, 'prompt_tokens': prompt_tokens,
            'prompt_hash': prompt_hash,
            'template_hash': template_hash,
            'context_policy': sampling.CONTEXT_POLICY,
            **context,
            'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
        }

    def _duplicate_inputs(self, code):
        # V10.7 keeps its original execution policy; V10.8 rejects input copies.
        return []

    def _advance(self) -> None:
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        pending = self.pending
        if pending['phase'] == 'selected':
            self._generate_pending()
        self._log_call(pending['completion'])
        response = pending['completion']
        parsed = self.parse_response(response['response'], response['finish_reason'])
        duplicates = self._duplicate_inputs(parsed[1]) if parsed is not None else []
        node = None
        reason = None
        if parsed is None:
            reason = ('length_truncated' if response['finish_reason'] == 'length'
                      else 'invalid_code_or_signature')
            self._invalid_streak += 1
            status = 'invalid_output'
        elif duplicates:
            self._invalid_streak += 1
            status = 'duplicate_code'
            reason = 'identical_to_parent_or_donor'
        else:
            self._invalid_streak = 0
            if pending['phase'] != 'evaluated':
                self._evaluate_pending(parsed)
            outcome = pending['outcome']
            if outcome['evaluation_id'] != self.budget_used + 1:
                raise ValueError('evaluation receipt is not the next budget slot')
            self.budget_used = outcome['evaluation_id']
            reason = outcome['reason']
            status = 'eval_failed' if outcome['fitness'] is None else 'ok'
            if outcome['fitness'] is not None:
                node = self.tree.add(
                    code=parsed[1], idea=parsed[0], fitness=outcome['fitness'],
                    evaluation_id=self.budget_used, parent_id=pending['parent_id'],
                    operator=pending['operator'], donor_id=pending['donor_id'],
                )
        if pending['parent_id'] is not None:
            self.step_counter += 1
        record = {
            key: value for key, value in pending.items()
            if key not in ['prompt', 'rng_state', 'completion', 'phase', 'outcome']
        }
        record.update(
            ts=datetime.now().isoformat(timespec='seconds'), step=self.step_counter,
            status=status, reason=reason, budget_used=self.budget_used,
            evaluation_id=pending.get('outcome', {}).get('evaluation_id'),
            eval_seconds=pending.get('outcome', {}).get('eval_seconds'),
            llm_seconds=response['seconds'], node_id=node.id if node else None,
            fitness=node.fitness if node else None,
        )
        if duplicates:
            record['duplicate_matches'] = duplicates
        if parsed is not None:
            record['code_hash'] = hashlib.sha256(parsed[1].encode()).hexdigest()
        if node is not None and pending['parent_id'] is not None:
            record.update(
                parent_improved=node.fitness > pending['parent_fitness'],
                frontier_improved=node.fitness > pending['best_before'],
                context_improved=node.fitness > pending['context_best_fitness'],
                parent_delta=node.fitness - pending['parent_fitness'],
                frontier_delta=node.fitness - pending['best_before'],
                context_delta=node.fitness - pending['context_best_fitness'],
            )
            if pending['donor_id'] is not None:
                baseline = max(pending['parent_fitness'], pending['donor_fitness'])
                record['both_improved'] = node.fitness > baseline
                record['both_delta'] = node.fitness - baseline
        if pending['candidate_id'] not in self._logged_events:
            self._append_record(self.events_path, record)
            self._logged_events.add(pending['candidate_id'])
        self.completed_attempts = pending['candidate_id']
        self._save_state()
        self.pending_path.unlink(missing_ok=True)
        self.pending = None
        if self._invalid_streak >= 50:
            raise RuntimeError('50 consecutive generations produced no evaluable output')

    def _save_state(self) -> None:
        atomic_json(self.state_path, {
            'version': 1073, 'mechanism': self.mechanism, 'started_at': self.started_at,
            'nodes': self.tree.to_state(), 'rng_state': list(self.rng.getstate()),
            'parent_selection_counts': self.parent_selection_counts,
            'step_counter': self.step_counter, 'batch_counter': self.step_counter,
            'budget_used': self.budget_used, 'completed_attempts': self.completed_attempts,
            'invalid_streak': self._invalid_streak,
        })

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        if state.get('version') != 1073 or state.get('mechanism') != self.mechanism:
            raise ValueError('checkpoint mechanism/source/backend differs from this V10.7R configuration')
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
                    pending['selection']['parent_count_before'] + 1
                )

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.state_path.exists():
                self._load_state()
            else:
                self._save_state()
            while self.budget_used < self.budget:
                self._advance()
                best = self.tree.best().fitness if self.tree.nodes else None
                print(
                    f'v107r: budget={self.budget_used}/{self.budget} '
                    f'nodes={len(self.tree.nodes)} best={best}',
                    flush=True,
                )
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError(
                    f'evaluation budget exhausted before {self.n_roots} requested valid roots'
                )
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

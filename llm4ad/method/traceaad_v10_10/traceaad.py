"""V10.10: quality-based search with formation history and one error-conditioned repair."""

import ast
import math
import platform
import time
import traceback
from functools import lru_cache
from pathlib import Path

import numpy
from llm4ad.base import TextFunctionProgramConverter

from llm4ad.method.traceaad_v10_3.traceaad import calibrate_beta
from llm4ad.method.traceaad_v10_8.traceaad import TraceAADV108
from llm4ad.method.traceaad_v10_5.traceaad import ess, read_journal, UnknownEvaluation
from . import trajectory
from .trajectory import digest
from . import errors

OPERATOR_PROBABILITIES = {'Refine': 0.25, 'Tune': 0.25, 'Pivot': 0.25, 'Fuse': 0.25}
REPAIRABLE_FAILURES = {'exec_error', 'runtime_error', 'timeout',
                       'invalid_result', 'nonfinite_fitness'}
QUALITY_ESS_TARGET = 8.0
PIVOT_UNIFORM_MIX = 0.5
DONOR_UNIFORM_MIX = 0.5
SELECTION_POLICY = 'ess8_quality_pivot_uniform_v2'
DEDUP_POLICY = 'parent_donor_ast_preserve_docstrings_v1'
ERROR_HANDLING = 'candidate_error_one_repair_v3'


@lru_cache(maxsize=8192)
def code_key(code):
    """Exact syntax signature, preserving docstrings and numeric constants."""
    return ast.dump(ast.parse(code), include_attributes=False)


class TraceAADV1010(TraceAADV108):
    METHOD = 'v1010'
    DISPLAY_NAME = 'V10.10'
    STATE_VERSION = 10100
    OPERATOR_PROBABILITIES = OPERATOR_PROBABILITIES
    TEMPLATE_HASH = trajectory.TEMPLATE_HASH
    CONTEXT_POLICY = trajectory.CONTEXT_POLICY

    def __init__(self, **kwargs):
        # allocation_arm only satisfies the inherited V10.8 constructor.
        # V10.10 defines its fixed allocation policy locally below.
        super().__init__(allocation_arm='C', **kwargs)
        target = TextFunctionProgramConverter.text_to_function(
            self.evaluation.template_program)
        if target is None:
            raise ValueError('evaluation template must define one target function')
        self.task_contract = (
            '# Task Contract\n\n' + self.evaluation.task_description.strip() +
            '\n\nThe evaluator uses a fixed program template and calls the target '
            'function below. Design this function; the system supplies the template '
            'imports and evaluation scaffold.\n\nTarget function:\n```python\n' +
            str(target).strip() + '\n```\n\nKeep the function name, arguments, '
            'and return contract unchanged.\n'
        )
        notes = getattr(self.evaluation, 'design_notes', '').strip()
        if notes:
            self.task_contract += f'\n\n# Evaluator Semantics\n{notes}'
        # The evaluator environment decides which APIs and how much time a
        # candidate really has; record it and state it as a short fact.
        self.mechanism['runtime'] = {
            'python': platform.python_version(), 'numpy': numpy.__version__,
        }
        runtime = (f"Python {self.mechanism['runtime']['python']}, "
                   f"NumPy {self.mechanism['runtime']['numpy']}.")
        if self.evaluation.timeout_seconds is not None:
            runtime += (' The complete evaluation of one candidate must finish '
                        f'within {self.evaluation.timeout_seconds} seconds.')
        self.task_contract += f'\n\n# Evaluation Runtime\n{runtime}'
        self._parse_interface = errors.expected_interface(
            self._template_func.name, self._template_func.args)
        self._template_program = self.evaluation.template_program
        self.builder = trajectory.TrajectoryBuilder(
            self.llm, self.task_contract,
            max_tokens=self.max_context_tokens - self.mechanism['context_margin'] - 1,
            history_tokens=self.mechanism['history_tokens'], max_events=self.traj_gens,
            lookup=self.tree.nodes.get, all_nodes=self.tree.all_nodes,
            log_count=lambda r: self._append_record(self.run_dir / 'tokenizer_calls.jsonl', r),
        )
        self.mechanism.update(
            initialization_policy=trajectory.INITIALIZATION_POLICY,
            operator_probabilities=self.OPERATOR_PROBABILITIES,
            context_policy=self.CONTEXT_POLICY, selection_policy=SELECTION_POLICY,
            dedup_policy=DEDUP_POLICY,
            quality_ess_target=QUALITY_ESS_TARGET,
            pivot_uniform_probability=PIVOT_UNIFORM_MIX,
            donor_uniform_probability=DONOR_UNIFORM_MIX,
            task_contract_hash=digest(self.task_contract),
        )
        for source in (Path(__file__), Path(trajectory.__file__)):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())
        self.mechanism.update(generation=trajectory.GENERATION,
                              parse_policy=errors.PARSE_POLICY,
                              error_handling=ERROR_HANDLING, max_repairs=1)
        for source in (Path(errors.__file__),):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())
        # The fixed policy above replaces the inherited V10.8 ablation axes;
        # unused parameters must not participate in the checkpoint identity.
        for key in ('allocation_arm', 'allocation_policy', 'count_exponent',
                    'quality_ess_schedule', 'ess_fraction', 'ess_minimum',
                    'reference_fit_attempts'):
            self.mechanism.pop(key, None)
        self.mechanism['inherited_unused']['history_tokens'] = self.mechanism.pop('history_tokens')
        events = read_journal(self.events_path)
        self._last_event = events[-1] if events else None
        self._last_response = None  # (candidate_id, response) of the newest durable reply

    def _generate_pending(self):
        # Reserve only the output space still available for this complete input.
        configured = self.output_tokens
        available = (self.max_context_tokens - self.mechanism['context_margin']
                     - self.pending['prompt_tokens'])
        if available < 1:
            raise ValueError('complete input leaves no room for model output')
        self.output_tokens = min(configured, available)
        try:
            super()._generate_pending()
        finally:
            self.output_tokens = configured

    def parse_response(self, response, finish_reason='unknown'):
        parsed, source, error = errors.parse_candidate(
            response, finish_reason, self._parse_interface,
            self._template_program)
        self._parse_diagnostics = (source, error)
        return parsed

    def _schedule(self):
        previous = self._last_event
        if (previous and previous['candidate_id'] == self.completed_attempts and
                previous['status'] == 'eval_failed' and
                previous['reason'] not in REPAIRABLE_FAILURES):
            raise RuntimeError(
                f"evaluation infrastructure failed: {previous['reason']}: "
                f"{previous.get('error') or 'unknown error'}"
            )
        if (previous and previous['candidate_id'] == self.completed_attempts and
                (previous['status'] == 'invalid_output' or
                 previous['reason'] in REPAIRABLE_FAILURES) and
                not previous.get('repair_of')):
            # Completed failures are durable events. A repair gets its own candidate and
            # receipt, so the existing crash recovery and actual-call budget apply unchanged.
            if self._last_response and self._last_response[0] == previous['candidate_id']:
                response = self._last_response[1]
            else:  # crash recovery: the cache is empty, the journal is the truth
                records = read_journal(self.llm_calls_path)
                response = next(record['response'] for record in reversed(records)
                                if record.get('candidate_id') == previous['candidate_id']
                                and 'response' in record)
            text = errors.repair_prompt(self.task_contract, response, previous)
            tokens = self.builder.count(text, chat=True)
            return {
                'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
                'repair_of': previous['candidate_id'], 'generation_kind': 'repair',
                **{k: previous[k] for k in ('operator', 'requested_operator', 'parent_id',
                    'donor_id', 'parent_fitness', 'donor_fitness', 'selection', 'operator_probabilities')},
                'best_before': self.tree.best().fitness if self.tree.nodes else None,
                'prompt': text, 'prompt_tokens': tokens, 'prompt_hash': digest(text),
                'template_hash': self.TEMPLATE_HASH, 'context_policy': 'failed_output_and_error_v1',
                'context_best_fitness': previous['parent_fitness'],
                'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
            }
        return super()._schedule()

    def _log_call(self, record):
        record['stage'] = 'repair' if self.pending and self.pending.get('repair_of') else 'generation'
        if self.pending and self.pending.get('repair_of'):
            record['repair_of'] = self.pending['repair_of']
        super()._log_call(record)
        # Durable only once the call has actually reached the journal.
        if 'response' in record:
            self._last_response = (record['candidate_id'], record['response'])

    def _evaluate_pending(self, parsed):
        p = self.pending
        if p['phase'] == 'evaluating':
            outcome = self._outcomes.get(p['candidate_id'])
            if outcome is None:
                raise UnknownEvaluation(f"candidate {p['candidate_id']} has an unknown evaluation reservation; refusing to repeat it")
        else:
            p.update(phase='evaluating', evaluation_id=self.budget_used + 1)
            self._persist_pending()
            started = time.time()
            fitness, reason, error_type, error, trace = None, None, None, None, None
            try:
                result = self.secure.evaluate_program_with_details(parsed[2])
                reason, error_type, error, trace = result.failure_kind, result.error_type, result.error, result.traceback
                if result.result is not None:
                    try:
                        value = float(result.result)
                    except (TypeError, ValueError, OverflowError) as exc:
                        reason = 'invalid_result'
                        error_type = type(exc).__name__
                        error = f'Evaluator returned a non-scalar fitness: {exc}'
                    else:
                        if math.isfinite(value):
                            fitness = value
                        else:
                            reason = 'nonfinite_fitness'
                            error_type = 'NonfiniteFitness'
                            error = 'Evaluator returned a nonfinite fitness.'
            except Exception as exc:
                reason, error_type, error, trace = 'evaluation_error', type(exc).__name__, str(exc), traceback.format_exc()
            outcome = dict(candidate_id=p['candidate_id'], evaluation_id=p['evaluation_id'],
                           fitness=fitness, reason=reason, error_type=error_type, error=error,
                           traceback=trace, eval_seconds=time.time() - started,
                           repair_of=p.get('repair_of'))
            self._append_record(self.evaluations_path, outcome)
            self._outcomes[p['candidate_id']] = outcome
        p.update(phase='evaluated', outcome=outcome)
        self._persist_pending()

    def eligible_nodes(self):
        return self.tree.all_nodes()

    def _quality_distribution(self, nodes):
        scores = [node.fitness for node in nodes]
        beta, attainable_target, quality_ess = calibrate_beta(scores, 0.0, QUALITY_ESS_TARGET)
        best = max(scores)
        weights = [math.exp(beta * (node.fitness - best)) for node in nodes]
        total = sum(weights)
        q = [w / total for w in weights]
        return q, {
            'eligible_nodes': len(nodes),
            'beta': beta,
            'ess_target': min(len(nodes), QUALITY_ESS_TARGET),
            'attainable_ess_target': attainable_target,
            'quality_ess': quality_ess,
        }

    def node_distribution(self, nodes, operator):
        q, stats = self._quality_distribution(nodes)
        if operator == 'Pivot':
            n = len(nodes)
            probabilities = [(1 - PIVOT_UNIFORM_MIX) * qi + PIVOT_UNIFORM_MIX / n
                             for qi in q]
        else:
            probabilities = q
        stats['parent_ess'] = ess(probabilities)
        return probabilities, stats

    def select_donor(self, parent):
        # Quality-biased sampling with uniform archive coverage.
        parent_key = code_key(parent.code)
        nodes = [n for n in self.tree.all_nodes()
                 if n.id != parent.id and code_key(n.code) != parent_key]
        if not nodes:
            return None, []
        q, _ = self._quality_distribution(nodes)
        n = len(nodes)
        p = [(1 - DONOR_UNIFORM_MIX) * qi + DONOR_UNIFORM_MIX / n for qi in q]
        donor = self.rng.choices(nodes, weights=p)[0]
        return donor, [{'node_id': donor.id, 'fitness': donor.fitness}]

    def _duplicate_inputs(self, code):
        # Search-stage input-copy filter; initialization accepts any valid program.
        return [{'role': role, 'node_id': self.pending[role + '_id'], 'view': 'ast'}
                for role in ('parent', 'donor')
                if self.pending[role + '_id'] is not None and
                code_key(code) == code_key(self.tree.nodes[self.pending[role + '_id']].code)]

    def _append_record(self, path, record):
        if path == self.events_path:
            record['parse_mode'], parse_error = self._parse_diagnostics
            if record['status'] == 'invalid_output':
                record.update(error_type='ParseError', error=parse_error)
            elif record['status'] == 'eval_failed':
                record.update({k: self.pending['outcome'].get(k) for k in ('error_type', 'error', 'traceback')})
        super()._append_record(path, record)
        if path == self.events_path:
            self._last_event = record

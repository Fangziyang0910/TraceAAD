"""V10.6 reuses V10.5 evaluation/journals; owns its generation protocol and state version."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import time
import traceback
from datetime import datetime

from llm4ad.method.traceaad_v10_3.schema import normalize_code
from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103, _strip_thinking
from llm4ad.method.traceaad_v10_5.traceaad import TraceAADV105, OPERATOR_PROBABILITIES, atomic_json, ess, UnknownEvaluation
from . import prompts

CODE_RE = re.compile(r"\A\s*(?:Design )?Idea:\s*(\S.*?)\s*```(?:python)?[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*\s*\Z", re.DOTALL | re.MULTILINE)
SUMMARY_RE = re.compile(r"\A(?:Implementation )?Idea:\s*(\S.*?)\s*\Z", re.DOTALL)


def joint_parent_distribution(p0):
    marginal = [.925*p + .075/len(p0) for p in p0]
    conditional = [dict(zip(OPERATOR_PROBABILITIES, [.50*p/m, (.075*p+.075/len(p0))/m, .35*p/m]))
                   for p, m in zip(p0, marginal)]
    return marginal, conditional


class TraceAADV106(TraceAADV105):
    METHOD = 'v106'

    def __init__(self, *, summary_tokens=1024, history_tokens=8192, task_name=None, **kwargs):
        super().__init__(history_tokens=history_tokens, **kwargs)
        self.task_contract = prompts.build_task_contract(self.evaluation)
        self.builder = prompts.PromptBuilder(self.llm, self.task_contract,
            max_tokens=self.builder.max_tokens, history_tokens=history_tokens, max_events=self.traj_gens,
            summary_tokens=summary_tokens, lookup=self.tree.nodes.get,
            log_count=lambda record: self._append_record(self.run_dir / 'tokenizer_calls.jsonl', record))
        self.mechanism.update(summary_tokens=summary_tokens, task_name=task_name,
            generation=prompts.GENERATION,
            task_contract_hash=hashlib.sha256(self.task_contract.encode()).hexdigest())
        for p in [Path(__file__), Path(prompts.__file__)]:
            self.mechanism['source_hashes'][str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()

    def parse_response(self, response, finish_reason='unknown'):
        if finish_reason not in ['stop', 'length', 'unknown']:
            return None
        match = CODE_RE.fullmatch(_strip_thinking(response))
        if match is None:
            return None
        idea, code = match.groups()
        if re.search(r'^\s*```', code, re.MULTILINE):
            return None
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == self._template_func.name]
        if len(targets) != 1:
            return None
        expected = ast.parse(f"def target({self._template_func.args}):\n    pass").body[0].args
        def interface(args):
            return ([a.arg for a in args.posonlyargs], [a.arg for a in args.args],
                    [a.arg for a in args.kwonlyargs],
                    args.vararg.arg if args.vararg else None,
                    args.kwarg.arg if args.kwarg else None)
        if interface(targets[0].args) != interface(expected):
            return None
        # Evaluate exactly the archived module. Reconstructing a Program silently
        # loses decorators and module statements after the first function.
        canonical = normalize_code(code)
        try:
            compile(canonical, '<candidate>', 'exec')
        except (SyntaxError, ValueError):
            return None
        return idea.strip(), canonical, canonical

    def _log_call(self, record):
        record.setdefault('stage', 'generation')
        super()._log_call(record)

    def _calibrate_pending(self, parsed):
        p = self.pending
        if 'summary_completion' in p:
            self._log_call(p['summary_completion'])
        if 'summary_status' in p:
            return
        p['design_idea'] = parsed[0]
        if 'summary_completion' not in p:
            parent = self.tree.nodes.get(p['parent_id'])
            prompt = prompts.build_summary_prompt(self.task_contract, parsed[0], parsed[1], parent)
            tokens = self.builder.count(prompt, chat=True)
            p.update(summary_prompt=prompt, summary_prompt_tokens=tokens)
            # Keep complete code and parent context; an oversized description
            # request leaves the valid candidate available for evaluation.
            if tokens > self.builder.max_tokens:
                p.update(implementation_idea='', summary_status='context_exceeded', summary_tokens=0)
                self._persist_pending()
                return
            p['summary_attempts'] = p.get('summary_attempts', 0) + 1
            self._persist_pending()
            record = dict(ts=datetime.now().isoformat(timespec='seconds'),
                call_id=f"{p['candidate_id']}:summary:{p['summary_attempts']}",
                candidate_id=p['candidate_id'], stage='thought_alignment',
                operator=p['operator'], requested_operator=p['requested_operator'],
                prompt=prompt, prompt_tokens=tokens,
                prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
                template_hash=prompts.TEMPLATE_HASH, sampling=self.mechanism['llm'],
                max_tokens=self.output_tokens)
            started = time.time()
            try:
                details = self.llm.draw_sample_with_details(prompt, max_tokens=self.output_tokens)
                record.update(response=details['content'], finish_reason=details.get('finish_reason') or 'unknown',
                    usage=details.get('usage'), model=details.get('model'), response_id=details.get('response_id'))
            except Exception:
                record['error'] = traceback.format_exc()
            record['seconds'] = time.time() - started
            p['summary_completion'] = record
            self._persist_pending()
            self._log_call(record)
        record = p['summary_completion']
        raw = record.get('response')
        text = _strip_thinking(raw).strip() if isinstance(raw, str) else ''
        match = SUMMARY_RE.fullmatch(text)
        finish = record.get('finish_reason')
        idea = (match.group(1).strip() if match and finish in ['stop', 'unknown']
                and not re.search(r'^\s*```', text, re.MULTILINE) else '')
        p.update(implementation_idea=idea,
                 summary_status='present' if idea else 'truncated' if finish == 'length' else 'unavailable',
                 summary_tokens=self.builder.count(idea) if idea else 0)
        self._persist_pending()

    def _schedule(self) -> dict:
        parent = donor = None
        selection: dict = {}
        requested = operator = "Init"
        if len(self.tree.roots) >= self.n_roots:
            nodes = self.eligible_nodes()
            p0, _, selection = self.parent_distribution(nodes, 'Refine')
            marginal, conditional = joint_parent_distribution(p0)
            index = self.rng.choices(range(len(nodes)), weights=marginal)[0]
            requested = self.rng.choices(list(conditional[index]), weights=list(conditional[index].values()))[0]
            operator = requested
            selection.update(parent_marginal_ess=ess(marginal), operator_conditional=conditional[index])
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(parent_route="joint_marginal",
                             parent_probability=marginal[index], parent_count_before=count)
            if requested == "Fuse":
                donors = self.fitting_donors(parent)
                if donors:
                    donor = self.rng.choice(donors)
                else:
                    operator = "Refine"
                    selection['fallback_reason'] = "no fitting cross-lineage donor"
        ancestors = self.tree.ancestors(parent.id) if parent else []
        prompt = self.builder.build(parent, ancestors, operator, donor)
        return {
            "candidate_id": self.completed_attempts + 1, "phase": "selected",
            "requested_operator": requested, "operator": operator,
            "parent_id": parent.id if parent else None, "donor_id": donor.id if donor else None,
            "operator_probabilities": OPERATOR_PROBABILITIES, "selection": selection,
            "parent_fitness": parent.fitness if parent else None,
            "donor_fitness": donor.fitness if donor else None,
            "best_before": self.tree.best().fitness if self.tree.nodes else None,
            "prompt": prompt.text, "prompt_tokens": prompt.tokens,
            "prompt_hash": hashlib.sha256(prompt.text.encode()).hexdigest(),
            "template_hash": prompts.TEMPLATE_HASH, "history_ids": prompt.history_ids,
            "context_omissions": prompt.omissions, "history_tokens": prompt.history_tokens,
            "context_summaries": prompt.summaries, "rng_state": list(self.rng.getstate()),
            "llm_attempts": 0,
        }


    def _advance(self) -> None:
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        p = self.pending
        if p['phase'] == 'selected':
            self._generate_pending()
        self._log_call(p['completion'])
        response = p['completion']
        parsed = self.parse_response(response['response'], response['finish_reason'])
        if parsed is not None:
            self._calibrate_pending(parsed)
            parsed = (p['implementation_idea'], parsed[1], parsed[2])
        node = None
        reason = None
        if parsed is None:
            reason = 'length_truncated' if response['finish_reason'] == 'length' else 'invalid_code_or_signature'
            self._invalid_streak += 1
            status = "invalid_output"
        else:
            self._invalid_streak = 0
            if p['phase'] != 'evaluated':
                self._evaluate_pending(parsed)
            outcome = p['outcome']
            if outcome['evaluation_id'] != self.budget_used + 1:
                raise ValueError("evaluation receipt is not the next budget slot")
            self.budget_used = outcome['evaluation_id']
            reason = outcome['reason']
            status = 'eval_failed' if outcome['fitness'] is None else 'ok'
            if outcome['fitness'] is not None:
                node = self.tree.add(code=parsed[1], idea=parsed[0], fitness=outcome['fitness'],
                                     evaluation_id=self.budget_used, parent_id=p['parent_id'],
                                     operator=p['operator'], donor_id=p['donor_id'])
        if p['parent_id'] is not None:
            self.step_counter += 1
        record = {k: v for k, v in p.items() if k not in [
            'prompt', 'rng_state', 'completion', 'phase', 'outcome',
            'summary_prompt', 'summary_completion', 'implementation_idea']}
        record.update(ts=datetime.now().isoformat(timespec="seconds"), step=self.step_counter,
                      status=status, reason=reason, budget_used=self.budget_used,
                      evaluation_id=p.get('outcome', {}).get('evaluation_id'),
                      eval_seconds=p.get('outcome', {}).get('eval_seconds'),
                      generation_llm_seconds=response['seconds'],
                      summary_llm_seconds=p.get('summary_completion', {}).get('seconds', 0),
                      llm_seconds=response['seconds'] + p.get('summary_completion', {}).get('seconds', 0),
                      node_id=node.id if node else None,
                      fitness=node.fitness if node else None)
        if node is not None and p['parent_id'] is not None:
            record.update(parent_improved=node.fitness > p['parent_fitness'],
                          frontier_improved=node.fitness > p['best_before'],
                          parent_delta=node.fitness - p['parent_fitness'],
                          frontier_delta=node.fitness - p['best_before'])
            if p['donor_id'] is not None:
                record['both_improved'] = node.fitness > max(p['parent_fitness'], p['donor_fitness'])
                record['both_delta'] = node.fitness - max(p['parent_fitness'], p['donor_fitness'])
        if p['candidate_id'] not in self._logged_events:
            self._append_record(self.events_path, record)
            self._logged_events.add(p['candidate_id'])
        self.completed_attempts = p['candidate_id']
        self._save_state()
        self.pending_path.unlink(missing_ok=True)
        self.pending = None
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")


    def _save_state(self) -> None:
        atomic_json(self.state_path, {
            "version": 106, "mechanism": self.mechanism, "started_at": self.started_at,
            "nodes": self.tree.to_state(), "rng_state": list(self.rng.getstate()),
            "parent_selection_counts": self.parent_selection_counts,
            "step_counter": self.step_counter, "batch_counter": self.step_counter,
            "budget_used": self.budget_used, "completed_attempts": self.completed_attempts,
            "invalid_streak": self._invalid_streak,
        })


    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        if state.get('version') != 106 or state.get('mechanism') != self.mechanism:
            raise ValueError("checkpoint mechanism/source/backend differs from this V10.6 configuration")
        TraceAADV103._load_state(self)
        self.completed_attempts = state['completed_attempts']
        self._invalid_streak = state['invalid_streak']
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if pending['candidate_id'] <= self.completed_attempts:
                self.pending_path.unlink()
                return
            if pending['candidate_id'] != self.completed_attempts + 1:
                raise ValueError("pending candidate is not the next attempt")
            self.pending = pending
            rng = pending['rng_state']
            self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
            if pending['parent_id'] is not None:
                self.parent_selection_counts[pending['parent_id']] = pending['selection']['parent_count_before'] + 1


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
                print(f"v106: budget={self.budget_used}/{self.budget} nodes={len(self.tree.nodes)} best={best}", flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError(f"evaluation budget exhausted before {self.n_roots} requested valid roots")
            self._write_summary("finished")
        except UnknownEvaluation:
            self._write_summary("blocked", error=traceback.format_exc())
            raise
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", error=traceback.format_exc())
            raise

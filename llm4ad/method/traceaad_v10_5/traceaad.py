"""TraceAAD V10.5 launch configuration.

Reuse V10.3's task setup, evaluator, tree and summaries. V10.5 owns
the operator-conditioned selection, token-bounded prompts and durable candidate
loop; alternative adaptive/two-stage policies are deliberately absent.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import time
import traceback
from datetime import datetime

from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103, calibrate_beta, _strip_thinking

from . import prompts
from .prompts import PromptBuilder

OPERATOR_PROBABILITIES = {"Refine": 0.50, "Pivot": 0.15, "Fuse": 0.35}
PIVOT_UNIFORM_PROBABILITY = 0.50
RESPONSE_RE = re.compile(
    r"\A\s*(?:Latest Design )?Idea:\s*(\S.*?)\s*```(?:python)?[ \t]*\r?\n(.*?)```\s*\Z",
    re.DOTALL,
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def read_journal(path: Path) -> list[dict]:
    """Recover a partial final write, but never ignore interior corruption."""
    if not path.exists():
        return []
    records = []
    with path.open("rb+") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                records.append(json.loads(line))
                if not line.endswith(b'\n'):
                    handle.write(b'\n')
                    handle.flush()
                    os.fsync(handle.fileno())
            except (json.JSONDecodeError, UnicodeDecodeError):
                if handle.read(1):
                    raise ValueError(f"corrupt journal record in {path} at byte {offset}")
                handle.truncate(offset)
                break
    return records


def ess(probabilities: list[float]) -> float:
    return 1.0 / sum(p * p for p in probabilities)


class UnknownEvaluation(RuntimeError):
    """A submitted evaluation cannot safely be charged or repeated after a crash."""


class TraceAADV105(TraceAADV103):
    METHOD = "v105"

    def __init__(self, *, history_tokens: int = 2048, context_margin: int = 256, **kwargs):
        super().__init__(**kwargs)
        if not 0 < self.n_roots <= self.budget or self.donor_topk < 1 or self.traj_gens < 0:
            raise ValueError("invalid root, donor or trajectory limits")
        if history_tokens < 1 or context_margin < 0 or not 0 < self.ess_fraction <= 1 or self.ess_minimum < 1:
            raise ValueError("invalid context or ESS settings")
        prompt_limit = self.max_context_tokens - self.output_tokens - context_margin
        if self.output_tokens < 1 or prompt_limit < 1:
            raise ValueError("model context must cover output tokens and safety margin")
        self.builder = PromptBuilder(self.llm, self.task_contract, max_tokens=prompt_limit,
                                     history_tokens=history_tokens, max_events=self.traj_gens,
                                     log_count=lambda record: self._append_record(self.run_dir / 'tokenizer_calls.jsonl', record))
        self.pending_path = self.run_dir / "pending_candidate.json"
        self.llm_calls_path = self.run_dir / "llm_calls.jsonl"
        self.evaluations_path = self.run_dir / "evaluations.jsonl"
        self.pending: dict | None = None
        self.completed_attempts = 0
        self._eligible: dict[int, bool] = {}
        self._logged_calls = {r['call_id'] for r in read_journal(self.llm_calls_path)}
        self._logged_events = {r['candidate_id'] for r in read_journal(self.events_path)}
        self._outcomes = {r['candidate_id']: r for r in read_journal(self.evaluations_path)}
        sources = [Path(__file__), Path(prompts.__file__), Path(inspect.getfile(TraceAADV103)),
                   Path(inspect.getfile(Node)), Path(inspect.getfile(prompts.build_task_contract)),
                   Path(inspect.getfile(type(self.secure))), Path(inspect.getfile(type(self._template))),
                   Path(inspect.getfile(type(self.evaluation))), Path(inspect.getfile(type(self.llm)))]
        self.mechanism = {
            "method": self.METHOD, "budget": self.budget, "n_roots": self.n_roots,
            "donor_topk": self.donor_topk, "traj_gens": self.traj_gens,
            "ess_fraction": self.ess_fraction, "ess_minimum": self.ess_minimum,
            "history_tokens": history_tokens, "context_margin": context_margin,
            "output_tokens": self.output_tokens, "max_context_tokens": self.max_context_tokens,
            "operator_probabilities": OPERATOR_PROBABILITIES,
            "pivot_uniform_probability": PIVOT_UNIFORM_PROBABILITY,
            "source_hashes": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "task_contract_hash": hashlib.sha256(self.task_contract.encode()).hexdigest(),
            "llm": {name: getattr(self.llm, name, None) for name in
                    ['model', 'base_url', 'temperature', 'top_p', 'enable_thinking', 'extra_body']},
        }

    def parse_response(self, response: str):
        match = RESPONSE_RE.fullmatch(_strip_thinking(response))
        if match is None:
            return None
        idea, code = match.groups()
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
        canonical = code.strip()
        try:
            compile(canonical, '<candidate>', 'exec')
        except (SyntaxError, ValueError):
            return None
        return idea.strip(), canonical, canonical

    def parent_distribution(self, nodes: list[Node], requested_operator: str):
        fitnesses = [n.fitness for n in nodes]
        beta, target, quality_ess = calibrate_beta(fitnesses, self.ess_fraction, self.ess_minimum)
        maximum = max(fitnesses)
        weights = [math.exp(beta * (q - maximum)) for q in fitnesses]
        corrected = [w / math.sqrt(self.parent_selection_counts.get(n.id, 0) + 1)
                     for w, n in zip(weights, nodes)]
        total = sum(corrected)
        p0 = [w / total for w in corrected]
        probabilities = ([0.5 * p + 0.5 / len(nodes) for p in p0]
                         if requested_operator == "Pivot" else p0)
        return p0, probabilities, {
            "eligible_nodes": len(nodes), "beta": beta, "ess_target": target,
            "quality_ess": quality_ess, "corrected_ess": ess(p0),
            "conditional_ess": ess(probabilities),
        }

    def eligible_nodes(self) -> list[Node]:
        for node in self.tree.all_nodes():
            if node.id not in self._eligible:
                self._eligible[node.id] = all(self.builder.fits(node, op) for op in ['Refine', 'Pivot'])
        nodes = [n for n in self.tree.all_nodes() if self._eligible[n.id]]
        if not nodes:
            raise ValueError("no archived parent fits the complete model context")
        return nodes

    def fitting_donors(self, parent: Node) -> list[Node]:
        excluded = {parent.id} | {n.id for n in self.tree.ancestors(parent.id)} | self.tree.descendants(parent.id)
        candidates = sorted((n for n in self.tree.all_nodes() if n.id not in excluded),
                            key=lambda n: (-n.fitness, n.id))
        donors = []
        for node in candidates:
            if self.builder.fits(parent, "Fuse", node):
                donors.append(node)
                if len(donors) == self.donor_topk:
                    break
        return donors

    def _schedule(self) -> dict:
        parent = donor = None
        selection: dict = {}
        requested = operator = "Init"
        if len(self.tree.roots) >= self.n_roots:
            requested = self.rng.choices(list(OPERATOR_PROBABILITIES), weights=list(OPERATOR_PROBABILITIES.values()))[0]
            operator = requested
            nodes = self.eligible_nodes()
            p0, probabilities, selection = self.parent_distribution(nodes, requested)
            uniform = requested == "Pivot" and self.rng.random() < PIVOT_UNIFORM_PROBABILITY
            index = self.rng.randrange(len(nodes)) if uniform else self.rng.choices(range(len(nodes)), weights=p0)[0]
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(parent_route="uniform" if uniform else "quality",
                             parent_probability=probabilities[index], parent_count_before=count)
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
            "context_omissions": prompt.omissions, "rng_state": list(self.rng.getstate()),
            "llm_attempts": 0,
        }

    def _append_record(self, path: Path, record: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _persist_pending(self) -> None:
        atomic_json(self.pending_path, self.pending)

    def _log_call(self, record: dict) -> None:
        if record['call_id'] not in self._logged_calls:
            self._append_record(self.llm_calls_path, record)
            self._logged_calls.add(record['call_id'])

    def _generate_pending(self) -> None:
        p = self.pending
        p['llm_attempts'] += 1
        self._persist_pending()
        started = time.time()
        record = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "call_id": f"{p['candidate_id']}:{p['llm_attempts']}",
            "candidate_id": p['candidate_id'], "operator": p['operator'],
            "requested_operator": p['requested_operator'], "prompt": p['prompt'],
            "prompt_tokens": p['prompt_tokens'], "prompt_hash": p['prompt_hash'],
            "template_hash": p['template_hash'], "sampling": self.mechanism['llm'],
            "max_tokens": self.output_tokens,
        }
        try:
            details = self.llm.draw_sample_with_details(p['prompt'], max_tokens=self.output_tokens)
        except Exception:
            record.update(seconds=time.time() - started, error=traceback.format_exc())
            self._log_call(record)
            raise
        record.update(response=details['content'], seconds=time.time() - started,
                      finish_reason=details.get('finish_reason') or "unknown",
                      usage=details.get('usage'), model=details.get('model'),
                      response_id=details.get('response_id'))
        p.update(phase="responded", completion=record)
        # The response is durable before parsing/evaluation, including after a crash.
        self._persist_pending()
        self._log_call(record)

    def _evaluate_pending(self, parsed) -> None:
        p = self.pending
        if p['phase'] == 'evaluating':
            outcome = self._outcomes.get(p['candidate_id'])
            if outcome is None:
                raise UnknownEvaluation(f"candidate {p['candidate_id']} has an unknown evaluation reservation; refusing to repeat it")
        else:
            p.update(phase="evaluating", evaluation_id=self.budget_used + 1)
            self._persist_pending()
            started = time.time()
            fitness = None
            reason = None
            try:
                result = self.secure.evaluate_program_with_details(parsed[2])
                reason = result.failure_kind
                if result.result is not None:
                    value = float(result.result)
                    if math.isfinite(value):
                        fitness = value
                    else:
                        reason = "nonfinite_fitness"
            except Exception:
                reason = traceback.format_exc()
            outcome = {
                "candidate_id": p['candidate_id'], "evaluation_id": p['evaluation_id'],
                "fitness": fitness, "reason": reason, "eval_seconds": time.time() - started,
            }
            # An evaluation receipt recovers the gap before the next tree checkpoint.
            self._append_record(self.evaluations_path, outcome)
            self._outcomes[p['candidate_id']] = outcome
        p.update(phase="evaluated", outcome=outcome)
        self._persist_pending()

    def _advance(self) -> None:
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        p = self.pending
        if p['phase'] == 'selected':
            self._generate_pending()
        self._log_call(p['completion'])
        response = p['completion']
        parsed = None if response['finish_reason'] == 'length' else self.parse_response(response['response'])
        node = None
        reason = None
        if parsed is None:
            reason = 'length_truncated' if response['finish_reason'] == 'length' else 'invalid_idea_code_or_signature'
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
        record = {k: v for k, v in p.items() if k not in ['prompt', 'rng_state', 'completion', 'phase', 'outcome']}
        record.update(ts=datetime.now().isoformat(timespec="seconds"), step=self.step_counter,
                      status=status, reason=reason, budget_used=self.budget_used,
                      evaluation_id=p.get('outcome', {}).get('evaluation_id'),
                      eval_seconds=p.get('outcome', {}).get('eval_seconds'),
                      llm_seconds=response['seconds'], node_id=node.id if node else None,
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
            "version": 105, "mechanism": self.mechanism, "started_at": self.started_at,
            "nodes": self.tree.to_state(), "rng_state": list(self.rng.getstate()),
            "parent_selection_counts": self.parent_selection_counts,
            "step_counter": self.step_counter, "batch_counter": self.step_counter,
            "budget_used": self.budget_used, "completed_attempts": self.completed_attempts,
            "invalid_streak": self._invalid_streak,
        })

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        if state.get('version') != 105 or state.get('mechanism') != self.mechanism:
            raise ValueError("checkpoint mechanism/source/backend differs from this V10.5 configuration")
        super()._load_state()
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

    def _write_summary(self, status: str, error: str | None = None) -> None:
        super()._write_summary(status, error)
        payload = json.loads(self.summary_path.read_text())
        unknown = []
        if self.pending and self.pending['phase'] == 'evaluating' and self.pending['candidate_id'] not in self._outcomes:
            unknown = [{key: self.pending[key] for key in ['candidate_id', 'evaluation_id']}]
        payload.update(confirmed_evaluations=len(self._outcomes),
                       unknown_reservation_count=len(unknown), unknown_reservations=unknown)
        atomic_json(self.summary_path, payload)

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
                print(f"v105: budget={self.budget_used}/{self.budget} nodes={len(self.tree.nodes)} best={best}", flush=True)
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

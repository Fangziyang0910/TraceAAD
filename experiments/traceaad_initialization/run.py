"""Initialization comparison under the standard evaluation-call budget."""
import argparse
import json
import re
import traceback
from pathlib import Path
from experiments.infra.runner import add_common_run_args, setup_experiment_run
from traceaad.v10_13.traceaad import TraceAADV1013, Candidate
from traceaad.v10_13.prompts import PromptBuilder

MODES = {'independent': 8, 'sequential': 1, 'hybrid': 4}

class InitializationExperiment(TraceAADV1013):
    def __init__(self, *, init_mode, init_only=False, max_calls=80, **kwargs):
        super().__init__(**kwargs)
        self.init_mode, self.init_only, self.max_calls = init_mode, init_only, max_calls
        self.empty_prompts = PromptBuilder(self.llm, self.task_prompt,
            max_tokens=self.prompts.max_tokens, history_depth=self.prompts.history_depth,
            lookup=self.tree.nodes.get, all_nodes=lambda: [])

    def _schedule(self):
        if len(self.tree.roots) < self.n_roots:
            independent = len(self.tree.roots) < MODES[self.init_mode]
            condition = 'independent' if independent else 'informed'
            builder = self.empty_prompts if independent else self.prompts
            prompt = builder.build_initial(condition=condition).prompt
            context_ids = [int(x) for x in re.findall(
                r'# Previous Initial Algorithm\nNode (\d+) \|', prompt)]
            metadata = {'candidate_id': self.candidate_count + 1,
                'init_mode': self.init_mode, 'root_ordinal': len(self.tree.roots) + 1,
                'condition': condition, 'init_context_ids': context_ids,
                'prompt_chars': len(prompt),
                'prompt_tokens': self.llm.count_prompt_tokens(prompt)}
            with (self.run_dir / 'init_context.jsonl').open('a') as f:
                f.write(json.dumps(metadata) + '\n')
            return Candidate(self.candidate_count + 1, prompt, 'Init', None, None)
        return super()._schedule()

    def run(self):
        if self.storage.state_path.exists():
            self._resume()
        else:
            self._save_state()
        try:
            while self.evaluations_used < self.budget and self.candidate_count < self.max_calls:
                if self.init_only and len(self.tree.roots) >= self.n_roots:
                    break
                self._run_candidate()
                print(
                    f'{self.init_mode}: evals={self.evaluations_used}/{self.budget}, '
                    f'attempts={self.candidate_count}, roots={len(self.tree.roots)}',
                    flush=True,
                )
            budget_exhausted = self.evaluations_used >= self.budget
            call_cap = self.candidate_count >= self.max_calls and not budget_exhausted
            if call_cap:
                status = 'call_cap'
            elif len(self.tree.roots) < self.n_roots:
                status = 'incomplete_initialization'
            else:
                status = 'finished'
            self._write_summary(status)
        except RuntimeError as exc:
            if '50 consecutive generations produced no valid output' in str(exc):
                self._write_summary('generation_stall', str(exc))
                return
            self._write_summary('error', traceback.format_exc())
            raise
        except Exception:
            self._write_summary('error', traceback.format_exc())
            raise

    def _write_summary(self, status, error=None):
        super()._write_summary(status, error)
        summary_path = self.storage.summary_path
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary.update({
            'budget_basis': 'evaluator_calls',
            'candidate_attempts': self.candidate_count,
            'evaluation_calls': self.evaluations_used,
            'initialization_complete': len(self.tree.roots) >= self.n_roots,
        })
        self.storage.save_summary(summary)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(p, default_output_tokens=8192, default_budget=24)
    p.add_argument('--init-mode', choices=MODES, required=True)
    p.add_argument('--init-only', action='store_true')
    args = p.parse_args()
    if args.budget < 8:
        p.error('--budget must allow at least eight evaluation calls')
    max_calls = max(80, args.budget * 3)
    params = dict(budget=args.budget, n_roots=8, history_depth=3,
        max_input_tokens=24320, output_tokens=args.output_tokens,
        init_mode=args.init_mode, init_only=args.init_only, max_calls=max_calls)
    ctx = setup_experiment_run(args, method='initialization_v1013',
        method_dir=Path(__file__).parent, resume_file='tree_state.json',
        method_params=params, budget_basis='evaluator_calls')
    try:
        method = InitializationExperiment(evaluation=ctx.evaluation, llm=ctx.llm,
            run_dir=ctx.run_dir, seed=args.seed, **params)
        ctx.run(method.run)
    finally:
        ctx.llm.close()

if __name__ == '__main__':
    main()

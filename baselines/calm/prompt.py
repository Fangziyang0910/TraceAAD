"""CALM prompts and operator detection.

Relative to upstream whxru/CALM: injection/simplification detectors are fixed to
match the current prompt text (upstream detectors still match obsolete wording).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence

import numpy as np

from .parse import dedent, get_code

if TYPE_CHECKING:
    from .pool import HeuristicRecord


class Prompt:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.base_codes = get_code(prompt)
        if not isinstance(self.base_codes, list):
            self.base_codes = [self.base_codes]
        self.n_calls = 0
        self.last_used_epoch = 0
        self.trials = {}
        self.feasible_algo_generated = False
        self.best_generated_algo_perf = -float('inf')
        self.statuses = []

    @property
    def op(self) -> str:
        if self.is_injection:
            return 'injection'
        if self.is_crossover:
            return 'crossover'
        if self.is_simplification:
            return 'simplify'
        if self.is_creation:
            return 'create'
        basic_mod = 'For the following algorithm, identify '
        if (basic_mod + 'a fixed, instance-independent decision rule') in self.prompt:
            return 'replacement_ins'
        if (basic_mod + 'a key hyper-parameter expressed as either a constant literal or a stationary variable') in self.prompt:
            return 'replacement_hyp'
        if (basic_mod + 'a fragment that assigns equal or near-equal credits to multiple elements') in self.prompt:
            return 'replacement_crd'
        return 'initialization'

    @property
    def is_creation(self) -> bool:
        return 'Be very creative and inventive. Generate an efficient algorithm following the template below' in self.prompt

    @property
    def is_injection(self) -> bool:
        # Fixed to match current prompt_injection text (upstream matched obsolete wording).
        return (
            'Inject one novel, meaningful component into the following algorithm '
            'while preserving its main data flow'
        ) in self.prompt

    @property
    def is_crossover(self) -> bool:
        return (
            'Please generate a new algorithm that is motivated by the following algorithms '
            'but performs better on any same instance'
        ) in self.prompt

    @property
    def is_replacement(self) -> bool:
        return 'For the following algorithm, identify' in self.prompt

    @property
    def is_simplification(self) -> bool:
        # Fixed to match current prompt_simplification text (upstream matched obsolete wording).
        return 'Please create a locally refined version of the following algorithm' in self.prompt

    @property
    def status_str(self) -> str:
        if not self.feasible_algo_generated:
            return 'no feasible child'
        return f'best child performance {self.best_generated_algo_perf:.6f}'

    def record_trial(self, trial: dict) -> None:
        self.n_calls += 1
        n_epoch = trial.pop('n_epoch')
        self.last_used_epoch = max(self.last_used_epoch, n_epoch)
        self.trials.setdefault(n_epoch, []).append(trial)
        if not isinstance(trial['performance'], str):
            self.feasible_algo_generated = True
            self.best_generated_algo_perf = max(self.best_generated_algo_perf, trial['performance'])
        self.statuses.append({
            'n_epoch': n_epoch,
            'status': self.status_str,
            'is_revisit': trial.get('is_revisit', False),
        })

    def age(self, curr_epoch: int) -> int:
        return curr_epoch - self.last_used_epoch

    def __hash__(self):
        return hash(self.prompt)

    def __eq__(self, other):
        if isinstance(other, Prompt):
            return self.prompt == other.prompt
        if isinstance(other, str):
            return self.prompt == other
        return False


class PromptBuilder:
    """Builds CALM operator prompts; strings match reference calm_trainer.py."""

    def __init__(
            self,
            *,
            problem_name: str,
            problem_description: str,
            problem_unit: str,
            algorithm_template: str,
            rs: np.random.RandomState,
            injected_components: List[str],
            train_epoch_getter: Callable[[], int],
    ):
        self.problem_name = problem_name
        self.problem_description = problem_description
        self.problem_unit = problem_unit
        self.algorithm_template = algorithm_template
        self.rs = rs
        self.injected_components = injected_components
        self._train_epoch_getter = train_epoch_getter

    @property
    def system_prompt(self) -> str:
        # Leading indent matches reference calm_trainer.Trainer.system_prompt
        # (indented f-string without dedent).
        return f"""\
            Searching superior heuristics on the {self.problem_name} problem in an evolutionary manner through conversation between User and Assistant. In this problem, {self.problem_description} The User provides existing algorithms and requests a new one.\n\n{self.prompt_algo_requirements()}"""

    @property
    def prompt_creation(self) -> str:
        assert self.algorithm_template is not None, 'No template was provided while prompting for creation'
        return (
            f'Be very creative and inventive. Generate an efficient algorithm following the template below:\n\n'
            f'{self.algorithm_template}{self.prompt_design_strategy()}'
        )

    def prompt_design_strategy(self) -> str:
        strategies = [
            'Keep the strongest working data flow intact and refine one to three expressions that determine the final priorities.',
            'Add one deterministic context modifier derived from the current inputs, then combine it with an existing priority expression.',
            'Prefer small algebraic edits, smooth transformations, or dimensionless normalizers over a complete rewrite.',
            "Preserve the parent algorithm's output shape, vectorization style, and validity conventions while changing its scoring logic.",
            'When two candidates are otherwise similar, introduce a deterministic secondary signal from information already available to the function.',
            'Use at most two new derived quantities, and make each one interact with an existing term instead of replacing the whole rule.',
            'Remove or soften brittle constants by deriving their scale from the current inputs or from aggregate statistics already present.',
        ]
        strategy = strategies[self.rs.choice(len(strategies))]
        return f'\n\n## General Search Strategy\n{strategy}'

    def prompt_local_refinement_guidance(self) -> str:
        return dedent("""\
            ## Local Refinement Guidance
            Preserve the parent implementation's main structure and make a small, coherent edit rather than replacing the algorithm wholesale.
            Change only the minimal code region needed to express the new idea.
            The returned function must use the exact template function name and signature.\
        """)

    def prompt_simplification(self, algos: Sequence[HeuristicRecord]) -> str:
        return (
            'Please create a locally refined version of the following algorithm. '
            'Keep the working structure, remove unnecessary complexity, and improve the expressions '
            f'that produce the final priorities.\n\n{self.prompt_algo_details(algos)}'
            f'{self.prompt_design_strategy()}\n\n{self.prompt_local_refinement_guidance()}'
        )

    def prompt_injection(self, algos: Sequence[HeuristicRecord]) -> str:
        prompt = (
            'Inject one novel, meaningful component into the following algorithm while preserving its main data flow. '
            'The component may be self-devised or inspired by ideas from other domains or problems, but it should '
            'modify or scale an existing priority expression rather than replace the whole algorithm.\n\n'
            f'{self.prompt_algo_details(algos)}{self.prompt_design_strategy()}\n\n'
            f'{self.prompt_local_refinement_guidance()}\n\n'
            'Use a concise noun phrase to describe the new component in the responded idea like '
            '"The new component ... has been introduced.".'
        )
        if len(self.injected_components) > 0:
            prompt += (
                ' Exclude the following components that have already been explored: '
                f'{", ".join(self.injected_components[-10:])}.'
            )
        return prompt

    def prompt_replacement(self, algos: Sequence[HeuristicRecord]) -> str:
        mode_specs = [
            (
                'a fixed, instance-independent decision rule',
                'an instance-dependent rule that derives its value from the current observation',
            ),
            (
                'a key hyper-parameter expressed as either a constant literal or a stationary variable',
                'a more principled constant justified by theory or practice',
            ),
            (
                'a fragment that assigns equal or near-equal credits to multiple elements',
                'a fragment where credits are deterministically and reasonably differentiated',
            ),
        ]
        p1, p2 = mode_specs[self.rs.choice(len(mode_specs))]
        return (
            f'For the following algorithm, identify {p1} and rewrite it to {p2}. '
            'Preserve the surrounding implementation and make the smallest coherent code change that '
            f'expresses the replacement.\n\n{self.prompt_algo_details(algos)}'
            f'{self.prompt_design_strategy()}\n\n{self.prompt_local_refinement_guidance()}'
        )

    def prompt_crossover(self, algos: Sequence[HeuristicRecord]) -> str:
        return (
            'Please generate a new algorithm that is motivated by the following algorithms but performs better '
            'on any same instance. Preserve the stronger working structure where possible and transfer at most '
            f'one clear component from the other algorithm.\n{self.prompt_algo_details(algos)}'
            f'{self.prompt_design_strategy()}\n\n{self.prompt_local_refinement_guidance()}\n        '
        )

    def prompt_algo_details(self, algos: Sequence[HeuristicRecord]) -> str:
        if len(algos) == 0:
            return ''
        sort_indices = np.argsort([a.perf for a in algos])[::-1]
        algos = [algos[i] for i in sort_indices]
        algo_detail = ''
        train_epoch = self._train_epoch_getter()
        for i, algo in enumerate(algos):
            score_profile = self.prompt_score_profile(algo)
            algo_detail += (
                f'## Algorithm {i + 1}\n'
                f'* Performance: {algo.perf_str} {self.problem_unit} (Rank: {i + 1})\n'
                f'* Score profile: {score_profile}\n'
                f'* Idea: {algo.idea}\n'
                f'* Code:```python\n{algo.code}```\n\n'
            )
            algo.last_used_epoch = train_epoch
        return algo_detail.strip()

    def prompt_score_profile(self, algo: HeuristicRecord) -> str:
        if algo.perfs is None:
            return 'not available'
        perfs = np.ravel(algo.perfs).astype(float)
        if len(perfs) == 0:
            return 'not available'
        summary = (
            f'mean={np.mean(perfs):.6f}, std={np.std(perfs):.6f}, '
            f'best={np.max(perfs):.6f}, worst={np.min(perfs):.6f}, n={len(perfs)}'
        )
        if len(perfs) <= 8:
            sample = perfs
        else:
            sample_indices = np.linspace(0, len(perfs) - 1, num=8, dtype=int)
            sample = perfs[sample_indices]
        sample_str = ', '.join(f'{x:.6f}' for x in sample)
        return f'{summary}; representative_scores=[{sample_str}]'

    def prompt_algo_requirements(self) -> str:
        return dedent("""\
            ## Your Task
            You should first present a concise conceptual description, followed by a complete code implementation.
            
            * The description must:
                * Be enclosed with a double brace and starts with "The idea of the algorithm is to".
                * Ensure it is self-contained, insightful, and creatively original.
                * Not reference or rely on any prior ideas or existing code.
            * The code must:
                * Use exactly the same function name, argument names, argument order, and type annotations shown in the provided template.
                * Not add, remove, rename, or reorder function arguments.
                * Strictly follow the input-output variable names and types used in the provided implementation.
                * Be a single Python function formatted within Python code blocks.
                * Exclude any usage examples.
                * Ensure the algorithm is deterministic.
                * Avoid introducing unnecessary, arbitrarily-tuned hyperparameters; any parameters used should be essential and systematically derived from the input.
                * When parent code is provided, prefer a minimal local refinement that preserves its working structure unless the prompt explicitly asks for a broader recombination.
                      
            Overall, your response should be like:
            {{The idea of the algorithm is to (sepcific description here)}}
            ```python
            your code here
            ```
            Except for the idea and code, do not give additional explanations or comments.\
        """)

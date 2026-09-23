"""Operator scheduling / prepare_dataset logic from CALM Trainer.prepare_dataset."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

import numpy as np

from .parse import idea_distance
from .prompt import Prompt, PromptBuilder

if TYPE_CHECKING:
    from .pool import HeuristicRecord
    from .task_config import CALMTaskHyperparams


def prepare_round_messages(
        *,
        algos: List[HeuristicRecord],
        seed_algos: Sequence[HeuristicRecord],
        used_prompts: List[Prompt],
        last_revisited_prompts: List[Prompt],
        hp: CALMTaskHyperparams,
        rs: np.random.RandomState,
        age_stuck: int,
        train_epoch: int,
        prompts: PromptBuilder,
        log_info: Callable[[str], None],
) -> Tuple[List[List[dict]], List[Prompt], List[Prompt], List[HeuristicRecord], int]:
    """Build the next round of chat messages.

    Returns:
        messages, used_prompts, last_revisited_prompts, algos (re-sorted), age_stuck
    """
    max_stuck_threshold = (
        hp.max_steps // 20 if hp.max_stuck_threshold < 0 else hp.max_stuck_threshold
    )
    if rs.random() < hp.speed_collapse * age_stuck or age_stuck >= max_stuck_threshold:
        log_info(f'\n Collapse after stucking for {age_stuck} rounds \n')
        age_stuck = 0
        if len(algos) > 0:
            used_prompts = []
            best_algo = algos[int(np.argmax([a.perf for a in algos]))]
            log_info(f'During collapse, the best algo with perf {best_algo.perf} has been kept')
            algos = [best_algo]
            for seed_algo in seed_algos:
                if seed_algo not in algos:
                    algos.append(seed_algo)

    sorted_indices = np.argsort([a.perf for a in algos])[::-1]
    algos = [algos[i] for i in sorted_indices]
    algos_head = algos[:hp.population_size]

    curr_used_prompts: List[Prompt] = []
    messages: List[List[dict]] = []

    ub_simplification = hp.ub_simplification
    ub_injection = hp.ub_injection
    ub_replacement = hp.ub_replacement
    ub_crossover = hp.ub_crossover

    n_ops = np.zeros(4, dtype=int)
    p_ops = np.array(
        [ub_simplification, ub_injection, ub_replacement, ub_crossover],
        dtype=float,
    )
    if len(algos) < 2:
        p_ops[-1] = 0
    if len(algos) < hp.population_size and ub_injection > 0:
        p_ops[1] = np.max(p_ops) + 1
    if np.sum(p_ops) <= 0:
        p_ops[:] = 1
        if len(algos) < 2:
            p_ops[-1] = 0
    p_ops /= np.sum(p_ops)
    sample_res = rs.choice(4, size=hp.n_prompts, p=p_ops, replace=True)
    for i_op in sample_res:
        n_ops[i_op] += 1
    ub_simplification, ub_injection, ub_replacement, ub_crossover = n_ops
    log_info(
        f'UB of OPs: simplification - {ub_simplification}, injection - {ub_injection}, '
        f'replacement - {ub_replacement}, crossover - {ub_crossover}'
    )

    rank = 1 + np.arange(len(algos_head))
    p = 1 / rank
    p /= np.sum(p)

    if len(algos_head) > 0:
        for upper_bound, prompt_template in zip(
            [ub_simplification, ub_injection, ub_replacement],
            [prompts.prompt_simplification, prompts.prompt_injection, prompts.prompt_replacement],
        ):
            n_new_prompts = 0
            n_trial = 0
            while n_trial <= 1000 and n_new_prompts < upper_bound:
                n_trial += 1
                indices = [rs.choice(len(algos_head), p=p)]
                algos_for_prompt = [algos_head[i] for i in indices]
                prompt = Prompt(prompt_template(algos_for_prompt))
                if prompt not in used_prompts:
                    used_prompts.append(prompt)
                if prompt not in curr_used_prompts:
                    curr_used_prompts.append(prompt)
                elif not hp.allow_duplicate_prompts:
                    continue
                messages.append([
                    {'role': 'system', 'content': prompts.system_prompt},
                    {'role': 'user', 'content': prompt.prompt},
                ])
                n_new_prompts += 1

        n_trial = 0
        n_new_crossover = 0
        while len(algos) >= 2 and n_trial < 1000 and n_new_crossover < ub_crossover:
            n_trial += 1
            algo_0_idx = rs.choice(len(algos_head), p=p)
            algo_0 = algos_head[algo_0_idx]
            log_msg = None
            if rs.random() <= 0.5:
                algo_1_idx = rs.choice(len(algos_head), p=p)
                if algo_0 == algos_head[algo_1_idx]:
                    continue
                log_msg = (
                    f'Crossover driven by performance: {algo_0.sid} (Rank {rank[algo_0_idx]}) x '
                    f'{algos_head[algo_1_idx].sid} (Rank {rank[algo_1_idx]})'
                )
                partner = algos_head[algo_1_idx]
            else:
                distances = [
                    -idea_distance(base_idea=algo_0.idea, new_idea=algo_1.idea)
                    for algo_1 in algos
                ]
                distance_rank = np.argsort(np.argsort(distances)) + 1
                distance_based_p = 1 / distance_rank
                distance_based_p /= np.sum(distance_based_p)
                algo_1_idx = rs.choice(len(algos), p=distance_based_p)
                if distances[algo_1_idx] == 0:
                    continue
                log_msg = (
                    f'Crossover driven by diversity: {algo_0.sid} (Rank {rank[algo_0_idx]}) x '
                    f'{algos[algo_1_idx].sid}, Distance: {distances[algo_1_idx]}'
                )
                partner = algos[algo_1_idx]
            prompt = Prompt(prompts.prompt_crossover([algo_0, partner]))
            if prompt not in used_prompts:
                used_prompts.append(prompt)
            if prompt not in curr_used_prompts:
                curr_used_prompts.append(prompt)
            elif not hp.allow_duplicate_prompts:
                continue
            messages.append([
                {'role': 'system', 'content': prompts.system_prompt},
                {'role': 'user', 'content': prompt.prompt},
            ])
            log_info(log_msg)
            n_new_crossover += 1

        last_revisited_prompts = []
        candidate_prompts = [
            prompt for prompt in used_prompts
            if prompt.n_calls > 0
            and not prompt.is_creation
            and prompt not in curr_used_prompts
            and prompt.age(train_epoch) >= hp.revisit_gap
            and prompt.feasible_algo_generated
        ]
        n_revisits = min(hp.ub_revisit, len(candidate_prompts))
        if n_revisits > 0:
            prompt_perfs = np.array([prompt.best_generated_algo_perf for prompt in candidate_prompts])
            prompt_rank = 1 + np.argsort(np.argsort(-prompt_perfs))
            prompt_p = 1 / prompt_rank
            prompt_p = prompt_p / np.sum(prompt_p)
            revisit_indices = rs.choice(
                len(candidate_prompts), size=n_revisits, p=prompt_p, replace=False,
            )
            for revisit_idx in revisit_indices:
                revisit_prompt = candidate_prompts[revisit_idx]
                last_revisited_prompts.append(revisit_prompt)
                log_info(f'Revisiting {revisit_prompt.op} prompt with {revisit_prompt.status_str}')
                messages.append([
                    {'role': 'system', 'content': prompts.system_prompt},
                    {'role': 'user', 'content': revisit_prompt.prompt},
                ])

        if len(messages) == 0:
            log_info('No prompts have been added, add creation')
            creation_text = prompts.prompt_creation
            messages.append([
                {'role': 'system', 'content': prompts.system_prompt},
                {'role': 'user', 'content': creation_text},
            ])
            if creation_text not in used_prompts:
                used_prompts.append(Prompt(creation_text))

    log_info(f'Prepared {len(messages)} prompts')
    return messages, used_prompts, last_revisited_prompts, algos, age_stuck

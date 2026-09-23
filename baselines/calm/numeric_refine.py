"""Numeric literal refinement ported from CALM Trainer."""

from __future__ import annotations

import ast
import copy
from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Set

import numpy as np

from .parse import extract_function_from_string
from .pool import HeuristicRecord

if TYPE_CHECKING:
    from .task_config import CALMTaskHyperparams


class NumericLiteralMutator(ast.NodeTransformer):
    def __init__(self, target_id, new_value):
        self.target_id = target_id
        self.new_value = new_value

    def visit_Constant(self, node):
        if getattr(node, '_numeric_refine_id', None) == self.target_id:
            return ast.copy_location(ast.Constant(value=self.new_value), node)
        return node


def code_signature(code: str) -> str:
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.name = '__candidate__'
        return ast.dump(tree, include_attributes=False)
    except SyntaxError:
        compact_lines = [line.strip() for line in code.splitlines() if line.strip()]
        return '\n'.join(compact_lines)


def numeric_literal_variants(code: str, max_variants: int, rs: np.random.RandomState) -> List[str]:
    if max_variants <= 0:
        return []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def has_ancestor(node, ancestor_type):
        parent = parents.get(id(node))
        while parent is not None:
            if isinstance(parent, ancestor_type):
                return True
            parent = parents.get(id(parent))
        return False

    targets = []
    for target_id, node in enumerate(ast.walk(tree)):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            continue
        value = float(node.value)
        if not np.isfinite(value) or abs(value) > 1e9:
            continue
        if has_ancestor(node, (ast.Subscript, ast.Slice)):
            continue
        node._numeric_refine_id = target_id
        targets.append((target_id, value))

    if not targets:
        return []

    target_order = rs.permutation(len(targets))
    factor_choices = np.array([0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0], dtype=float)
    zero_choices = np.array([-1.0, -0.5, 0.5, 1.0], dtype=float)
    variants = []
    variant_signatures = set()
    for target_pos in target_order:
        target_id, value = targets[int(target_pos)]
        if value == 0:
            proposals = list(zero_choices[rs.permutation(len(zero_choices))])
        else:
            proposals = [value * float(f) for f in factor_choices[rs.permutation(len(factor_choices))]]
        for new_value in proposals:
            if new_value == value or not np.isfinite(new_value):
                continue
            mutated_tree = copy.deepcopy(tree)
            mutated_tree = NumericLiteralMutator(target_id, float(new_value)).visit(mutated_tree)
            ast.fix_missing_locations(mutated_tree)
            try:
                variant_code = ast.unparse(mutated_tree)
            except Exception:
                continue
            variant_key = code_signature(variant_code)
            if variant_key in variant_signatures:
                continue
            variant_signatures.add(variant_key)
            variants.append(variant_code)
            if len(variants) >= max_variants:
                return variants
    return variants


def run_numeric_refinement(
        *,
        candidate_parents: Sequence[HeuristicRecord],
        algos: List[HeuristicRecord],
        hp: CALMTaskHyperparams,
        rs: np.random.RandomState,
        evaluations_used: int,
        evaluation_budget: int,
        log_step: int,
        seen_code_signatures: Set[str],
        numeric_refine_attempted: Set,
        matches_expected_signature: Callable,
        evaluate_code: Callable[[str], tuple],
        is_seen_performance_profile: Callable,
        record_performance_profile: Callable,
        best_perf: float,
        log_info: Callable[[str], None],
        register_accepted: Optional[Callable[[HeuristicRecord], None]] = None,
        on_new_best: Optional[Callable[[HeuristicRecord], None]] = None,
        variants_per_parent=None,
        top_k=None,
        parent_limit=None,
        source_label: str = 'NUMERIC_REFINE',
) -> tuple[List[HeuristicRecord], int, float]:
    """Returns (refined_algos, evaluations_used, best_perf)."""
    if variants_per_parent is None:
        variants_per_parent = hp.numeric_refine_variants
    variants_per_parent = max(0, int(variants_per_parent))
    if variants_per_parent <= 0 or evaluations_used >= evaluation_budget:
        return [], evaluations_used, best_perf

    if parent_limit is None:
        parent_limit = hp.numeric_refine_parent_limit
    parent_limit = max(1, int(parent_limit))
    if top_k is None:
        top_k = hp.numeric_refine_top_k
    top_k = max(0, int(top_k))
    archive_parents = algos[:top_k] if top_k > 0 else []
    parent_pool = [
        a for a in list(candidate_parents) + list(archive_parents)
        if getattr(a, 'perf', None) is not None
    ]
    parent_pool = sorted(parent_pool, key=lambda a: a.perf, reverse=True)

    unique_parents = []
    seen_parent_keys = set()
    seen_parent_profiles = []
    profile_tolerance = float(hp.numeric_refine_parent_profile_tolerance)
    for parent in parent_pool:
        parent_key = getattr(parent, 'code_key', None) or code_signature(parent.code)
        if parent_key in seen_parent_keys:
            continue
        if profile_tolerance >= 0 and getattr(parent, 'perfs', None) is not None:
            parent_profile = np.ravel(parent.perfs).astype(float)
            is_duplicate_profile = any(
                parent_profile.shape == seen_profile.shape
                and np.max(np.abs(parent_profile - seen_profile)) <= profile_tolerance
                for seen_profile in seen_parent_profiles
            )
            if is_duplicate_profile:
                continue
            seen_parent_profiles.append(parent_profile.copy())
        seen_parent_keys.add(parent_key)
        unique_parents.append(parent)
        if len(unique_parents) >= parent_limit:
            break

    variant_algos: List[HeuristicRecord] = []
    variant_parents: List[HeuristicRecord] = []
    batch_signatures = set()
    for parent in unique_parents:
        parent_key = getattr(parent, 'code_key', None) or code_signature(parent.code)
        parent_attempt_key = (parent_key, log_step)
        if parent_attempt_key in numeric_refine_attempted:
            continue
        numeric_refine_attempted.add(parent_attempt_key)
        n_parent_variants = 0
        for variant_code in numeric_literal_variants(parent.code, variants_per_parent * 8, rs):
            variant_key = code_signature(variant_code)
            if variant_key in seen_code_signatures or variant_key in batch_signatures:
                continue
            step_func = extract_function_from_string(variant_code)
            if step_func is None or not matches_expected_signature(step_func):
                continue
            algo = HeuristicRecord(
                code=variant_code,
                idea=(
                    'The idea of the algorithm is to preserve a strong implementation while '
                    'locally perturbing one numeric literal and keeping the same code structure.'
                ),
                name=f'numeric_refine({log_step})',
                parent_prompt_type='numeric_refine',
                birth=log_step,
                code_key=variant_key,
            )
            variant_algos.append(algo)
            variant_parents.append(parent)
            batch_signatures.add(variant_key)
            n_parent_variants += 1
            if n_parent_variants >= variants_per_parent:
                break

    remaining_evaluations = evaluation_budget - evaluations_used
    if remaining_evaluations <= 0:
        return [], evaluations_used, best_perf
    if len(variant_algos) > remaining_evaluations:
        variant_algos = variant_algos[:remaining_evaluations]
        variant_parents = variant_parents[:remaining_evaluations]
    if not variant_algos:
        return [], evaluations_used, best_perf

    evaluations_used += len(variant_algos)
    refined_algos = []
    for algo, parent in zip(variant_algos, variant_parents):
        score, perfs, status = evaluate_code(algo.code)
        if status == 'code_bug' or score is None:
            log_info(
                f'{source_label} | Based on: [{parent.sid}] | Result: code_bug | '
                f'Evals: {evaluations_used}/{evaluation_budget}'
            )
            continue
        if 'random' in algo.code or 'np.random' in algo.code:
            log_info(
                f'{source_label} | Based on: [{parent.sid}] | Result: random_algorithm | '
                f'Evals: {evaluations_used}/{evaluation_budget}'
            )
            continue
        algo.perf = float(score)
        algo.perfs = np.asarray(perfs, dtype=float).copy()
        seen_code_signatures.add(algo.code_key)
        if is_seen_performance_profile(algo.perfs):
            log_info(
                f'{source_label} | Based on: [{parent.sid}] | Result: duplicate_performance_profile | '
                f'Perf: {algo.perf:.6f} | Evals: {evaluations_used}/{evaluation_budget}'
            )
            continue
        record_performance_profile(algo.perfs)
        is_new_best = algo.perf > best_perf
        is_new = algo not in algos
        if is_new:
            algos.append(algo)
            refined_algos.append(algo)
            if register_accepted is not None:
                register_accepted(algo)
        is_better = algo.perf > parent.perf
        log_info(
            f'{source_label} | Based on: [{parent.sid}] | Perf: {algo.perf:.6f} | '
            f'New Best: {is_new_best} | Better than parent: {is_better} | '
            f'Evals: {evaluations_used}/{evaluation_budget}'
        )
        if is_new_best:
            best_perf = algo.perf
            log_info(f'New best performance: {algo.perf} at step {log_step} by {algo.name}')
            log_info(f'Idea: {algo.idea}')
            if on_new_best is not None:
                on_new_best(algo)
    return refined_algos, evaluations_used, best_perf

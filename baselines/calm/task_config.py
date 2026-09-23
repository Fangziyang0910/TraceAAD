"""Per-task CALM search hyperparameters from reference configs/*/local.yaml."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class CALMTaskHyperparams:
    population_size: int = 10
    n_prompts: int = 1
    n_generations: int = 4
    ub_simplification: int = 1
    ub_injection: int = 1
    ub_replacement: int = 2
    ub_crossover: int = 4
    ub_revisit: int = 0
    revisit_gap: int = 4
    allow_duplicate_prompts: bool = False
    speed_collapse: float = 0.0
    max_stuck_threshold: int = 1000
    max_steps: int = 500
    generation_temperature: float = 0.8
    generation_top_p: float = 0.95
    numeric_refine_variants: int = 4
    numeric_refine_top_k: int = 5
    numeric_refine_parent_limit: int = 2
    numeric_refine_initial_variants: int = 32
    numeric_refine_parent_profile_tolerance: float = 1e-12
    archive_profile_tolerance: float = 0.0
    reward_idea_not_exist: float = -1.0
    reward_code_not_exist: float = -0.95
    reward_function_not_exist: float = -0.90
    reward_bug_in_function: float = -0.85
    reward_random_algorithm: float = -0.75
    profile_reward_max_bonus: float = 0.25
    profile_reward_worst_weight: float = 0.18
    profile_reward_consistency_weight: float = 0.12
    profile_reward_non_worse_fraction: float = 0.5
    profile_reward_gate_non_worse_fraction: float = 0.75
    profile_reward_gate_worst_tolerance: float = 0.02
    # Display / system-prompt metadata from CALM problems/*.py
    problem_name: str = 'problem'
    problem_unit: str = 'units'


# From reference_code/CALM/configs/<task>/local.yaml (+ problems/*.py name/unit).
TASK_HYPERPARAMS: Dict[str, CALMTaskHyperparams] = {
    'online_bin_packing': CALMTaskHyperparams(
        population_size=8,
        n_generations=5,
        ub_simplification=1,
        ub_injection=1,
        ub_replacement=2,
        ub_crossover=2,
        ub_revisit=0,
        revisit_gap=6,
        numeric_refine_variants=8,
        numeric_refine_top_k=5,
        numeric_refine_parent_limit=1,
        numeric_refine_initial_variants=24,
        numeric_refine_parent_profile_tolerance=3.898645735341032e-13,
        archive_profile_tolerance=0.01283387864234295,
        profile_reward_max_bonus=0.34330751130745446,
        profile_reward_worst_weight=0.33095716581974916,
        profile_reward_consistency_weight=0.0777695103055566,
        profile_reward_non_worse_fraction=0.6335035091844401,
        profile_reward_gate_non_worse_fraction=0.8294723203513769,
        profile_reward_gate_worst_tolerance=0.06449459786077394,
        max_stuck_threshold=1000,
        generation_temperature=0.6182571366832558,
        generation_top_p=0.879554458037353,
        problem_name='online bin packing',
        problem_unit='percent of the gap to the lower bound',
    ),
    'tsp_construct': CALMTaskHyperparams(
        population_size=10,
        n_generations=3,
        ub_simplification=0,
        ub_injection=2,
        ub_replacement=1,
        ub_crossover=2,
        ub_revisit=0,
        revisit_gap=8,
        numeric_refine_variants=2,
        numeric_refine_top_k=5,
        numeric_refine_parent_limit=4,
        numeric_refine_initial_variants=32,
        numeric_refine_parent_profile_tolerance=1.3215119569781297e-11,
        archive_profile_tolerance=0.048232046898646054,
        profile_reward_max_bonus=0.18516527955642376,
        profile_reward_worst_weight=0.19796861871741245,
        profile_reward_consistency_weight=0.24715488287385526,
        profile_reward_non_worse_fraction=0.7323408451721475,
        profile_reward_gate_non_worse_fraction=0.6083970808062072,
        profile_reward_gate_worst_tolerance=0.044754383774483764,
        max_stuck_threshold=500,
        generation_temperature=0.9612113151100183,
        generation_top_p=0.8822595888215602,
        problem_name='traveling salesman',
        problem_unit='length units of the tour',
    ),
    'op_aco': CALMTaskHyperparams(
        population_size=8,
        n_generations=3,
        ub_simplification=0,
        ub_injection=2,
        ub_replacement=2,
        ub_crossover=2,
        ub_revisit=2,
        revisit_gap=3,
        numeric_refine_variants=2,
        numeric_refine_top_k=4,
        numeric_refine_parent_limit=3,
        numeric_refine_initial_variants=32,
        numeric_refine_parent_profile_tolerance=3.935252877376601e-09,
        archive_profile_tolerance=0.014408610943009183,
        profile_reward_max_bonus=0.30539686769658453,
        profile_reward_worst_weight=0.2469072618672435,
        profile_reward_consistency_weight=0.2068623246134137,
        profile_reward_non_worse_fraction=0.6694579402693874,
        profile_reward_gate_non_worse_fraction=0.6618604523557647,
        profile_reward_gate_worst_tolerance=0.07223349436725184,
        max_stuck_threshold=500,
        generation_temperature=0.7592716186561026,
        generation_top_p=0.9340793100759698,
        problem_name='orienteering',
        problem_unit='units of collected reward',
    ),
    'cvrp_aco': CALMTaskHyperparams(
        population_size=12,
        n_generations=4,
        ub_simplification=2,
        ub_injection=1,
        ub_replacement=3,
        ub_crossover=3,
        ub_revisit=1,
        revisit_gap=5,
        numeric_refine_variants=4,
        numeric_refine_top_k=8,
        numeric_refine_parent_limit=2,
        numeric_refine_initial_variants=24,
        numeric_refine_parent_profile_tolerance=4.6288991663972227e-10,
        archive_profile_tolerance=0.04216964713742902,
        profile_reward_max_bonus=0.23614906961856566,
        profile_reward_worst_weight=0.1801574912336565,
        profile_reward_consistency_weight=0.12036100574624126,
        profile_reward_non_worse_fraction=0.7315943644213531,
        profile_reward_gate_non_worse_fraction=0.6993058297429222,
        profile_reward_gate_worst_tolerance=0.03045822541000411,
        max_stuck_threshold=1500,
        generation_temperature=0.5746373293518763,
        generation_top_p=0.9037059170405967,
        problem_name='capacitated vehicle routing',
        problem_unit='units of travel distance',
    ),
    # No reference config exists for VRPTW; search behavior uses the generic
    # CALM defaults and only the display metadata is task-specific.
    'vrptw_construct': CALMTaskHyperparams(
        problem_name='vehicle routing with time windows',
        problem_unit='units of travel distance',
    ),
}


def resolve_task_key(evaluation) -> str:
    """Map an Evaluation instance to a CALM task hyperparam key."""
    module = type(evaluation).__module__
    for key in TASK_HYPERPARAMS:
        if key in module:
            return key
    name = type(evaluation).__name__.lower()
    mapping = {
        'obpevaluation': 'online_bin_packing',
        'tspevaluation': 'tsp_construct',
        'opacoevaluation': 'op_aco',
        'cvrpacoevaluation': 'cvrp_aco',
    }
    if name in mapping:
        return mapping[name]
    raise ValueError(f'Cannot resolve CALM task key for evaluation {type(evaluation)!r}')


def get_task_hyperparams(task_key: str, overrides: Optional[Dict[str, Any]] = None) -> CALMTaskHyperparams:
    base = TASK_HYPERPARAMS.get(task_key, CALMTaskHyperparams())
    if not overrides:
        return base
    data = asdict(base)
    for key, value in overrides.items():
        if key in data and value is not None:
            data[key] = value
    return CALMTaskHyperparams(**data)

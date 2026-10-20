"""RefineEvo's evolve loop (samwu-learn/RefineEvo@7de015d5 evo_agent.py, evo_interface.py), run serially.

Local adaptations are documented in README.md.
"""

import heapq
import math
import random
import re
import time

import numpy as np

from core import SecureEvaluator
from core.code import function_from_source, parse_program, replace_function
from core.llm import generate
from baselines.shared.checkpoint import save_checkpoint
from baselines.shared.observability import (
    close_llm, finish_profiler, init_observability, is_search_aborted, log_error,
    log_event, log_llm_call, log_state, record_sample_failure, reset_sample_failures,
)
from .planner import Planner
from .pool import ExperiencePool, process_memories
from .utils import PROMPTS, ensure_required_imports, extract_algorithm_and_code, messages, text

# EvolutionOperators.init_prompt
OPERATOR_DESCRIPTIONS = {
    "op1": "create a new algorithm that has a totally different form from the given ones",
    "op2": "create a new algorithm that has a totally different form from the given ones but can be motivated from them",
    "op3": "create a new algorithm that has a different form but can be a modified version of the algorithm provided",
    "op4": "identify the main algorithm parameters and create a new algorithm that has a different parameter settings of the score function provided",
    "op5": "identify the main components in the function and simplify the components to enhance the generalization to potential out-of-distribution instances"}
N_PARENTS = {"op1": 2, "op2": 2, "op3": 1, "op4": 1, "op5": 1}
# EvolutionInterface.get_offspring
MAX_GENERATION_ATTEMPTS = 5
MAX_DUPLICATE_RETRIES = 3


def prob_rank(population, m):
    """
    Probability-based rank selection
    Higher rank individuals have higher probability of being selected
    """
    ranks = [i for i in range(len(population))]
    probs = [1 / (rank + 1 + len(population)) for rank in ranks]
    parents = random.choices(population, weights=probs, k=m)
    return parents


def pop_greedy(population, size):
    """
    Greedy population management
    Keep the size individuals with best fitness
    """
    # Filter out individuals with None objective
    pop = [individual for individual in population if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)

    # Deduplicate: keep only one individual for same objective
    unique_pop = []
    unique_objectives = []
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual['objective'])

    # Select the best size individuals
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])
    return pop_new


def add2pop(population, offspring):
    """
    Add offspring to population
    Automatically check for duplicate objective values
    """
    dup_obj_count = 0
    for off in offspring:
        # Check for duplicate objectives
        is_duplicate = False
        for ind in population:
            if ind['objective'] == off['objective']:
                is_duplicate = True
                dup_obj_count += 1
                break

        if not is_duplicate:
            population.append(off)

    return population, dup_obj_count


def build_query(parents):
    """
    Construct query text for experience retrieval from the parents' core information
    """
    # Parent summaries
    parent_summaries = []
    for idx, parent in enumerate(parents or [], 1):
        obj_val = parent.get("objective")
        algorithm_desc = parent.get("algorithm") or ""
        algorithm_desc = re.sub(r"\s+", " ", algorithm_desc).strip()
        if len(algorithm_desc) > 400:
            algorithm_desc = algorithm_desc[:400] + " ..."

        meta_bits = []
        if obj_val is not None:
            meta_bits.append(f"objective={obj_val}")

        meta_prefix = "; ".join(meta_bits) if meta_bits else "meta=unknown"
        parent_summaries.append(f"Parent {idx} ({meta_prefix}): {algorithm_desc}")

    if not parent_summaries:
        parent_summaries.append("Parents: none selected")

    return "\n".join(parent_summaries)


def check_improvement(parents, offspring):
    """True when the offspring beats every parent (minimization)."""
    if not parents or not offspring:
        return False
    offspring_obj = offspring.get('objective')
    if offspring_obj is None:
        return False
    parent_objectives = []
    for parent in parents:
        if isinstance(parent, dict) and parent.get('objective') is not None:
            try:
                parent_objectives.append(float(parent['objective']))
            except (TypeError, ValueError):
                continue
    if not parent_objectives:
        return False
    try:
        offspring_obj_val = float(offspring_obj)
    except (TypeError, ValueError):
        return False
    return offspring_obj_val < min(parent_objectives)


class Terminated(Exception):
    """A state the upstream run cannot continue from; the best program so far is kept."""


class RefineEvo:
    def __init__(self, llm, evaluation, embedding_fn, profiler=None, max_sample_nums=1000, *,
                 pop_size=10, init_pop_size=30, top_k=3, debug_mode=False, **evaluator_options):
        if max_sample_nums < 1:
            raise ValueError("Budget must be positive")
        template = parse_program(str(evaluation.template_program))
        if template is None or len(template.functions) != 1:
            raise ValueError("RefineEvo requires exactly one task entrypoint")
        self._template_program = template
        self._func_name = template.functions[0].name
        module = type(evaluation).__module__
        self._problem_name = module.split('.')[-2] if '.' in module else module
        problem_desc = evaluation.task_description
        func_desc = template.functions[0].docstring or ''
        # Upstream seed_func.txt holds the reference module; the task template plays that role.
        self._seed_func = str(template).strip()
        knowledge = PROMPTS / self._problem_name / 'external_knowledge.txt'
        self._external_knowledge = knowledge.read_text() if knowledge.is_file() else ''
        self._system_generator = text('system_generator').format(
            func_name=self._func_name, problem_desc=problem_desc, func_desc=func_desc)
        self._llm = llm
        self._embedding_fn = embedding_fn
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_options)
        self._profiler = profiler
        self._max_sample_nums = max_sample_nums
        self._pop_size, self._init_pop_size = pop_size, init_pop_size
        self._debug_mode = debug_mode
        self._tot_sample_nums = 0
        # Search state, saved at completed boundaries.
        self._phase = 'init'
        self._init_index = 0
        self._init_population = []
        self._population = []
        self._generation = 0
        self._id_counter = 0
        self._op_desc = dict(OPERATOR_DESCRIPTIONS)
        self._strategy_history = []
        self._improvement_history = []
        self._planner = Planner(self._ask, self._op_desc)
        self._pool = ExperiencePool(self._ask, self._embed, problem_desc, func_desc, top_k=top_k)
        init_observability(self)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    def _has_budget(self):
        return not is_search_aborted(self) and self._tot_sample_nums < self._max_sample_nums

    def run(self):
        status, payload = 'finished', {}
        try:
            save_checkpoint(self, 'refineevo', 'ready')
            if self._phase == 'init':
                self._initialize_population()
            while self._phase == 'evolve' and self._has_budget():
                self._evolve_generation()
                save_checkpoint(self, 'refineevo', 'generation')
        except Terminated as exc:
            payload = dict(search_complete=True, termination=str(exc))
            log_event(self, event='search_terminated', status='terminated', reason=str(exc),
                      sample_order=self._tot_sample_nums)
        except KeyboardInterrupt:
            status = 'interrupted'
        except Exception as exc:
            status = 'aborted'
            self._search_aborted = True
            log_error(self, 'run', exc, method='refineevo', counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            finish_profiler(self, status=status, refineevo={
                **self._planner.get_summary(), 'operator_improvements': len(self._improvement_history),
                'memories': len(self._pool.data)}, **payload)
            close_llm(self._llm)
            close_llm(self._embedding_fn)

    # ----- model and embedding calls -----

    def _ask(self, prompt, stage, operator=None):
        try:
            result = generate(self._llm, prompt)
        except Exception as exc:
            record_sample_failure(self, exc, stage=stage, operator=operator, prompt=prompt,
                                  counts_budget=False)
            return None
        reset_sample_failures(self)
        log_llm_call(self, llm=self._llm, transport_calls=result['calls'], method='refineevo', stage=stage,
                     operator=operator, sample_order=self._tot_sample_nums + 1, iteration=self._generation)
        return result['content'] or ''

    def _embed(self, query):
        """Upstream swallows embedding errors; here they stop the search at the last checkpoint."""
        started = time.monotonic()
        try:
            vector = self._embedding_fn(query)
        except Exception as exc:
            log_event(self, event='embedding_call', status='error', error=f"{type(exc).__name__}: {exc}",
                      counts_budget=False, sample_order=self._tot_sample_nums,
                      usage=getattr(self._embedding_fn, 'last_usage', {}))
            raise
        log_event(self, event='embedding_call', status='ok', counts_budget=False,
                  sample_order=self._tot_sample_nums, seconds=time.monotonic() - started,
                  usage=getattr(self._embedding_fn, 'last_usage', {}))
        return vector

    # ----- offspring generation and evaluation -----

    def _make_individual_id(self, operator, parents=None):
        self._id_counter += 1
        return {'uid': self._id_counter, 'operator': operator,
                'parents': [p['id']['uid'] for p in parents or []]}

    def _get_algorithm(self, pop, operator):
        """One generator call; None if the call failed. Retrieval happens before the call."""
        memories, vector = [], None
        if operator == "init":
            parents, query = None, None
            ext_knowledge = self._external_knowledge or "No additional domain knowledge provided."
            user = text('user_init').format(seed_func=self._seed_func, external_knowledge=ext_knowledge)
        else:
            parents = prob_rank(pop, N_PARENTS[operator])
            op_desc = self._op_desc[operator]
            query = build_query(parents)
            if query:
                memories, vector = self._pool.retrieve_memories(query, operator_goal=op_desc)
            # Process experiences, separating successful and failed experiences
            processed_memories = process_memories(memories)
            # Format parent algorithms with clear structure
            prompt_indiv = ""
            for i in range(len(parents)):
                prompt_indiv += f"\n**Parent Algorithm {i + 1}**\n"
                prompt_indiv += f"Description: {{{parents[i]['algorithm']}}}\n"
                prompt_indiv += f"Code:\n```python\n{parents[i]['code']}\n```\n"
            user = text('user_operate').format(
                n_parents=len(parents), prompt_indiv=prompt_indiv, success_exp=processed_memories['success_exp'],
                fail_exp=processed_memories['fail_exp'], op_desc=op_desc)
        started = time.monotonic()
        response = self._ask(messages(self._system_generator, user), 'generate', operator)
        if response is None:
            return None
        self._tot_sample_nums += 1  # Every returned generation spends budget.
        return parents, query, memories, vector, response, time.monotonic() - started

    def _get_offspring(self, pop, operator):
        """EvolutionInterface.get_offspring up to evaluation; None if no generation returned.

        Returns the slot's parents, query, retrieved memories, query vector and evaluated offspring.
        """
        generation_attempts = duplicate_attempts = 0
        last_generation_error = None
        p, q, retrieved_memories, vector = [], None, [], None
        while True:
            if not self._has_budget():
                return None
            drawn = self._get_algorithm(pop, operator)
            if drawn is None:
                return None  # Transport failure; the slot is skipped without spending budget.
            generation_attempts += 1
            parents, query, memories, query_vector, response, sample_time = drawn
            extracted = extract_algorithm_and_code(response)
            if extracted is None:
                last_generation_error = "LLM response missing algorithm description or python code block."
                log_event(self, event='sample_rejected', status='parse_failed', operator=operator,
                          attempt=generation_attempts, sample_order=self._tot_sample_nums, counts_budget=True)
                if generation_attempts >= MAX_GENERATION_ATTEMPTS:
                    offspring = {'code': '', 'objective': None, 'algorithm': '', 'parents': [],
                                 'id': self._make_individual_id(operator, []), 'other_inf': {}}
                    retrieved_memories = []
                    break
                continue
            last_generation_error = None
            p, q, retrieved_memories, vector = parents or [], query, memories, query_vector
            offspring = {'algorithm': extracted[1], 'code': extracted[0], 'objective': None,
                         'id': self._make_individual_id(operator, parents or []), 'other_inf': None}
            if any(extracted[0] == ind['code'] for ind in pop) and duplicate_attempts < MAX_DUPLICATE_RETRIES:
                duplicate_attempts += 1
                log_event(self, event='sample_rejected', status='duplicate', operator=operator,
                          parent_ids=[i.get('candidate_id') for i in p], sample_order=self._tot_sample_nums,
                          counts_budget=True)
                continue
            break

        offspring['other_inf'] = offspring.get('other_inf') or {}
        offspring['other_inf'].update(duplicate_attempts=duplicate_attempts, generation_attempts=generation_attempts)
        if last_generation_error:
            offspring['other_inf']['generation_error'] = last_generation_error
        code = offspring.get('code', '')
        if code and code.strip():
            self._evaluate(offspring, p, sample_time)
        else:
            offspring['other_inf']['evaluation_skipped'] = True
        return p, q, retrieved_memories, vector, offspring

    def _evaluate(self, offspring, parents, sample_time):
        """Upstream writes the code to gpt.py and imports the entrypoint by name."""
        source = ensure_required_imports(offspring['code'])
        function = function_from_source(source, self._func_name)
        program = replace_function(function, self._template_program) if function is not None else None
        parent_ids = [p.get('candidate_id') for p in parents]
        if program is None:
            offspring['other_inf']['error'] = f"No parsable module defining {self._func_name}."
            log_event(self, event='sample_rejected', status='invalid_program', operator=offspring['id']['operator'],
                      parent_ids=parent_ids, sample_order=self._tot_sample_nums, counts_budget=True)
            return
        score, seconds = self._evaluator.evaluate_program_record_time(program)
        valid = score is not None and math.isfinite(float(score))
        function.score = float(score) if valid else None
        function.operator = offspring['id']['operator']
        function.algorithm = offspring['algorithm']
        function.sample_time, function.evaluate_time = sample_time, seconds
        function.parent_id = parent_ids[0] if parent_ids else None
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program), budget_used=self._tot_sample_nums)
        offspring['candidate_id'] = self._profiler._num_samples if self._profiler else self._tot_sample_nums
        if valid:
            offspring['objective'] = float(np.round(float(score), 5))
        log_event(self, event='sample_registered', status='valid' if valid else 'invalid',
                  operator=function.operator, candidate_id=offspring['candidate_id'], parent_ids=parent_ids,
                  score=function.score, sample_order=self._tot_sample_nums, counts_budget=True)

    def _get_algorithm_batch(self, pop, operator):
        """EvolutionInterface.get_algorithm: pop_size offspring for one operator.

        Upstream runs the pop_size slots in parallel threads, so every slot retrieves from the pool as it
        stood when the batch began. Serially, experience writes therefore wait until the batch ends.
        """
        slots = []
        for _ in range(self._pop_size):
            if not self._has_budget():
                break
            slot = self._get_offspring(pop, operator)
            if slot is not None:
                slots.append(slot)
        # With the budget spent, the search ends and these reflections would never be read.
        if self._has_budget():
            for p, q, memories, vector, offspring in slots:
                self._add_experience(q, p, offspring, vector)
                # Update score of retrieved memories
                if memories and p:
                    improved = check_improvement(p, offspring)
                    for memory in memories:
                        score = self._pool.update_memory_score(memory, improved)
                        log_event(self, event='memory_score', memory_id=memory['_id'], improved=improved,
                                  score=score, deleted=score is not None and score < 0,
                                  sample_order=self._tot_sample_nums)
        return [slot[0] for slot in slots], [slot[4] for slot in slots]

    def _build_trajectory(self, parents, offspring):
        parent_infos = []
        parent_objectives = []
        for parent in parents or []:
            if not isinstance(parent, dict):
                continue
            parent_obj = parent.get("objective")
            try:
                parent_obj_val = float(parent_obj) if parent_obj is not None else None
            except (TypeError, ValueError):
                parent_obj_val = None
            if parent_obj_val is not None:
                parent_objectives.append(parent_obj_val)
            parent_infos.append({
                "objective": parent_obj_val,
                "algorithm": (parent.get("algorithm") or "")[:400]
            })

        offspring_obj = offspring.get("objective")
        try:
            offspring_obj_val = float(offspring_obj) if offspring_obj is not None else None
        except (TypeError, ValueError):
            offspring_obj_val = None

        raw_id = offspring.get("id") or {}
        operator_desc = self._op_desc.get(raw_id.get("operator"), "")
        if parent_objectives:
            baseline = min(parent_objectives)
            improvement = baseline - offspring_obj_val if offspring_obj_val is not None else None
        else:
            baseline = None
            improvement = None

        return {
            "operator_goal": operator_desc,
            "parents": parent_infos,
            "offspring": {
                "objective": offspring_obj_val,
                "algorithm": (offspring.get("algorithm") or "")[:400]
            },
            "objective_type": "min",
            "best_parent_objective": baseline,
            "improvement": improvement,
            "problem_name": self._problem_name
        }

    def _add_experience(self, query, parents, offspring, vector):
        if not query or not query.strip():
            return
        record = self._pool.add_experience(query, self._build_trajectory(parents, offspring), vector)
        if record is not None:
            metadata = record['metadata']
            log_event(self, event='experience_added', memory_id=record['_id'], is_success=metadata['is_success'],
                      operator=offspring['id']['operator'], candidate_id=offspring.get('candidate_id'),
                      distilled_items=metadata['distilled_items'], pool_size=len(self._pool.data),
                      sample_order=self._tot_sample_nums)

    # ----- initialization -----

    def _initialize_population(self):
        """EvolutionInterface.population_init, then population management."""
        target = int(self._init_pop_size / self._pop_size) * self._pop_size
        while self._init_index < target and self._has_budget():
            slot = self._get_offspring([], 'init')
            if slot is not None:
                self._init_population.append(slot[4])
            self._init_index += 1
            save_checkpoint(self, 'refineevo', 'initialization')
        if self._init_index < target:
            return
        population = pop_greedy(self._init_population, self._pop_size)
        if not population:
            # The planner's population statistics need a valid individual; upstream raises there.
            raise Terminated("No valid individual in the initial population.")
        self._population, self._init_population = population, []
        self._phase = 'evolve'
        self._log_population()

    # ----- one generation of EvoAgent.evolve() -----

    def _evolve_generation(self):
        population = self._population
        decision = self._planner.decide_strategy(population=population, generation=self._generation)
        selected_operators = decision.get("selected_operators")
        self._strategy_history.append({"generation": self._generation, "strategy": decision.get("strategy"),
                                       "operators": selected_operators, "reasoning": decision.get("reasoning")})
        log_event(self, event='strategy', status='fallback' if self._planner.fallback else 'llm',
                  fallback=self._planner.fallback, generation=self._generation, strategy=decision.get("strategy"),
                  operators=list(selected_operators), sample_order=self._tot_sample_nums)

        # Dynamically improve operators (based on LLM decision)
        for op in selected_operators:
            needs_improvement, stagnation_reason = self._planner.check_operator_needs_improvement(op, population)
            if not needs_improvement:
                continue
            improvement = self._planner.improve_operator(op, population)
            new_description = improvement.get("new_description") if improvement else None
            if new_description:
                # Synchronize operator description update to evo.operators
                self._op_desc[op] = new_description
                self._improvement_history.append({
                    "generation": self._generation, "operator_id": op,
                    "improvement_type": improvement.get("improvement_type", "unknown"),
                    "reasoning": improvement.get("reasoning", ""), "confidence": improvement.get("confidence", 0.0),
                    "new_description": new_description})
            log_event(self, event='operator_refinement', status='improved' if new_description else 'failed',
                      operator=op, trigger=stagnation_reason, new_description=new_description,
                      improvement_type=improvement.get("improvement_type") if improvement else None,
                      generation=self._generation, sample_order=self._tot_sample_nums)

        offsprings_gen = []
        self._generation += 1
        # Apply operators sequentially (using selected operators)
        for op in selected_operators:
            if not self._has_budget():
                break
            _, offsprings = self._get_algorithm_batch(population, op)
            # Update planner's operator performance statistics
            self._planner.update_operator_performance(op, offsprings, generation=self._generation)
            offsprings_gen.extend(offsprings)

        # Population management
        population, dup_obj_skips_gen = add2pop(population, offsprings_gen)
        size_act = min(len(population) + len(offsprings_gen), self._pop_size)
        self._population = pop_greedy(population, size_act)
        self._log_population(duplicate_objectives=dup_obj_skips_gen)

    def _log_population(self, **payload):
        best = self._population[0] if self._population else None
        log_state(self, phase='population', method='refineevo', generation=self._generation,
                  population=[[i.get('candidate_id'), i['id']['operator'], i['objective']] for i in self._population],
                  best=best.get('candidate_id') if best else None, best_score=best['objective'] if best else None,
                  operator_descriptions=dict(self._op_desc), memories=len(self._pool.data),
                  sample_count=self._tot_sample_nums, **payload)

"""HSEvo's evolve loop (datphamvn/HSEvo@f3afffab hsevo.py), run serially under the shared budget.

Local adaptations are documented in README.md.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import SecureEvaluator
from core.code import parse_program, replace_function
from core.llm import generate
from baselines.shared.checkpoint import save_checkpoint
from baselines.shared.observability import (
    close_llm, finish_profiler, init_observability, is_search_aborted, log_error,
    log_event, log_llm_call, log_state, record_sample_failure, reset_sample_failures,
)
from .sampler import GLOBAL_IMPORTS, code_to_function, extract_code_from_generator, extract_to_hs, filter_code, rename

PROMPTS = Path(__file__).with_name('prompts')
SCIENTISTS = [
    "You are an expert in the domain of optimization heuristics.",
    "You are Albert Einstein, relativity theory developer.",
    "You are Isaac Newton, the father of physics.",
    "You are Marie Curie, pioneer in radioactivity.",
    "You are Nikola Tesla, master of electricity.",
    "You are Galileo Galilei, champion of heliocentrism.",
    "You are Stephen Hawking, black hole theorist.",
    "You are Richard Feynman, quantum mechanics genius.",
    "You are Rosalind Franklin, DNA structure revealer.",
    "You are Ada Lovelace, computer programming pioneer."
]
# evolve() retries harmony search this many times per generation.
HS_TRIES = 3


def text(name):
    return (PROMPTS / 'common' / (name + '.txt')).read_text()


def messages(system, user):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]


@dataclass(eq=False)
class Individual:
    """An upstream individual dict; failed or unparsed programs keep obj = inf."""
    code: str | None
    operator: str
    obj: float = math.inf
    exec_success: bool = False
    tryHS: bool = False
    candidate_id: int | None = None


class Terminated(Exception):
    """evolve() catches RuntimeError and returns the best program found so far."""


class HSEvo:
    def __init__(self, llm, evaluation, profiler=None, max_sample_nums=1000, *, pop_size=10,
                 init_pop_size=30, mutation_rate=0.5, hm_size=5, hmcr=0.7, par=0.5, bandwidth=0.2,
                 max_iter=5, debug_mode=False, **evaluator_options):
        if max_sample_nums < 1:
            raise ValueError("Budget must be positive")
        template = parse_program(str(evaluation.template_program))
        if template is None or len(template.functions) != 1:
            raise ValueError("HSEvo requires exactly one task entrypoint")
        self._template_program = template
        self._function = template.functions[0]
        self._func_name = self._function.name
        self._problem_desc = evaluation.task_description
        self._func_desc = self._function.docstring or ''
        self._llm = llm
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_options)
        self._profiler = profiler
        self._max_sample_nums = max_sample_nums
        self._pop_size, self._init_pop_size, self._mutation_rate = pop_size, init_pop_size, mutation_rate
        self._hm_size, self._hmcr, self._par = hm_size, hmcr, par
        self._bandwidth, self._max_iter = bandwidth, max_iter
        self._debug_mode = debug_mode
        module = type(evaluation).__module__
        knowledge = PROMPTS / (module.split('.')[-2] if '.' in module else '') / 'external_knowledge.txt'
        self._external_knowledge = knowledge.read_text() if knowledge.is_file() else ''
        self._tot_sample_nums = 0
        # Search state, saved at completed boundaries.
        self._phase = 'seed'
        self._iteration = 0
        self._init_index = 0
        self._init_population = []
        self._population = []
        self._elitist = None
        self._str_flash_memory = {'analyze': '', 'exp': ''}
        self._str_comprehensive_memory = self._external_knowledge
        self._lst_good_reflection = []
        self._lst_bad_reflection = []
        init_observability(self)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    def _has_budget(self):
        return not is_search_aborted(self) and self._tot_sample_nums < self._max_sample_nums

    def run(self):
        status, payload = 'finished', {}
        try:
            save_checkpoint(self, 'hsevo', 'ready')
            if self._phase == 'seed' and self._has_budget():
                self._evaluate_seed()
                save_checkpoint(self, 'hsevo', 'initialization')
            if self._phase == 'init':
                self._initialize_population()
            while self._phase == 'evolve' and self._has_budget():
                self._evolve_generation()
                save_checkpoint(self, 'hsevo', 'generation')
        except Terminated as exc:
            payload = dict(search_complete=True, termination=str(exc))
            log_event(self, event='search_terminated', status='terminated', reason=str(exc),
                      sample_order=self._tot_sample_nums)
        except KeyboardInterrupt:
            status = 'interrupted'
        except Exception as exc:
            status = 'aborted'
            self._search_aborted = True
            log_error(self, 'run', exc, method='hsevo', counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            finish_profiler(self, status=status, **payload)
            close_llm(self._llm)

    # ----- generation and evaluation -----

    def _ask(self, prompt, stage, operator):
        try:
            result = generate(self._llm, prompt)
        except Exception as exc:
            record_sample_failure(self, exc, stage=stage, operator=operator, prompt=prompt,
                                  counts_budget=False)
            return None
        reset_sample_failures(self)
        log_llm_call(self, llm=self._llm, transport_calls=result['calls'], method='hsevo', stage=stage,
                     operator=operator, sample_order=self._tot_sample_nums + 1, iteration=self._iteration)
        return result['content'] or ''

    def _generate(self, prompt, operator, parents=()):
        """One generator response turned into an evaluated individual; None if the call failed."""
        if not self._has_budget():
            return None
        started = time.monotonic()
        response = self._ask(prompt, 'generate', operator)
        if response is None:
            return None
        return self._evaluate(extract_code_from_generator(response), operator, parents,
                              time.monotonic() - started)

    def _evaluate(self, code, operator, parents=(), sample_time=0.0):
        """One upstream function evaluation; an unparsed response also spends it."""
        self._tot_sample_nums += 1
        individual = Individual(code=code, operator=operator)
        function = code_to_function(code, self._func_name) if code is not None else None
        program = replace_function(function, self._template_program) if function is not None else None
        parent_ids = [p.candidate_id for p in parents]
        if program is None:
            log_event(self, event='sample_rejected', status='parse_failed', operator=operator,
                      parent_ids=parent_ids, sample_order=self._tot_sample_nums, counts_budget=True)
            return individual
        score, seconds = self._evaluator.evaluate_program_record_time(program)
        valid = score is not None and math.isfinite(float(score))
        function.score = float(score) if valid else None
        function.operator = function.algorithm = operator
        function.sample_time, function.evaluate_time = sample_time, seconds
        function.parent_id = parent_ids[0] if parent_ids else None
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program), budget_used=self._tot_sample_nums)
        individual.candidate_id = self._profiler._num_samples if self._profiler else self._tot_sample_nums
        if valid:
            individual.obj, individual.exec_success = float(score), True
        log_event(self, event='sample_registered', status='valid' if valid else 'invalid', operator=operator,
                  candidate_id=individual.candidate_id, parent_ids=parent_ids, score=function.score,
                  sample_order=self._tot_sample_nums, counts_budget=True)
        return individual

    def _update_iter(self):
        population = self._population
        if population:
            objs = [individual.obj for individual in population]
            best_sample_idx = int(np.argmin(np.array(objs)))
            if self._elitist is None or objs[best_sample_idx] < self._elitist.obj:
                self._elitist = population[best_sample_idx]
        self._iteration += 1

    # ----- initialization -----

    def _evaluate_seed(self):
        code = GLOBAL_IMPORTS + rename(str(self._function), self._func_name, self._func_name + '_v2')
        seed = self._evaluate(code, 'seed')
        if not seed.exec_success:
            raise RuntimeError("Seed function is invalid.")
        self._population = [seed]
        self._update_iter()
        self._phase = 'init'

    def _generator(self, seed):
        return text('user_generator').format(seed=seed, func_name=self._func_name,
                                             problem_desc=self._problem_desc, func_desc=self._func_desc)

    def _signature(self, version):
        function = self._function
        returns = f' -> {function.return_type}' if function.return_type else ''
        return f'def {self._func_name}_v{version}({function.args}){returns}:'

    def _initialize_population(self):
        seed_prompt = text('seed').format(
            seed_func=rename(str(self._function), self._func_name, self._func_name + '_v1'),
            func_name=self._func_name)
        while self._init_index < self._init_pop_size and self._has_budget():
            scientist = SCIENTISTS[self._init_index % len(SCIENTISTS)]
            # long_term_reflection_str is never set in HSEvo, so it stays empty.
            prompt = messages(text('system_generator').format(seed=scientist),
                              self._generator(scientist) + "\n" + seed_prompt + "\n" + "")
            individual = self._generate(prompt, 'init')
            if individual is not None:
                self._init_population.append(individual)
            self._init_index += 1
            save_checkpoint(self, 'hsevo', 'initialization')
        if self._init_index < self._init_pop_size:
            return
        self._population, self._init_population = self._init_population, []
        self._update_iter()
        self._phase = 'evolve'
        self._log_population()

    # ----- one generation of evolve() -----

    def _evolve_generation(self):
        if all(not individual.exec_success for individual in self._population):
            raise Terminated("All individuals are invalid.")
        population_to_select = (self._population if self._elitist is None
                                or any(i is self._elitist for i in self._population)
                                else [self._elitist] + self._population)
        selected_population, tied = self._random_select(population_to_select)
        if selected_population is None:
            raise Terminated("Selection failed.")
        log_event(self, event='generation_start', status='scheduled', iteration=self._iteration,
                  sample_order=self._tot_sample_nums, selected=[i.candidate_id for i in selected_population])
        self._flash_reflection(selected_population, tied)
        self._comprehensive_reflection()
        previous_elitist = self._elitist

        self._population = self._crossover(selected_population, tied)
        self._update_iter()
        self._population.extend(self._mutate())
        self._update_iter()

        if previous_elitist is not self._elitist:
            self._lst_good_reflection.append(self._str_flash_memory["exp"])
        else:
            self._lst_bad_reflection.append(self._str_flash_memory["exp"])

        for _ in range(HS_TRIES):
            individual_hs = self._harmony_search()
            if individual_hs is not None:
                self._population.append(individual_hs)
                break
        self._update_iter()
        self._log_population()

    def _random_select(self, population):
        """Random selection, select individuals with equal probability."""
        population = [individual for individual in population if individual.exec_success]
        if len(population) < 2:
            return None, False
        # Upstream ends the search when every score is tied; then tied pairs are allowed, as in ReEvo.
        tied = len({individual.obj for individual in population}) < 2
        if tied:
            log_event(self, event='selection_fallback', status='all_scores_tied', iteration=self._iteration,
                      sample_order=self._tot_sample_nums, pool_size=len(population), score=population[0].obj)
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self._pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical;
            # otherwise, add them to the selected population
            if parents[0].obj != parents[1].obj or tied:
                selected_population.extend(parents)
            if trial > 1000:
                return None, False
        return selected_population, tied

    def _flash_reflection(self, population, tied):
        lst_str_method = []
        seen_elements = set()
        sorted_population = sorted(population, key=lambda x: x.obj, reverse=False)
        for idx, individual in enumerate(sorted_population):
            suffix = "th" if 11 <= idx + 1 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get((idx + 1) % 10, "th")
            str_idx_method = f"[Heuristics {idx + 1}]" if tied else f"[Heuristics {idx + 1}{suffix}]"
            temp_str = str_idx_method + "\n" + individual.code + "\n"
            if temp_str not in seen_elements:
                seen_elements.add(temp_str)
                lst_str_method.append(temp_str)
        user = text('user_flash_reflection_tie' if tied else 'user_flash_reflection').format(
            problem_desc=self._problem_desc, lst_method="\n".join(lst_str_method),
            schema_reflection={"analyze": "str", "exp": "str"})
        flash_reflection_res = self._ask(messages(text('system_reflector'), user),
                                         'flash_reflection', 'reflection')
        if flash_reflection_res is None:
            return  # A failed call keeps the previous memory.
        analyze_start = flash_reflection_res.find("**Analysis:**") + len("**Analysis:**")
        exp_start = flash_reflection_res.find("**Experience:**")
        analysis_text = flash_reflection_res[analyze_start:exp_start].strip()
        experience_text = flash_reflection_res[exp_start + len("**Experience:**"):].strip()
        self._str_flash_memory = {"analyze": analysis_text, "exp": experience_text}

    def _comprehensive_reflection(self):
        good_reflection = '\n\n'.join(self._lst_good_reflection) if len(self._lst_good_reflection) > 0 else "None"
        bad_reflection = '\n\n'.join(self._lst_bad_reflection) if len(self._lst_bad_reflection) > 0 else "None"
        user = text('user_comprehensive_reflection').format(
            bad_reflection=bad_reflection, good_reflection=good_reflection,
            curr_reflection=self._str_flash_memory["exp"])
        response = self._ask(messages(text('system_reflector'), user), 'comprehensive_reflection', 'reflection')
        if response is not None:
            self._str_comprehensive_memory = self._external_knowledge + '\n' + response
            log_event(self, event='reflection_update', status='updated', operator='comprehensive_reflection',
                      sample_order=self._tot_sample_nums)

    def _crossover(self, population, tied):
        crossed_population = []
        for i in range(0, len(population), 2):
            # Select two individuals
            if population[i].obj < population[i + 1].obj:
                parent_1, parent_2 = population[i], population[i + 1]
            else:
                parent_1, parent_2 = population[i + 1], population[i]
            user = text('crossover_tie' if tied else 'crossover').format(
                user_generator=self._generator(SCIENTISTS[0]),
                func_signature_m1=self._signature(0), func_signature_m2=self._signature(1),
                code_method1=filter_code(parent_1.code), code_method2=filter_code(parent_2.code),
                analyze=self._str_flash_memory["analyze"], exp=self._str_comprehensive_memory,
                func_name=self._func_name)
            prompt = messages(text('system_generator').format(seed=SCIENTISTS[0]), user)
            individual = self._generate(prompt, 'crossover', (parent_1, parent_2))
            if individual is not None:
                crossed_population.append(individual)
        return crossed_population

    def _mutate(self):
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        user = text('mutation').format(
            user_generator=self._generator(SCIENTISTS[0]), reflection=self._str_comprehensive_memory,
            func_signature1=self._signature(1), elitist_code=filter_code(self._elitist.code),
            func_name=self._func_name)
        prompt = messages(text('system_generator').format(seed=SCIENTISTS[0]), user)
        population = []
        for _ in range(int(self._pop_size * self._mutation_rate)):
            individual = self._generate(prompt, 'mutation', (self._elitist,))
            if individual is not None:
                population.append(individual)
        return population

    # ----- harmony search -----

    def _sel_individual_hs(self):
        candidate_hs = [individual for individual in self._population if individual.tryHS is False]
        if not candidate_hs:
            return None
        best_candidate_id = int(np.argmin(np.array([individual.obj for individual in candidate_hs])))
        # Upstream indexes the population with the candidate list's index; retries often reuse a code.
        self._population[best_candidate_id].tryHS = True
        return self._population[best_candidate_id]

    def _initialize_harmony_memory(self, bounds):
        problem_size = len(bounds)
        harmony_memory = np.zeros((self._hm_size, problem_size))
        for i in range(problem_size):
            lower_bound, upper_bound = bounds[i]
            harmony_memory[:, i] = np.random.uniform(lower_bound, upper_bound, self._hm_size)
        return harmony_memory

    def _create_population_hs(self, str_code, parameter_ranges, harmony_memory, source):
        str_create_pop = []
        for i in range(len(harmony_memory)):
            tmp_str = str_code
            for j in range(len(list(parameter_ranges))):
                tmp_str = tmp_str.replace(('{' + list(parameter_ranges)[j] + '}'), str(harmony_memory[i][j]))
                if tmp_str == str_code:
                    return None
            str_create_pop.append("```python\n" + tmp_str + "\n```")
        population_hs = []
        for response in str_create_pop:
            if not self._has_budget():
                break
            population_hs.append(self._evaluate(extract_code_from_generator(response), 'harmony_search',
                                                (source,)))
        return population_hs

    def _create_new_harmony(self, harmony_memory, bounds):
        new_harmony = np.zeros((harmony_memory.shape[1],))
        for i in range(harmony_memory.shape[1]):
            if np.random.rand() < self._hmcr:
                new_harmony[i] = harmony_memory[np.random.randint(0, harmony_memory.shape[0]), i]
                if np.random.rand() < self._par:
                    adjustment = np.random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0]) * self._bandwidth
                    new_harmony[i] += adjustment
            else:
                new_harmony[i] = np.random.uniform(bounds[i][0], bounds[i][1])
        return new_harmony

    def _update_harmony_memory(self, population_hs, harmony_memory, new_harmony, func_block,
                               parameter_ranges, source):
        objs = [individual.obj for individual in population_hs]
        worst_index = int(np.argmax(np.array(objs)))
        created = self._create_population_hs(func_block, parameter_ranges, [new_harmony.tolist()], source)
        if not created:
            return population_hs, harmony_memory
        new_individual = created[0]
        if new_individual.obj < population_hs[worst_index].obj:
            population_hs[worst_index] = new_individual
            harmony_memory[worst_index] = new_harmony
        return population_hs, harmony_memory

    def _harmony_search(self):
        if not self._has_budget():
            return None
        source = self._sel_individual_hs()
        if source is None:
            return None
        user = text('harmony_search').format(code_extract=source.code)
        response = self._ask(messages(text('system_harmony_search'), user), 'harmony_search', 'harmony_search')
        if response is None:
            return None
        parameter_ranges, func_block = extract_to_hs(response)
        if parameter_ranges is None or func_block is None:
            log_event(self, event='harmony_search_rejected', status='parse_failed',
                      parent_id=source.candidate_id, sample_order=self._tot_sample_nums)
            return None
        bounds = [value for value in parameter_ranges.values()]

        harmony_memory = self._initialize_harmony_memory(bounds)
        population_hs = self._create_population_hs(func_block, parameter_ranges, harmony_memory, source)
        if population_hs is None:
            log_event(self, event='harmony_search_rejected', status='parameters_not_in_signature',
                      parent_id=source.candidate_id, sample_order=self._tot_sample_nums)
            return None
        if not population_hs:
            return None  # The budget ran out.
        if not any(individual.exec_success for individual in population_hs):
            # Upstream refunds these evaluations; the shared protocol counts every evaluated program.
            log_event(self, event='harmony_search_rejected', status='all_invalid',
                      parent_id=source.candidate_id, sample_order=self._tot_sample_nums)
            return None
        init_objs = [individual.obj for individual in population_hs if individual.exec_success]
        for _ in range(self._max_iter):
            if not self._has_budget():
                break
            new_harmony = self._create_new_harmony(harmony_memory, bounds)
            population_hs, harmony_memory = self._update_harmony_memory(
                population_hs, harmony_memory, new_harmony, func_block, parameter_ranges, source)
        best = population_hs[int(np.argmin(np.array([individual.obj for individual in population_hs])))]
        best.tryHS = True
        log_event(self, event='harmony_search', status='completed', parent_id=source.candidate_id,
                  parameters=list(parameter_ranges), init_best=min(init_objs), hs_best=best.obj,
                  candidate_id=best.candidate_id, sample_order=self._tot_sample_nums)
        return best

    def _log_population(self):
        log_state(self, phase='population', method='hsevo', iteration=self._iteration,
                  population=[[i.candidate_id, i.operator, i.obj if i.exec_success else None]
                              for i in self._population],
                  elitist=self._elitist.candidate_id if self._elitist else None,
                  elite_score=self._elitist.obj if self._elitist else None, sample_count=self._tot_sample_nums)

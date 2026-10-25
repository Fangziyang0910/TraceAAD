"""Clade-AHD's search under TraceAAD's generation budget and task evaluator."""

import copy
import math
import random
import time

import numpy as np

from core import SecureEvaluator
from core.code import function_from_source, parse_program, replace_function
from core.llm import generate
from baselines.mcts_ahd.prompt import MAPrompt
from baselines.shared.checkpoint import save_checkpoint
from baselines.shared.observability import (
    close_llm, finish_profiler, init_observability, is_search_aborted, log_error,
    log_event, log_llm_call, log_state, record_sample_failure, reset_sample_failures,
)
from .upstream.bayesian_node import BayesianNode
from .upstream.evolution import Evolution
from .upstream.mcts import MCTS
from .upstream.pop_greedy import population_management, population_management_s1
from .upstream.prob_rank import parent_selection, parent_selection_e1


class SearchStopped(Exception):
    """Budget exhausted or the shared transport/scheduler has stopped the run."""


class CladeAHD:
    LEAF_OPERATORS = ('e2', 'm1', 'm1', 'm2', 'm2', 's1')

    def __init__(self, llm, evaluation, profiler=None, max_sample_nums=1000, *,
                 init_size=4, pop_size=10, num_pseudo_evals=10, backprop_lambda=0.8,
                 cool_down_enabled=True, cool_down_beta=1.0, pruning_enabled=True,
                 pruning_threshold=0.1, pruning_min_visits=10, debug_mode=False,
                 **evaluator_options):
        for name, value in (('max_sample_nums', max_sample_nums), ('init_size', init_size),
                            ('pop_size', pop_size), ('pruning_min_visits', pruning_min_visits)):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not isinstance(num_pseudo_evals, int) or num_pseudo_evals < 0:
            raise ValueError('num_pseudo_evals must be a nonnegative integer')
        if not 0 <= backprop_lambda <= 1 or not 0 <= pruning_threshold <= 1:
            raise ValueError('backprop_lambda and pruning_threshold must be in [0, 1]')
        if not math.isfinite(cool_down_beta) or cool_down_beta < 0:
            raise ValueError('cool_down_beta must be finite and nonnegative')
        self._template_program = parse_program(str(evaluation.template_program))
        if self._template_program is None or len(self._template_program.functions) != 1:
            raise ValueError('Clade-AHD requires exactly one task entrypoint')
        template = self._template_program.functions[0]
        self._func_name = template.name
        self._llm, self._profiler = llm, profiler
        self._evaluator = SecureEvaluator(evaluation, debug_mode=debug_mode, **evaluator_options)
        self._max_sample_nums, self._init_size, self._pop_size = max_sample_nums, init_size, pop_size
        self._debug_mode = debug_mode
        self._evolution = Evolution()
        # Reuse only the task-contract adapter; Clade's own prompt builders and parser run below.
        vars(self._evolution).update(vars(MAPrompt.adapter(evaluation.task_description, template)))
        self._evolution.interface_llm = self
        self._evolution.debug_mode = False
        self._mcts = MCTS('Root', optimization_type='min', max_depth=10,
                          num_pseudo=num_pseudo_evals, backprop_lambda=backprop_lambda,
                          cool_down_enabled=cool_down_enabled, cool_down_beta=cool_down_beta,
                          pruning_enabled=pruning_enabled, pruning_threshold=pruning_threshold,
                          pruning_min_visits=pruning_min_visits)
        self._tot_sample_nums = self._evaluation_count = self._upstream_evals = self._iteration = 0
        self._phase = 'init'
        self._population = []
        self._init_population = []
        self._tree_state = {}
        self._operator = None
        init_observability(self)
        if profiler is not None:
            profiler.record_parameters(llm, evaluation, self)

    def _has_budget(self):
        return not is_search_aborted(self) and self._tot_sample_nums < self._max_sample_nums

    def get_response(self, prompt, mode=1):
        """Transport used by the unchanged upstream parser (mode=0 is thought alignment)."""
        stage = 'generate' if mode else 'thought_alignment'
        while not is_search_aborted(self):
            if mode and not self._has_budget():
                raise SearchStopped()
            try:
                result = generate(self._llm, prompt)
            except Exception as exc:
                record_sample_failure(self, exc, stage=stage, operator=self._operator,
                                      prompt=prompt, counts_budget=False)
                continue
            reset_sample_failures(self)
            if mode:
                self._tot_sample_nums += 1
            log_llm_call(self, llm=self._llm, transport_calls=result['calls'], method='clade_ahd',
                         stage=stage, operator=self._operator, sample_order=self._tot_sample_nums,
                         counts_budget=bool(mode))
            if mode:
                log_event(self, event='generation_returned', operator=self._operator,
                          sample_order=self._tot_sample_nums, counts_budget=True)
            return result['content'] or ''
        raise SearchStopped()

    def _draw(self, population, operator, father):
        self._operator = operator
        if operator == 'i1':
            parents = []
            prompt = self._evolution.get_prompt_i1()
        elif operator == 'e1':
            parents = parent_selection_e1(population, min(random.randint(2, 5), len(population)))
            prompt = self._evolution.get_prompt_e1(parents)
        elif operator == 'e2':
            others = [p for p in population if p != father]
            if not others:
                return None
            parents = parent_selection(others, 1) + [father]
            prompt = self._evolution.get_prompt_e2(parents)
        else:
            parents = population if operator == 's1' else [father]
            prompt = getattr(self._evolution, 'get_prompt_' + operator)(
                parents if operator == 's1' else father)
        started = time.monotonic()
        code, thought = self._evolution._get_alg(prompt)
        algorithm = self._evolution.post_thought(code, thought)
        return {'code': code, 'thought': thought, 'algorithm': algorithm, 'objective': None,
                'other_inf': None, 'parent_ids': [p.get('candidate_id') for p in parents],
                'sample_time': time.monotonic() - started, 'operator': operator}

    def _offspring(self, population, operator, father=None, *, initial=False, batch=False):
        """InterfaceEC's duplicate retry and three evaluation attempts, with bounded generation."""
        attempts = 0
        while self._has_budget() and (initial or attempts < 3):
            try:
                child = self._draw(population, operator, father)
                if child is None:
                    return None
                if any(p['code'] == child['code'] for p in population) and self._has_budget():
                    child = self._draw(population, operator, father)
            except ValueError as exc:
                log_event(self, event='sample_rejected', status='parse_failed', error=str(exc),
                          operator=operator, sample_order=self._tot_sample_nums, counts_budget=True)
                continue
            attempts += 1
            if not batch:
                self._upstream_evals += 1
            timed_out = self._evaluate(child)
            if not initial and timed_out:
                return None
            if child['objective'] is None:
                continue
            if initial:
                duplicate = any(p['objective'] == child['objective'] for p in population)
            else:
                duplicate = any(p['code'] == child['code'] for p in population)
            if not duplicate:
                return child
        return None

    def _evaluate(self, child):
        function = function_from_source(child['code'], self._func_name)
        program = replace_function(function, self._template_program) if function is not None else None
        if program is None:
            log_event(self, event='sample_rejected', status='invalid_program', operator=child['operator'],
                      sample_order=self._tot_sample_nums, counts_budget=True)
            return False
        outcome, seconds = self._evaluator.evaluate_program_record_time_with_details(program)
        self._evaluation_count += 1
        score = outcome.result
        valid = score is not None and math.isfinite(float(score))
        function.score = float(score) if valid else None
        function.algorithm, function.operator = child['algorithm'], child['operator']
        function.sample_time, function.evaluate_time = child['sample_time'], seconds
        function.parent_id = child['parent_ids'][0] if child['parent_ids'] else None
        if self._profiler is not None:
            self._profiler.register_function(function, program=str(program), budget_used=self._tot_sample_nums)
        child['candidate_id'] = self._profiler._num_samples if self._profiler else self._tot_sample_nums
        if valid:
            child['objective'] = float(np.round(float(score), 5))
        return outcome.failure_kind == 'timeout'

    def _attach(self, parent, child, *, initial=False, batch=False):
        if not initial and child['operator'] == 'e1' and any(
                c['objective'] == child['objective'] for c in self._mcts.root.children_info):
            return
        node = BayesianNode(parent, child['code'], child['algorithm'], parent.depth + 1,
                            operator=child['operator'], raw_info=child)
        node.objective = child['objective']
        node.Q = node.reward = -node.objective
        node.visits = int(initial or batch)
        normalizer = self._mcts.normalizer
        normalizer.update(node.objective)
        score = normalizer.normalize(node.objective)
        # Upstream applies this observation twice to the child; preserve its actual belief update.
        node.update_belief(score)
        self._mcts.backpropagate(node, score)
        if initial or child['operator'] == 'e1':
            node.subtree.append(node)
        parent.add_child(node)
        parent.children_info.append(child)
        if initial:
            self._init_population.append(child)
            self._population = population_management(self._init_population, self._pop_size)
        else:
            if not any(p['objective'] == child['objective'] for p in self._population):
                self._population.append(child)
            self._population = population_management(self._population, self._pop_size)
        if batch:
            self._upstream_evals += 1
        log_event(self, event='tree_expand', operator=child['operator'],
                  parent_id=parent.raw_info.get('candidate_id'), candidate_id=child['candidate_id'],
                  alpha=node.alpha, beta=node.beta, visits=node.visits, depth=node.depth,
                  sample_order=self._tot_sample_nums)

    def _references(self, node, operator):
        if operator == 'e1':
            return [random.choices(child.subtree, k=1)[0].raw_info for child in self._mcts.root.children]
        if operator == 's1':
            path = []
            while node.parent is not None:
                path.append(node.raw_info)
                node = node.parent
            return population_management_s1(path, len(path))
        return self._population

    def _expand(self, node, operator, *, batch=False):
        population = self._references(node, operator)
        if operator == 's1' and len(population) <= 1:
            return None
        return self._offspring(population, operator, node.raw_info, batch=batch)

    def _round(self):
        node = self._mcts.root
        progress = self._tot_sample_nums / self._max_sample_nums
        while node.children and node.depth < self._mcts.max_depth and self._has_budget():
            if int(node.visits ** self._mcts.alpha) > len(node.children):
                child = self._expand(node, 'e1' if node is self._mcts.root else 'e2')
                if child is not None:
                    self._attach(node, child)
            node = node.children[self._mcts.select_clade_thompson(node.children, progress)]
        # All leaf offspring see the batch-start tree and elite set, as in upstream parallel_operators.
        offspring = []
        try:
            for operator in self.LEAF_OPERATORS:
                if not self._has_budget():
                    break
                child = self._expand(node, operator, batch=True)
                if child is not None:
                    offspring.append(child)
        finally:
            for child in offspring:
                self._attach(node, child, batch=True)
        if self._upstream_evals % 5 == 0 and self._population:
            best = self._population[0]['objective']
            self._mcts.normalizer.update(best)
            self._mcts.prune_tree(self._mcts.normalizer.normalize(best))
        self._iteration += 1

    def _snapshot_tree(self):
        nodes = self._mcts.get_all_nodes()
        indices = {id(node): i for i, node in enumerate(nodes)}
        records = []
        for node in nodes:
            record = {k: v for k, v in vars(node).items() if k not in ('parent', 'children', 'subtree')}
            record.update(parent=indices[id(node.parent)] if node.parent is not None else None,
                          children=[indices[id(c)] for c in node.children],
                          subtree=[indices[id(c)] for c in node.subtree])
            records.append(record)
        return copy.deepcopy({'nodes': records, 'normalizer': vars(self._mcts.normalizer),
                              'rank_list': self._mcts.rank_list})

    def _restore_tree(self):
        records = self._tree_state['nodes']
        nodes = [BayesianNode(None, record['code']) for record in records]
        for node, record in zip(nodes, records):
            vars(node).update({k: v for k, v in record.items() if k not in ('parent', 'children', 'subtree')})
            node.parent = nodes[record['parent']] if record['parent'] is not None else None
            node.children = [nodes[i] for i in record['children']]
            node.subtree = [nodes[i] for i in record['subtree']]
        self._mcts.root = nodes[0]
        vars(self._mcts.normalizer).update(self._tree_state['normalizer'])
        self._mcts.rank_list = self._tree_state['rank_list']

    def _checkpoint(self, boundary):
        self._tree_state = self._snapshot_tree()
        log_state(self, method='clade_ahd', phase=self._phase, sample_order=self._tot_sample_nums,
                  search_evaluations=self._evaluation_count, upstream_evals=self._upstream_evals,
                  iteration=self._iteration, nodes=len(self._tree_state['nodes']))
        save_checkpoint(self, 'clade_ahd', boundary)

    def run(self):
        status = 'finished'
        try:
            self._checkpoint('ready')
            while self._phase == 'init' and self._has_budget():
                if len(self._init_population) >= self._init_size:
                    self._phase = 'evolve'
                    break
                child = self._offspring(self._init_population,
                                         'e1' if self._init_population else 'i1', initial=True)
                if child is not None:
                    self._attach(self._mcts.root, child, initial=True)
                self._checkpoint('initialization')
            while self._phase == 'evolve' and self._has_budget():
                self._round()
                self._checkpoint('round')
        except SearchStopped:
            self._checkpoint('budget_or_transport_stop')
        except KeyboardInterrupt:
            status = 'interrupted'
        except Exception as exc:
            self._search_aborted = True
            log_error(self, 'run', exc, method='clade_ahd', counts_budget=False)
            if self._debug_mode:
                raise
        finally:
            finish_profiler(self, status=status, search_evaluations=self._evaluation_count,
                            upstream_evals=self._upstream_evals)
            close_llm(self._llm)

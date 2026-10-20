"""Execution and score measurement shared by seeded optimization tasks."""

import numpy as np

from core import Evaluation
from .seeded_data import SEED, generate_dataset


class FixedEvaluation(Evaluation):
    """Task modules supply their generator, template, solver and fixed settings."""

    executes_source = True
    INTERFACE = None

    def __init__(self, split='train', timeout_seconds=60, limit=None, settings=None, **kwargs):
        super().__init__(template_program=self.TEMPLATE, task_description=self.DESCRIPTION,
                         timeout_seconds=timeout_seconds, **kwargs)
        self.task = self.DATASET.TASK
        if self.INTERFACE:
            self.interface = self.INTERFACE
        self.seed = SEED
        self.data_distribution = self.DATASET.DISTRIBUTION
        rows, self._instances = generate_dataset(self.DATASET, split, limit)
        self.split = rows[0]['split']
        self._rows = rows
        self._datasets = [dict(row) for row in rows]
        self.n_instance = len(rows)
        self.problem_size = max(r['scale'] for r in rows)
        self.score_metric = 'raw_objective'
        self.score_meaning = f'the mean {self.OBJECTIVE_NAME} over all instances (lower is better)'
        settings = dict(self.DEFAULT_SETTINGS if settings is None else settings)
        if set(settings) != set(self.DEFAULT_SETTINGS) or any(type(v) is not int or v < 1 for v in settings.values()):
            raise ValueError('outer-solver settings must be the task iteration counts as positive integers')
        self.outer_settings = settings

    def evaluate_program(self, program_str, callable_func):
        values = []
        for original in self._instances:
            namespace = {}
            exec(program_str, namespace)
            data = {key: value.copy() for key, value in original.items()}
            _, objective = self.SOLVER(data, namespace[self.FUNCTION_NAME], **self.outer_settings)
            values.append(float(objective))
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise ValueError('outer solver produced a non-finite objective')
        return result

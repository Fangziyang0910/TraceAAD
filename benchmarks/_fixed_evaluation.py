"""Execution and score measurement shared by seeded fixed-framework tasks."""

import numpy as np

from core import Evaluation
from core.evaluate import InvalidEvaluationResult
from ._seeded_data import PROTOCOL, SEED, generate_dataset


def scores(value, length, name):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationResult(f"{name} must return numeric candidate scores") from exc
    if value.shape != (length,) or not np.isfinite(value).all():
        raise InvalidEvaluationResult(f"{name} must return {length} finite scores in candidate order")
    return value


class FixedEvaluation(Evaluation):
    """Task modules supply their generator, template, solver and fixed settings."""

    MAXIMIZE = False
    executes_source = True

    def __init__(self, split='train', timeout_seconds=60, limit=None, settings=None, **kwargs):
        super().__init__(template_program=self.TEMPLATE, task_description=self.DESCRIPTION,
                         timeout_seconds=timeout_seconds, **kwargs)
        self.task, self.suite_protocol = self.DATASET.TASK, PROTOCOL
        self.seed = SEED
        self.data_distribution = self.DATASET.DISTRIBUTION
        rows, self._instances = generate_dataset(self.DATASET, split, limit)
        self.split = rows[0]['split']
        self._rows = rows
        self._datasets = [dict(row) for row in rows]
        self.n_instance = len(rows)
        self.problem_size = max(r['scale'] for r in rows)
        self.instance_description = self.DATASET.describe(self.split, len(rows))
        self.score_meaning = 'the mean relative deviation from the generated reference, in percent (lower is better)'
        settings = dict(self.DEFAULT_SETTINGS if settings is None else settings)
        if set(settings) != set(self.DEFAULT_SETTINGS) or any(type(v) is not int or v < 1 for v in settings.values()):
            raise ValueError('outer-solver settings must be the task iteration counts as positive integers')
        self.outer_settings = settings
        direction = '(reference - profit)' if self.MAXIMIZE else '(objective - reference)'
        references = ', '.join(sorted({r['reference_kind'] for r in rows}))
        self.design_notes = (
            f'Evaluation uses {self.instance_description}. '
            'Inputs and references are generated once at evaluator initialization with fixed split-specific seeds. '
            'Dataset generation and reference computation are not timed. '
            f'The fixed outer-solver settings are {settings}. The target is called repeatedly inside this solver; '
            'its return is consumed exactly as specified in the template. The program is executed afresh '
            'for each instance, so candidate globals are reset between instances. Inputs passed to the '
            'target are copies; changing them does not change solver state. Scores are averaged equally '
            f'over instances: 100 * {direction} / abs(reference). Lower is better. '
            f'References for this split are: {references}. They are hidden from the candidate and are '
            'not guaranteed integer optima. Negative deviations mean the reference was improved. '
            'Score computation does not clip negative deviations. Invalid return '
            'contracts or errors fail the evaluation; score quality is always computed from the final valid solution.')
        self.task_description += '\n\n' + self.design_notes

    def evaluate_program(self, program_str, callable_func):
        values = []
        for row, original in zip(self._rows, self._instances):
            namespace = {}
            exec(program_str, namespace)
            data = {key: value.copy() for key, value in original.items()}
            _, objective = self.SOLVER(data, namespace[self.FUNCTION_NAME], **self.outer_settings)
            reference = row['reference']
            delta = reference - objective if self.MAXIMIZE else objective - reference
            values.append(100.0 * delta / abs(reference))
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise ValueError('outer solver produced a non-finite objective')
        return result

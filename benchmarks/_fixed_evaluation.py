"""Execution and score measurement shared by prepared fixed-framework tasks."""

from pathlib import Path

import numpy as np

from core import Evaluation
from core.evaluate import InvalidEvaluationResult
from ._prepared_data import PROTOCOL, load_instance, read_records


def scores(value, length, name):
    try:
        value = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise InvalidEvaluationResult(f"{name} must return numeric candidate scores") from exc
    if value.shape != (length,) or not np.isfinite(value).all():
        raise InvalidEvaluationResult(f"{name} must return {length} finite scores in candidate order")
    return value


class FixedEvaluation(Evaluation):
    """Task modules supply their dataset, template, solver and fixed settings."""

    MAXIMIZE = False

    def __init__(self, split='train', timeout_seconds=60, data_root=None, limit=None, settings=None, **kwargs):
        super().__init__(template_program=self.TEMPLATE, task_description=self.DESCRIPTION,
                         timeout_seconds=timeout_seconds, **kwargs)
        self.task, self.suite_protocol = self.DATASET.TASK, PROTOCOL
        self._data_root = Path(data_root) if data_root is not None else self.DATASET.DATA_ROOT
        rows = read_records(self._data_root, split, task=self.task)
        self.split = rows[0]['split']
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError('limit must be a positive integer')
            rows = rows[:limit]
        self._rows = rows
        self._datasets = [{k: r[k] for k in ('id', 'group', 'scale', 'dimensions', 'content', 'sha256', 'reference', 'reference_kind')}
                          for r in rows]
        self._instances = [load_instance(row, self._data_root) for row in rows]
        self.n_instance = len(rows)
        self.problem_size = max(r['scale'] for r in rows)
        self.instance_description = self.DATASET.describe(rows)
        self.score_meaning = 'the mean relative deviation from the stored reference, in percent (lower is better)'
        settings = dict(self.DEFAULT_SETTINGS if settings is None else settings)
        if set(settings) != set(self.DEFAULT_SETTINGS) or any(type(v) is not int or v < 1 for v in settings.values()):
            raise ValueError('outer-solver settings must be the task iteration counts as positive integers')
        self.outer_settings = settings
        limit_text = f'{timeout_seconds:g} seconds' if timeout_seconds is not None else 'no configured limit'
        direction = '(reference - profit)' if self.MAXIMIZE else '(objective - reference)'
        references = ', '.join(sorted({r['reference_kind'] for r in rows}))
        self.design_notes = (
            f'Evaluation uses {self.instance_description}. The WHOLE dataset evaluation has {limit_text}. '
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

"""Official prompt text with the benchmark's minimized objective converted to utility."""
from types import SimpleNamespace


def prompt_record(program, code=None):
    return SimpleNamespace(code=code if code is not None else program.program,
        combined_score=-program.combined_score, public_metrics=program.public_metrics,
        text_feedback=program.text_feedback, correct=program.correct)


def messages(system, user):
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

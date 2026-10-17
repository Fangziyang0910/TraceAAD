from pathlib import Path
from .sampler import filter_code

_COMMON = Path(__file__).with_name('prompts') / 'common'

def text(name, **values):
    return (_COMMON / (name + '.txt')).read_text().format(**values)


def messages(system, content):
    return [{'role': 'system', 'content': text(system)}, {'role': 'user', 'content': content}]


class ReEvoPrompt:
    @staticmethod
    def generator(task, function):
        return text('user_generator', func_name=function.name, problem_desc=task,
                    func_desc=function.docstring or '')

    @staticmethod
    def signature(function, version):
        returns = f' -> {function.return_type}' if function.return_type else ''
        return f'def {function.name}_v{version}({function.args}){returns}:'

    @staticmethod
    def _worse_better(indivs):
        if len(indivs) != 2 or indivs[0].score == indivs[1].score:
            raise ValueError('ReEvo reflection requires parents with different scores.')
        return sorted(indivs, key=lambda f: f.score, reverse=True)

    @classmethod
    def get_pop_init_prompt(cls, task, function):
        return messages('system_generator', cls.generator(task, function) + '\n' +
                        text('seed', seed_func=str(function), func_name=function.name))

    @classmethod
    def get_short_term_reflection_prompt(cls, task, indivs):
        worse, better = cls._worse_better(indivs)
        return messages('system_reflector', text('user_reflector_st', func_name=better.name,
            problem_desc=task, func_desc=better.docstring or '',
            worse_code=filter_code(str(worse)), better_code=filter_code(str(better))))

    @classmethod
    def get_crossover_prompt(cls, task, reflection, indivs):
        worse, better = cls._worse_better(indivs)
        return messages('system_generator', text('crossover', user_generator=cls.generator(task, better),
            func_signature0=cls.signature(worse, 0), func_signature1=cls.signature(better, 1),
            worse_code=filter_code(str(worse)), better_code=filter_code(str(better)),
            reflection=reflection, func_name=better.name))

    @staticmethod
    def get_long_term_reflection_prompt(task, prior, reflections):
        return messages('system_reflector', text('user_reflector_lt', problem_desc=task,
            prior_reflection=prior, new_reflection='\n'.join(reflections)))

    @classmethod
    def get_elist_mutation_prompt(cls, task, reflection, elite):
        return messages('system_generator', text('mutation', user_generator=cls.generator(task, elite),
            reflection=reflection, func_signature1=cls.signature(elite, 1),
            elitist_code=filter_code(str(elite)), func_name=elite.name))

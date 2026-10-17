"""EoH 47254578 prompt builders; task values are supplied by TraceAAD."""

class EoHPrompt:

    def _func_spec(self) -> str:
        if self._template_kind == 'class':
            verb = "implement the following Python class"
        elif self._template_kind == 'multi_function':
            verb = "implement the following Python functions"
        else:
            verb = "implement the following Python function"
        return (
            f"{verb}:\n"
            f"```python\n{self.template.strip()}\n```\n"
            "Do not give additional explanations."
        )

    def _parent_block(self, parents: list) -> str:
        return "\n".join(
            f"No.{i+1} algorithm and the corresponding code are:\n{p['algorithm']}\n{p['code']}"
            for i, p in enumerate(parents)
        )

    def _build_prompt(self, operator: str, parents=None) -> str:
        spec = self._func_spec()
        if operator == "i1":
            return (
                f"{self.task}\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "e1":
            block = self._parent_block(parents)
            return (
                f"{self.task}\n"
                f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
                "Please help me create a new algorithm that has a totally different form from the given ones.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "e2":
            block = self._parent_block(parents)
            return (
                f"{self.task}\n"
                f"I have {len(parents)} existing algorithms with their codes as follows:\n{block}\n"
                "Please help me create a new algorithm that has a totally different form from the given ones "
                "but can be motivated from them.\n"
                "Firstly, identify the common backbone idea in the provided algorithms. "
                "Secondly, based on the backbone idea describe your new algorithm in one sentence. "
                f"The description must be inside a brace. Thirdly, {spec}"
            )
        if operator == "m1":
            return (
                f"{self.task}\n"
                f"I have one algorithm with its code as follows.\n"
                f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
                "Please assist me in creating a new algorithm that has a different form but can be a "
                "modified version of the algorithm provided.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "m2":
            return (
                f"{self.task}\n"
                f"I have one algorithm with its code as follows.\n"
                f"Algorithm description: {parents['algorithm']}\nCode:\n{parents['code']}\n"
                "Please identify the main algorithm parameters and assist me in creating a new algorithm "
                "that has different parameter settings.\n"
                "First, describe your new algorithm and main steps in one sentence. "
                f"The description must be inside a brace. Next, {spec}"
            )
        if operator == "m3":
            if self._template_kind == 'class':
                keep_str = "keeping the class interface (name, method signatures, inputs, and outputs) unchanged"
            elif self._template_kind == 'multi_function':
                keep_str = "keeping all function names, inputs, and outputs unchanged"
            else:
                keep_str = "keeping the function name, inputs, and outputs unchanged"
            return (
                "First, identify the main components in the code below. "
                "Next, analyze whether any can be overfit to in-distribution instances. "
                "Then, simplify the components to enhance generalization to out-of-distribution instances. "
                f"Finally, provide the revised code {keep_str}.\n"
                f"{parents['code']}\nDo not give additional explanations."
            )
        raise ValueError(f"Unknown operator: {operator}")

    @classmethod
    def build(cls, task, template, operator, parents=None):
        prompt = cls()
        prompt.task, prompt.template = task, str(template)
        prompt._template_kind = 'function'
        if isinstance(parents, list):
            parents = [{'algorithm': p.algorithm, 'code': str(p)} for p in parents]
        elif parents is not None:
            parents = {'algorithm': parents.algorithm, 'code': str(parents)}
        return prompt._build_prompt(operator, parents)

    @classmethod
    def get_prompt_i1(cls, task, template):
        return cls.build(task, template, 'i1')

    @classmethod
    def get_prompt_e1(cls, task, parents, template):
        return cls.build(task, template, 'e1', parents)

    @classmethod
    def get_prompt_e2(cls, task, parents, template):
        return cls.build(task, template, 'e2', parents)

    @classmethod
    def get_prompt_m1(cls, task, parents, template):
        return cls.build(task, template, 'm1', parents)

    @classmethod
    def get_prompt_m2(cls, task, parents, template):
        return cls.build(task, template, 'm2', parents)

    @classmethod
    def get_prompt_m3(cls, task, parents, template):
        return cls.build(task, template, 'm3', parents)

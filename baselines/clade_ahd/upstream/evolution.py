"""Prompt builders and parser from Mriya0306/Clade-AHD@ffd566b7; transport is supplied locally."""
import re
import time


class Evolution:
    def get_prompt_post(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    def get_prompt_i1(self):

        prompt_content = self.prompt_task + "\n" + "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _validate_individual(self, indiv, index=None):
        """Validate that an individual contains all required fields."""
        if not isinstance(indiv, dict):
            return False
            
        required_fields = ['algorithm', 'code', 'objective']
        for field in required_fields:
            if field not in indiv or indiv[field] is None:
                return False
            if field in ['algorithm', 'code'] and (not isinstance(indiv[field], str) or len(indiv[field].strip()) == 0):
                return False
        
        return True

    def _validate_individuals(self, indivs, operator_name):
        """Validate a list of individuals before using them in prompts."""
        if isinstance(indivs, dict):
            if not self._validate_individual(indivs):
                print(f"Error: Invalid individual for operator {operator_name}")
                return False
            return True
        
        if not isinstance(indivs, list) or len(indivs) == 0:
            print(f"Error: Operator {operator_name} received empty individuals list")
            return False
        
        for i, indiv in enumerate(indivs):
            if not self._validate_individual(indiv, index=i):
                print(f"Error: Individual {i} invalid for operator {operator_name}")
                return False
        
        return True

    def get_prompt_e1(self, indivs):
        # Validate individuals before building prompt
        if not self._validate_individuals(indivs, "e1"):
            raise ValueError("Invalid individuals provided to e1 operator")
        
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self, indivs):
        # Validate individuals before building prompt
        if not self._validate_individuals(indivs, "e2"):
            raise ValueError("Invalid individuals provided to e2 operator")
        
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m1(self, indiv1):
        # Validate individual before building prompt
        if not self._validate_individuals(indiv1, "m1"):
            raise ValueError("Invalid individual provided to m1 operator")
        
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        # Validate individual before building prompt
        if not self._validate_individuals(indiv1, "m2"):
            raise ValueError("Invalid individual provided to m2 operator")
        
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_s1(self, indivs):
        # Validate individuals before building prompt
        if not self._validate_individuals(indivs, "s1"):
            raise ValueError("Invalid individuals provided to s1 operator")
        
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _get_thought(self, prompt_content):

        response = self.interface_llm.get_response(prompt_content, 0)

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content):
        """
        Extract algorithm description and code from LLM response.
        
        Expected response format:
        {Algorithm description here}
        
        ```python (optional)
        import ...
        def function_name(...):
            ...
            return result
        ```
        """

        response = self.interface_llm.get_response(prompt_content)

        # Parse algorithm description (inside braces)
        algorithm = ""
        algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
        if algorithm_match:
            algorithm = algorithm_match.group(1).strip()
        
        # Fallback: extract text before code if no braces found
        if len(algorithm) == 0:
            if '```python' in response:
                algorithm = re.findall(r'^(.*?)```python', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^(.*?)(?=import)', response, re.DOTALL)
            elif 'def ' in response:
                algorithm = re.findall(r'^(.*?)(?=def )', response, re.DOTALL)
            
            if isinstance(algorithm, list) and len(algorithm) > 0:
                algorithm = algorithm[0].strip()
            elif isinstance(algorithm, list):
                algorithm = ""

        # Parse code (handle both markdown and plain formats)
        code = []
        
        # Strategy 0: Clean response - remove algorithm description if it's mixed with code
        # Handle case where algorithm is in {} and code follows
        cleaned_response = response
        brace_pattern = r'\{[^{}]*?\}[\s\n]*'
        if re.search(brace_pattern, response, re.DOTALL):
            # Remove {...} blocks that appear before 'def'
            cleaned_response = re.sub(r'\{[^{}]*?\}[\s\n]*(?=def\s)', '', response, flags=re.DOTALL)
        
        # Strategy 1: Extract from ```python...``` or ```...``` blocks
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", cleaned_response, re.DOTALL | re.IGNORECASE)
        if code_blocks:
            for block in code_blocks:
                block = block.strip()
                # Check if this block contains actual Python code
                if 'def ' in block or 'import ' in block or 'return' in block:
                    code.append(block)
        
        # Strategy 2: Extract plain text code (from import/def to end of function)
        if len(code) == 0:
            # Pattern 1: From import to complete function (greedy match to capture all)
            pattern_import = r"((?:import\s+[^\n]+\n)+\s*def\s+\w+.*?)(?=\n\S|\n*$)"
            import_code = re.findall(pattern_import, cleaned_response, re.DOTALL)
            if import_code:
                code.extend(import_code)
            
            # Pattern 2: Just function definition (from def to next def or end)
            if len(code) == 0:
                pattern_def = r"(def\s+\w+\s*\([^)]*\).*?)(?=\n(?:def\s+|class\s+|\Z))"
                def_code = re.findall(pattern_def, cleaned_response, re.DOTALL)
                if def_code:
                    code.extend(def_code)
        
        # Strategy 3: More aggressive extraction for malformed responses
        if len(code) == 0:
            # Try to find any 'def function_name' and extract until we see unindented text
            pattern_aggressive = r"(def\s+\w+[^\n]*\n(?:\s{2,}[^\n]*\n)*)"
            aggressive_code = re.findall(pattern_aggressive, cleaned_response)
            if aggressive_code:
                code.extend(aggressive_code)

        # Retry mechanism with better error logging
        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0) and n_retry <= 3:
            if self.debug_mode:
                print(f"[Retry {n_retry}/3] Algorithm or code not identified, retrying...")

            time.sleep(1)
            response = self.interface_llm.get_response(prompt_content)

            # Re-parse algorithm
            algorithm = ""
            algorithm_match = re.search(r"\{(.*?)\}", response, re.DOTALL)
            if algorithm_match:
                algorithm = algorithm_match.group(1).strip()
            
            if len(algorithm) == 0:
                if '```python' in response:
                    algorithm = re.findall(r'^(.*?)```python', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^(.*?)(?=import)', response, re.DOTALL)
                elif 'def ' in response:
                    algorithm = re.findall(r'^(.*?)(?=def )', response, re.DOTALL)
                
                if isinstance(algorithm, list) and len(algorithm) > 0:
                    algorithm = algorithm[0].strip()
                elif isinstance(algorithm, list):
                    algorithm = ""

            # Re-parse code with enhanced strategies
            code = []
            
            # Clean response first
            cleaned_retry_response = response
            brace_pattern = r'\{[^{}]*?\}[\s\n]*'
            if re.search(brace_pattern, response, re.DOTALL):
                cleaned_retry_response = re.sub(r'\{[^{}]*?\}[\s\n]*(?=def\s)', '', response, flags=re.DOTALL)
            
            code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", cleaned_retry_response, re.DOTALL | re.IGNORECASE)
            if code_blocks:
                for block in code_blocks:
                    block = block.strip()
                    if 'def ' in block or 'import ' in block or 'return' in block:
                        code.append(block)
            
            if len(code) == 0:
                # Pattern 1: From import to complete function
                pattern_import = r"((?:import\s+[^\n]+\n)+\s*def\s+\w+.*?)(?=\n\S|\n*$)"
                import_code = re.findall(pattern_import, cleaned_retry_response, re.DOTALL)
                if import_code:
                    code.extend(import_code)
                
                # Pattern 2: Just function definition
                if len(code) == 0:
                    pattern_def = r"(def\s+\w+\s*\([^)]*\).*?)(?=\n(?:def\s+|class\s+|\Z))"
                    def_code = re.findall(pattern_def, cleaned_retry_response, re.DOTALL)
                    if def_code:
                        code.extend(def_code)
            
            # Aggressive extraction
            if len(code) == 0:
                pattern_aggressive = r"(def\s+\w+[^\n]*\n(?:\s{2,}[^\n]*\n)*)"
                aggressive_code = re.findall(pattern_aggressive, cleaned_retry_response)
                if aggressive_code:
                    code.extend(aggressive_code)

            n_retry += 1

        # Final validation
        if len(code) == 0:
            if self.debug_mode:
                print(f"Error: Failed to extract code after {n_retry-1} retries")
            raise ValueError("Could not extract valid code from LLM response")
        
        if len(algorithm) == 0:
            algorithm = "No algorithm description provided"

        # Take the longest code block (most likely complete)
        code = max(code, key=len) if isinstance(code, list) else code
        code = code.strip()
        
        # Clean up malformed code: remove trailing }
        # Handle case like "return x\n}\n    return x"
        if '}' in code:
            # Remove lines that only contain } or are after }
            lines = code.split('\n')
            cleaned_lines = []
            found_closing_brace = False
            for line in lines:
                if line.strip() == '}':
                    found_closing_brace = True
                    continue  # Skip the } line
                if not found_closing_brace:
                    cleaned_lines.append(line)
                else:
                    # After }, only keep if it's indented code (not duplicate return at indent 0)
                    if line.strip() and not line.startswith('    ') and line.strip().startswith('return'):
                        continue  # Skip unindented return after }
                    elif line.strip():  # Keep other content
                        cleaned_lines.append(line)
            code = '\n'.join(cleaned_lines)
        
        # Check if code already has return statement - if not, add it
        # Important: We need to carefully check the last line for return
        code_lines = code.split('\n')
        last_significant_line = ""
        for line in reversed(code_lines):
            if line.strip():  # Find last non-empty line
                last_significant_line = line.strip()
                break
        
        # Only add return if it's truly missing
        if 'return' not in last_significant_line:
            # Need to add return with proper indentation
            # Typically functions end with 4 spaces indent
            code_all = code + "\n    return " + ", ".join(s for s in self.prompt_func_outputs)
        else:
            # Code already has return, use as-is
            code_all = code

        return [code_all, algorithm]

    def post_thought(self, code, algorithm):

        prompt_content = self.get_prompt_refine(code, algorithm)

        post_thought = self._get_thought(prompt_content)

        return post_thought

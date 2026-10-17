"""PathWise 9d014a36: response rules copied from utils/pathwise_helpers.py."""
from __future__ import annotations
import re
from typing import List, Tuple
from .graph import PathWiseAction as Action

def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end+1])

    if code_string is None:
        return None

    # Remove any "Description:" or "Algorithmic description:" lines that appear before the function definition
    # These can cause SyntaxError when the code is imported
    lines = code_string.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that start with description formats (case-insensitive)
        stripped_lower = stripped.lower()
        if (stripped_lower.startswith('description:') or
            stripped_lower.startswith('# description:') or
            stripped_lower.startswith('algorithm description:') or
            stripped_lower.startswith('# algorithm description:') or
            stripped_lower.startswith('algorithmic description:') or
            stripped_lower.startswith('# algorithmic description:')):
            continue
        cleaned_lines.append(line)
    code_string = '\n'.join(cleaned_lines)
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    if 'random' in code_string:
        code_string = "import random\n" + code_string
    if 'DBSCAN' in code_string:
        code_string = "from sklearn.cluster import DBSCAN\n" + code_string
    if 'KMeans' in code_string:
        code_string = "from sklearn.cluster import KMeans\n" + code_string
    if 'itertools' in code_string:
        code_string = "import itertools\n" + code_string
    if 'heapq' in code_string:
        code_string = "import heapq\n" + code_string
    if 'AgglomerativeClustering' in code_string:
        code_string = "from sklearn.cluster import AgglomerativeClustering\n" + code_string
    if 'Set' in code_string:
        code_string = "from typing import Set\n" + code_string
    if 'log' in code_string:
        code_string = "from math import log\n" + code_string
    if 'math.' in code_string:
        code_string = "import math\n" + code_string
    if 'deque' in code_string:
        code_string = "from collections import deque\n" + code_string
    return code_string

class ResponseRules:

    def parse_initialization_response(self, response: str) -> Tuple[str, str]:
        """Parse initialization response to extract description and derivation rationale"""
        lines = response.split('\n')
        description = ""
        rationale = ""

        for line in lines:
            line = line.strip()
            # Remove backticks if present (e.g., ```Description: ... ```)
            line = line.replace('```', '')
            line = line.strip()
            # Strip markdown bold/italic markers (e.g., **Description:** or *Description:*)
            line = line.strip('*').strip()

            # Check for both regular format and comment format (# Description: ...)
            if line.startswith("Description:") or line.startswith("# Description:"):
                description = line.replace("Description:", "").replace("# Description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
            elif line.startswith("Derivation Rationale:") or line.startswith("# Derivation Rationale:"):
                rationale = line.replace("Derivation Rationale:", "").replace("# Derivation Rationale:", "").strip()

        # Don't provide fallbacks - require successful parsing
        if not description or not rationale:
            raise ValueError(f"Failed to parse description ('{description}') or rationale ('{rationale}') from LLM response")

        return description, rationale

    def parse_world_model_response(self, response: str) -> Tuple[str, str]:
        """Parse world model response to extract code and description"""
        # Extract code block
        code = extract_code_from_generator(response)
        if code is None:
            raise ValueError("No code found in response")

        # Check if code contains placeholder text (invalid output)
        placeholder_patterns = [
            "[your generated code here]",
            "[your code here]",
            "[code here]",
            "# your code here",
            "# TODO",
        ]
        code_lower = code.lower()
        for pattern in placeholder_patterns:
            if pattern.lower() in code_lower:
                raise ValueError(f"Code contains placeholder text: '{pattern}'")

        # Extract description from 'Description:' or 'Algorithmic description:' line as instructed in prompt
        lines = response.split('\n')
        description = ""

        # First pass: Look for standard description formats in lines
        for line in lines:
            line_stripped = line.strip()
            # Strip markdown bold/italic markers (e.g., **Description:** or *Description:*)
            line_stripped = line_stripped.strip('*').strip()
            # Check for various description formats (with or without # comment prefix)
            if line_stripped.startswith("Description:"):
                description = line_stripped.replace("Description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break
            elif line_stripped.startswith("# Description:"):
                description = line_stripped.replace("# Description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break
            elif line_stripped.startswith("Algorithm description:"):
                description = line_stripped.replace("Algorithm description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break
            elif line_stripped.startswith("# Algorithm description:"):
                description = line_stripped.replace("# Algorithm description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break
            elif line_stripped.startswith("Algorithmic description:"):
                description = line_stripped.replace("Algorithmic description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break
            elif line_stripped.startswith("# Algorithmic description:"):
                description = line_stripped.replace("# Algorithmic description:", "").strip()
                # Remove brackets from description
                description = description.strip('[]')
                break

        # Second pass: If no description found, look for description inside triple backticks (e.g., ```Description: ...```)
        if not description:
            import re
            # Pattern to match ```Description: ...``` or ```description: ...``` (case-insensitive)
            desc_pattern = r'```\s*(?:Algorithm(?:ic)?\s+)?[Dd]escription:\s*(.+?)```'
            match = re.search(desc_pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                # Remove any leading/trailing brackets
                description = description.strip('[]')

        if not description:
            raise ValueError("No description found in response")

        # Check if description contains placeholder text
        desc_lower = description.lower()
        desc_placeholder_patterns = [
            "[algorithmic description",
            "[your description",
            "[description here",
        ]
        for pattern in desc_placeholder_patterns:
            if pattern.lower() in desc_lower:
                raise ValueError(f"Description contains placeholder text: '{pattern}'")

        return code, description

    def parse_policy_response(self, response: str, state: List[HeuristicNode], graph: EntailmentGraph) -> Action:
        """Parse policy model response to extract action"""
        lines = response.strip().split('\n')

        parents = []
        kappa = ""

        # Look for PARENTS and DIRECTIVE in the response
        collecting_directive = False
        directive_lines = []

        for line in lines:
            line = line.strip()

            if line.startswith("PARENTS:"):
                parent_part = line.replace("PARENTS:", "").strip()
                # Remove brackets and extract parent IDs
                parent_part = parent_part.replace("[", "").replace("]", "")

                # Extract all mentioned IDs from parent_part (even if not in current state)
                mentioned_ids = [pid.strip() for pid in parent_part.replace(",", " ").split() if pid.strip()]

                # Extract parent IDs - look for exact matches in current state
                for node in state:
                    if node.node_id in parent_part:
                        if node.node_id not in parents:
                            parents.append(node.node_id)

                # Check if policy mentioned IDs that aren't in current candidates
                available_ids = [node.node_id for node in state]
                invalid_ids = [pid for pid in mentioned_ids if pid not in available_ids]

                if invalid_ids:
                    self.invalid_parent_selections += 1
                    return None  # Skip this action

                # If no exact ID matches or invalid IDs detected with no ancestors added, skip action
                if not parents:
                    return None  # Skip this action

                collecting_directive = False

            elif line.startswith("DIRECTIVE:"):
                kappa = line.replace("DIRECTIVE:", "").strip()
                collecting_directive = True

            elif collecting_directive and line and not line.startswith("PARENTS:"):
                directive_lines.append(line)

        # Combine directive lines
        if directive_lines:
            kappa += " " + " ".join(directive_lines)

        # Fallback if parsing fails
        if not parents and state:
            # Select the first available node
            parents = [state[0].node_id]
        if not kappa:
            kappa = "Mutate the selected parent by adjusting its core logic and improving its selection criteria"

        # Track number of parents selected for statistics
        self.parent_selection_counts.append(len(parents))

        return Action(parents=parents, rationale=kappa)

    def clean_code_for_prompt(self, code: str) -> str:
        """Clean code by removing imports, backticks, and 'python' marker for prompt formatting"""
        # Remove backticks and 'python' marker
        code = code.replace('```python', '').replace('```', '')

        # Find the first 'def' and remove everything before it (imports, etc.)
        lines = code.split('\n')
        first_def_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                first_def_idx = i
                break

        if first_def_idx is not None:
            # Keep only from the first 'def' onwards
            cleaned_lines = lines[first_def_idx:]
            return '\n'.join(cleaned_lines).strip()
        else:
            # If no 'def' found, just remove common import patterns
            cleaned_lines = []
            for line in lines:
                stripped = line.strip()
                if not (stripped.startswith('import ') or stripped.startswith('from ') or stripped == ''):
                    cleaned_lines.append(line)
            return '\n'.join(cleaned_lines).strip()

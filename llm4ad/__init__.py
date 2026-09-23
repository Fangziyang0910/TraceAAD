"""Shared core from the LLM4AD platform (kept with attribution):

- ``llm4ad.base``: code representation (Function/Program/TextFunctionProgramConverter),
  secure multiprocess evaluation (SecureEvaluator) and the LLM sampling helpers.
- ``llm4ad.tools``: OpenAI-compatible LLM client and the run profiler.

Methods and tasks live in the top-level ``traceaad``, ``baselines`` and
``benchmarks`` packages.
"""

from . import base
from . import tools

__version__ = '1.0.0'

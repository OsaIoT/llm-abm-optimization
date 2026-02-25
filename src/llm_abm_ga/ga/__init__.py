"""
Subpackage implementing the genetic algorithm used to evolve prompt
configurations for the firms.

The core logic resides in ``evolution.py`` and ``operators.py``. Import
``evolve_prompts`` from this package to run the algorithm.
"""

from .evolution import evolve_prompts

__all__ = ["evolve_prompts"]

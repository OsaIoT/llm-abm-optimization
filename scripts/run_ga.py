"""
CLI script to run prompt evolution using a genetic algorithm.

This script wires together the components of the genetic algorithm and
market simulation to evolve prompt configurations over multiple
generations. Fitness is measured by the cumulative profit that each
firm earns when participating in the market.

The results of each generation are logged to a JSON Lines file in the
``results/logs`` directory. Each line contains summary statistics for
that generation as well as the encoded population.

Usage:
    python scripts/run_ga.py

You can customise the genetic algorithm parameters by editing the
arguments passed to ``evolve_prompts``. See ``llm_abm_ga.ga.evolution``
for parameter descriptions.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_abm_ga.ga.evolution import evolve_prompts
from llm_abm_ga.prompts.components import (
    system_template_phrases,
    user_template_phrases,
)
from llm_abm_ga.logging_utils import setup_logger
from llm_abm_ga.config import OPENAI_API_KEY

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set. Put it in .env (project root) or in environment variables.")


def main() -> None:
    # Prepare a log file for this run. The logger will create the
    # directory if necessary and return the path to a JSONL file.
    log_file = setup_logger("results/logs")
    # Run the genetic algorithm. Adjust these parameters as needed.
    _ = evolve_prompts(
        system_template_phrases,
        user_template_phrases,
        n_generations=15,
        pop_size=50,
        n_buyers=80000,
        periods=10,
        tournament_k=6,
        elite_size=5,
        mutation_rate=0.1,
        log_file=log_file,
    )
    print(f"Evolution complete. See {log_file} for details.")


if __name__ == "__main__":
    main()

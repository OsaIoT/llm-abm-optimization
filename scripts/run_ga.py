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
    python scripts/run_ga.py                          # paper defaults
    python scripts/run_ga.py --generations 5 --pop-size 20 --buyers 5000

Every generation runs one market with ``pop_size`` LLM firms, i.e.
``pop_size * periods`` API calls per generation: start small to check
the cost before launching a full run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_abm_ga import config
from llm_abm_ga.ga.evolution import evolve_prompts
from llm_abm_ga.logging_utils import setup_logger
from llm_abm_ga.prompts.components import (
    system_template_phrases,
    user_template_phrases,
)
from llm_abm_ga.utils import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (defaults are the settings used in the paper)."""
    parser = argparse.ArgumentParser(description="Evolve system-prompt components with a binary GA.")
    parser.add_argument("--generations", type=int, default=15, help="Number of generations (default: 15)")
    parser.add_argument("--pop-size", type=int, default=50, help="Population size = firms per market (default: 50)")
    parser.add_argument("--buyers", type=int, default=80_000, help="Buyers per market (default: 80000)")
    parser.add_argument("--periods", type=int, default=10, help="Trading periods per market (default: 10)")
    parser.add_argument("--tournament-k", type=int, default=6, help="Tournament size (default: 6)")
    parser.add_argument("--elite-size", type=int, default=5, help="Elites kept each generation (default: 5)")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Per-bit mutation rate (default: 0.1)")
    parser.add_argument("--log-dir", default="results/logs", help="Where the JSONL log is written")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the GA/market RNGs. LLM replies stay non-deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.require_api_key()
    if args.seed is not None:
        seed_everything(args.seed)

    # Prepare a log file for this run. The logger will create the
    # directory if necessary and return the path to a JSONL file.
    log_file = setup_logger(args.log_dir)
    _ = evolve_prompts(
        system_template_phrases,
        user_template_phrases,
        n_generations=args.generations,
        pop_size=args.pop_size,
        n_buyers=args.buyers,
        periods=args.periods,
        tournament_k=args.tournament_k,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate,
        log_file=log_file,
    )
    print(f"Evolution complete. See {log_file} for details.")


if __name__ == "__main__":
    main()

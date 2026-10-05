"""
CLI script to run a single simulation of the market model.

This script constructs a simple market with one firm and a set of buyers
and runs it for a number of periods. It demonstrates how to use the
``MarketModel`` class in isolation without the genetic algorithm.

By default the firm uses a prompt built from all of the available
``system_template_phrases`` and no ``user_template_phrases``. You can
customise the number of buyers and periods by editing the values in
this script or by modifying the call in your own code.

Usage:
    python scripts/run_market.py

When executed, the script prints the cumulative profit earned by the
firm at the end of the simulation.
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
from llm_abm_ga.market.model import MarketModel
from llm_abm_ga.prompts.components import (
    system_template_phrases,
    user_template_phrases,
)
from llm_abm_ga.prompts.templates import decode_individual
from llm_abm_ga.utils import seed_everything


def build_default_prompt() -> dict:
    """Build a default prompt using all system phrases and no user phrases."""
    system_bits = [1] * len(system_template_phrases)
    # ensure at least one user bit exists even if list is empty
    user_bits = [0] * len(user_template_phrases)
    return decode_individual((system_bits, user_bits), system_template_phrases, user_template_phrases)


def run_simulation(n_buyers: int, periods: int) -> None:
    """Run a simulation with a single firm and print the results.

    Args:
        n_buyers: Number of buyers participating in the market.
        periods: Number of time periods to simulate.
    """
    prompt_config = build_default_prompt()
    model = MarketModel([prompt_config], n_buyers=n_buyers, periods=periods)
    model.run()
    firm = model.firms[0]
    total_profit = sum(firm.profit_history)
    print(f"Simulation finished. Total profit: {total_profit:.2f}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description="Run a single market simulation.")
    parser.add_argument(
        "--buyers",
        type=int,
        default=100,
        help="Number of buyers in the simulation (default: 100)",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=10,
        help="Number of periods to simulate (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the market RNGs. LLM replies stay non-deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.require_api_key()
    if args.seed is not None:
        seed_everything(args.seed)
    run_simulation(args.buyers, args.periods)


if __name__ == "__main__":
    main()

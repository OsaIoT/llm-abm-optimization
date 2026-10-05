"""
Reproducibility re-run of the gpt-4o-mini GA.

Re-runs 3 independent GA evolutions with gpt-4o-mini at T=0, with the same settings
as the runs in results/ga_runs/gpt-4o-mini, to see whether the GA finds the same
genome again. The committed logs in experiments/reproducibility/data were produced
on 2026-05-31.

Settings: 15 generations, pop_size=50, 80k buyers, 10 periods,
tournament_k=6, elite_size=5, mutation_rate=0.1.

Resumable: skips runs whose log file already exists.

Usage (calls the API: 3 runs x 15 generations x 50 firms x 10 periods):
    python experiments/reproducibility/rerun_ga.py
"""

import sys
import os
import random
import pathlib
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.chdir(ROOT)

from llm_abm_ga import config
from llm_abm_ga.ga.evolution import evolve_prompts
from llm_abm_ga.prompts.components import (
    system_template_phrases,
    user_template_phrases,
)

config.LLM_PROVIDER = "openai"
config.LLM_MODEL = "gpt-4o-mini"
config.LLM_TEMPERATURE = 0.0

config.require_api_key()

N_RUNS = 3
N_GENERATIONS = 15
POP_SIZE = 50
N_BUYERS = 80_000
PERIODS = 10
TOURNAMENT_K = 6
ELITE_SIZE = 5
MUTATION_RATE = 0.1
BASE_SEED = 1000

DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"


def run_single(run_idx: int) -> str:
    """Run one GA evolution. Returns log file path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log_file = str(DATA_DIR / f"run_{run_idx + 1:02d}.jsonl")

    if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
        print(f"  [SKIP] {log_file} already exists")
        return log_file

    seed = BASE_SEED + run_idx
    random.seed(seed)
    np.random.seed(seed)

    evolve_prompts(
        system_template_phrases,
        user_template_phrases,
        n_generations=N_GENERATIONS,
        pop_size=POP_SIZE,
        n_buyers=N_BUYERS,
        periods=PERIODS,
        tournament_k=TOURNAMENT_K,
        elite_size=ELITE_SIZE,
        mutation_rate=MUTATION_RATE,
        log_file=log_file,
    )
    return log_file


def main():
    print("=" * 60)
    print("Reproducibility re-run (gpt-4o-mini, T=0)")
    print("=" * 60)
    print(f"Model: {config.LLM_MODEL}, T={config.LLM_TEMPERATURE}")
    print(f"Runs: {N_RUNS}")
    print(f"GA: {N_GENERATIONS} gens, pop={POP_SIZE}, buyers={N_BUYERS}, periods={PERIODS}")
    print()

    t0 = time.time()

    for run_idx in range(N_RUNS):
        print(f"\n--- Run {run_idx + 1}/{N_RUNS} ---")
        t_run = time.time()
        log_file = run_single(run_idx)
        elapsed = time.time() - t_run
        print(f"  Done in {elapsed:.1f}s -> {log_file}")

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All done. Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Logs saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()

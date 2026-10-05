"""
Bankruptcy frequency analysis.

Runs 20 optimized-vs-baseline market simulations (1 optimized + 9 random
firms each) and records how often each firm type triggers the bankruptcy
rule (3 consecutive negative-profit periods).

100k buyers, 10 periods, T=0, same protocol as results/optimized_vs_baseline.

Which model plays: ``LLM_MODEL`` from .env / the environment (default gpt-4o-mini).
The genome below was evolved for gpt-3.5-turbo, the model of results/optimized_vs_baseline;
the committed results played it with gpt-4o-mini, i.e. a transfer setting. For a matched
run use LLM_MODEL=gpt-3.5-turbo.

Usage (calls the API: 20 markets x 10 firms x 10 periods):
    python experiments/bankruptcy/run_bankruptcy.py
"""

import sys, os, random, pathlib, json
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

os.chdir(ROOT)

from llm_abm_ga import config
from llm_abm_ga.market.model import MarketModel
from llm_abm_ga.prompts.templates import decode_individual
from llm_abm_ga.prompts.components import system_template_phrases, user_template_phrases
from llm_abm_ga.ga.operators import generate_individual

OPTIMIZED_GENOME = ([1, 1, 1, 0, 0, 1, 1, 0], [0])  # 11100110: evolved for gpt-3.5-turbo
N_SYSTEM = len(system_template_phrases)
N_USER = len(user_template_phrases)
N_RUNS = 20
N_BASELINE = 9
N_BUYERS = 100_000
PERIODS = 10
BASE_SEED = 42


def run_single(run_idx: int):
    random.seed(BASE_SEED + run_idx)
    np.random.seed(BASE_SEED + run_idx)

    opt_cfg = decode_individual(OPTIMIZED_GENOME)

    base_cfgs = []
    opt_bits = OPTIMIZED_GENOME[0]
    for _ in range(N_BASELINE):
        while True:
            ind = generate_individual(N_SYSTEM, N_USER)
            if ind[0] != opt_bits:
                break
        base_cfgs.append(decode_individual(ind))

    all_cfgs = [opt_cfg] + base_cfgs
    model = MarketModel(all_cfgs, n_buyers=N_BUYERS, periods=PERIODS)
    model.run()

    opt_firm = model.firms[0]
    base_firms = model.firms[1:]

    return {
        "run": run_idx + 1,
        "opt_active": opt_firm.active,
        "opt_profit": opt_firm.cumulative_profit,
        "base_active": [f.active for f in base_firms],
        "base_profits": [f.cumulative_profit for f in base_firms],
    }


def main():
    config.require_api_key()
    print(f"Model: {config.LLM_MODEL}, T={config.LLM_TEMPERATURE}")
    print(f"Running {N_RUNS} markets: 1 optimized + {N_BASELINE} baseline firms")
    print(f"Buyers: {N_BUYERS}, Periods: {PERIODS}")
    print()

    results = []
    for i in range(N_RUNS):
        print(f"--- Run {i+1}/{N_RUNS} ---")
        r = run_single(i)
        results.append(r)
        status = "active" if r["opt_active"] else "BANKRUPT"
        n_base_bankrupt = sum(1 for a in r["base_active"] if not a)
        print(f"  Opt: {status} (profit={r['opt_profit']:.0f}), "
              f"Baseline bankrupt: {n_base_bankrupt}/{N_BASELINE}")

    opt_bankrupt = sum(1 for r in results if not r["opt_active"])
    total_base = N_RUNS * N_BASELINE
    base_bankrupt = sum(
        1 for r in results for a in r["base_active"] if not a
    )

    opt_profits = [r["opt_profit"] for r in results]
    base_profits_flat = [p for r in results for p in r["base_profits"]]

    lines = []
    lines.append("=" * 60)
    lines.append("Bankruptcy Frequency Analysis")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Model: {config.LLM_MODEL}, T={config.LLM_TEMPERATURE}")
    lines.append(f"Protocol: {N_RUNS} runs x (1 opt + {N_BASELINE} baseline)")
    lines.append(f"Buyers: {N_BUYERS}, Periods: {PERIODS}")
    lines.append("")
    lines.append("--- Bankruptcy rates ---")
    lines.append(f"  Optimized firms:  {opt_bankrupt}/{N_RUNS} "
                 f"({opt_bankrupt/N_RUNS*100:.1f}%)")
    lines.append(f"  Baseline firms:   {base_bankrupt}/{total_base} "
                 f"({base_bankrupt/total_base*100:.1f}%)")
    lines.append("")
    lines.append("--- Profit summary ---")
    lines.append(f"  Optimized mean profit: {np.mean(opt_profits):+.1f} "
                 f"(std={np.std(opt_profits):.1f})")
    lines.append(f"  Baseline mean profit:  {np.mean(base_profits_flat):+.1f} "
                 f"(std={np.std(base_profits_flat):.1f})")
    lines.append("")

    lines.append("--- Per-run detail ---")
    for r in results:
        n_bb = sum(1 for a in r["base_active"] if not a)
        opt_s = "active" if r["opt_active"] else "BANKRUPT"
        lines.append(
            f"  Run {r['run']:2d}  opt={opt_s:8s} profit={r['opt_profit']:+10.0f}  "
            f"base_bankrupt={n_bb}/{N_BASELINE}"
        )

    output = "\n".join(lines)
    print("\n" + output)

    out_dir = pathlib.Path(__file__).resolve().parent
    (out_dir / "results_summary.txt").write_text(output, encoding="utf-8")

    raw = {
        "config": {
            "model": config.LLM_MODEL,
            "optimized_genome": "".join(str(b) for b in OPTIMIZED_GENOME[0]),
            "temperature": config.LLM_TEMPERATURE,
            "n_runs": N_RUNS,
            "n_baseline": N_BASELINE,
            "n_buyers": N_BUYERS,
            "periods": PERIODS,
        },
        "summary": {
            "opt_bankrupt_count": opt_bankrupt,
            "opt_bankrupt_rate": opt_bankrupt / N_RUNS,
            "base_bankrupt_count": base_bankrupt,
            "base_bankrupt_rate": base_bankrupt / total_base,
        },
        "runs": results,
    }
    (out_dir / "results_raw.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

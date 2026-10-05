"""
Prompt-level variance test.

Sends the SAME optimized prompt to gpt-4o-mini at T=0 many times and measures how
often the (price, quality) decision changes: the residual non-determinism of a
hosted model at temperature 0.

Two market states are defined (period 1 without history, period 5 with history), but
the prompt does not include the market state (see "Limitations" in the README), so
both scenarios send the same messages: this is 2 x 100 calls of one prompt.

Usage (calls the API: 200 requests):
    python experiments/reproducibility/prompt_variance_test.py
"""

import os
import sys
import json
import re
import pathlib
import time
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except Exception:
    pass

from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set.")

client = OpenAI(api_key=OPENAI_API_KEY)
MODEL = "gpt-4o-mini"
N_CALLS = 100
COST_INDEX = 0.75
OUT_DIR = pathlib.Path(__file__).resolve().parent

# --- Prompt components (optimized genotype: 10011100 = bits 0,3,4,5) ---
system_template_phrases = [
    "You are a company operating in a competitive market composed by {n_competitors} other firms and {n_buyers} buyers.",
    "You have access to historical data on your own performance and market trends.",
    "Use this information to analyze the market and determine the optimal price and quality level for your product.",
    "Buyers select products based on the highest quality-to-price ratio, The higher it is, the more likely they are to buy from you.",
    "Each buyer purchases only one product per period.",
    "The cost of production is {cost_index} times the quality level.",
    "The profit of each period is given by the formula: profit = #sold_products * (price - cost).",
    "Your primary goal is to maximize profit in each period, and never incur a loss.",
]

MANDATORY_SYSTEM_LINE = (
    "Price must be between 0.1 and 1. Quality must be between 0 and 1.\n"
    "   output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)
MANDATORY_USER_LINE = (
    "You must output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)

OPTIMIZED_BITS = [1, 0, 0, 1, 1, 1, 0, 0]

SCENARIOS = {
    "period_1_no_history": {
        "n_competitors": 9,
        "n_buyers": 80000,
        "current_period": 1,
        "total_periods": 10,
        "price_history": [],
        "quality_history": [],
        "sold_history": [],
        "last_period_profit": "0.00",
        "cumulative_profit": "0.00",
        "competitor_prices": [0.0] * 9,
        "competitor_qualities": [0.0] * 9,
        "competitor_sales": [0] * 9,
    },
    "period_5_with_history": {
        "n_competitors": 9,
        "n_buyers": 80000,
        "current_period": 5,
        "total_periods": 10,
        "price_history": [0.5, 0.5, 0.5, 0.5],
        "quality_history": [0.6, 0.6, 0.6, 0.6],
        "sold_history": [12000, 11500, 11800, 12200],
        "last_period_profit": "1830.00",
        "cumulative_profit": "7100.00",
        "competitor_prices": [0.4, 0.6, 0.3, 0.5, 0.7, 0.45, 0.55, 0.35, 0.65],
        "competitor_qualities": [0.5, 0.7, 0.3, 0.6, 0.8, 0.5, 0.65, 0.4, 0.7],
        "competitor_sales": [9000, 8500, 7000, 10000, 9500, 8000, 9200, 7500, 8800],
    },
}


def build_messages(scenario):
    active_phrases = [p for b, p in zip(OPTIMIZED_BITS, system_template_phrases) if b]
    system_prompt = "\n".join(active_phrases).format(
        n_competitors=scenario["n_competitors"],
        n_buyers=scenario["n_buyers"],
        cost_index=COST_INDEX,
    )
    system_prompt += "\n" + MANDATORY_SYSTEM_LINE

    user_prompt = " "  # user_template_phrases = [" "], bit is off so empty
    user_prompt += "\n" + MANDATORY_USER_LINE

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_api(messages):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return completion.choices[0].message.content.strip()


def parse_decision(response):
    match = re.search(r'DECISION:\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', response)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def run_scenario(name, scenario, n_calls):
    messages = build_messages(scenario)
    print(f"\n--- Scenario: {name} ({n_calls} calls) ---")
    print(f"System prompt:\n  {messages[0]['content'][:120]}...")

    results = []
    parse_errors = 0
    t0 = time.time()

    for i in range(n_calls):
        try:
            response = call_api(messages)
            p, q = parse_decision(response)
            if p is not None:
                results.append((p, q))
            else:
                parse_errors += 1
                print(f"  Call {i+1}: parse error — {response[:60]}")
        except Exception as e:
            parse_errors += 1
            print(f"  Call {i+1}: API error — {e}")

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n_calls} done...")

    elapsed = time.time() - t0

    prices = [r[0] for r in results]
    qualities = [r[1] for r in results]
    pq_pairs = Counter(results)

    stats = {
        "n_calls": n_calls,
        "n_valid": len(results),
        "parse_errors": parse_errors,
        "elapsed_seconds": round(elapsed, 1),
        "price_mean": round(np.mean(prices), 6) if prices else None,
        "price_std": round(np.std(prices), 6) if prices else None,
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "quality_mean": round(np.mean(qualities), 6) if qualities else None,
        "quality_std": round(np.std(qualities), 6) if qualities else None,
        "quality_min": min(qualities) if qualities else None,
        "quality_max": max(qualities) if qualities else None,
        "unique_pq_pairs": len(pq_pairs),
        "most_common_pq": pq_pairs.most_common(3),
        "all_results": results,
    }
    return stats


def main():
    print("=" * 60)
    print("Prompt-level Variance Test (gpt-4o-mini, T=0)")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Optimized genotype: {''.join(str(b) for b in OPTIMIZED_BITS)}")
    print(f"Calls per scenario: {N_CALLS}")

    all_stats = {}
    lines = []
    lines.append("=" * 60)
    lines.append("Prompt-level Variance Test Results")
    lines.append("=" * 60)
    lines.append(f"Model: {MODEL}, T=0")
    lines.append(f"Optimized genotype: {''.join(str(b) for b in OPTIMIZED_BITS)}")
    lines.append(f"Calls per scenario: {N_CALLS}")
    lines.append("")

    for name, scenario in SCENARIOS.items():
        stats = run_scenario(name, scenario, N_CALLS)
        all_stats[name] = stats

        lines.append(f"--- Scenario: {name} ---")
        lines.append(f"  Valid responses: {stats['n_valid']}/{stats['n_calls']}")
        lines.append(f"  Parse errors: {stats['parse_errors']}")
        lines.append(f"  Price:   mean={stats['price_mean']}, std={stats['price_std']}, "
                      f"range=[{stats['price_min']}, {stats['price_max']}]")
        lines.append(f"  Quality: mean={stats['quality_mean']}, std={stats['quality_std']}, "
                      f"range=[{stats['quality_min']}, {stats['quality_max']}]")
        lines.append(f"  Unique (p,q) pairs: {stats['unique_pq_pairs']}")
        lines.append(f"  Most common:")
        for pq, count in stats['most_common_pq']:
            lines.append(f"    (p={pq[0]}, q={pq[1]}): {count}/{stats['n_valid']} "
                          f"({count/stats['n_valid']*100:.1f}%)")
        lines.append("")

    determinism_rate = min(
        s["most_common_pq"][0][1] / s["n_valid"] * 100
        for s in all_stats.values()
        if s["n_valid"] > 0 and s["most_common_pq"]
    )
    lines.append(f"--- Overall ---")
    lines.append(f"  Minimum determinism rate: {determinism_rate:.1f}%")
    lines.append(f"  (fraction of calls returning the most common response)")

    output = "\n".join(lines)
    print("\n" + output)
    (OUT_DIR / "variance_test_results.txt").write_text(output, encoding="utf-8")

    raw = {
        "config": {
            "model": MODEL,
            "temperature": 0,
            "n_calls_per_scenario": N_CALLS,
            "optimized_genotype": "".join(str(b) for b in OPTIMIZED_BITS),
        },
        "scenarios": {
            name: {k: v for k, v in stats.items() if k != "all_results"}
            for name, stats in all_stats.items()
        },
        "raw_decisions": {
            name: stats["all_results"]
            for name, stats in all_stats.items()
        },
    }
    (OUT_DIR / "variance_test_raw.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

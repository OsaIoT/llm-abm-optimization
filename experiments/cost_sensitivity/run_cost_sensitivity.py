"""
Cost index out-of-distribution test.

Tests whether the genome optimized at k=0.75 remains competitive when
deployed in markets with different cost regimes (k=0.3, 0.5, 0.75, 0.9).

Self-contained: re-implements the market with direct OpenAI calls (the notebook
code that produced the original genome) so that the cost index can be varied;
it does not import the llm_abm_ga package.

Protocol per k value:
  - 10 market simulations: 1 optimized firm + 9 random baseline firms
  - 100k buyers, 10 periods, gpt-4o-mini, T=0

Usage:
    python experiments/cost_sensitivity/run_cost_sensitivity.py
"""

import os
import sys
import random
import json
import re
import time
import pathlib
from concurrent.futures import ThreadPoolExecutor

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
BUYER_TEMPERATURE = 0.1

OPTIMIZED_GENOME = ([1, 0, 0, 1, 1, 1, 0, 0], [0])
K_VALUES = [0.3, 0.5, 0.75, 0.9]
N_RUNS = 10
N_BASELINE = 9
N_BUYERS = 100_000
PERIODS = 10
BASE_SEED = 3000

OUT_DIR = pathlib.Path(__file__).resolve().parent

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
user_template_phrases = [" "]

MANDATORY_SYSTEM_LINE = (
    "Price must be between 0.1 and 1. Quality must be between 0 and 1. \n"
    "   output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)
MANDATORY_USER_LINE = (
    "You must output your answer EXACTLY in this format: \n"
    "DECISION: (price, quality), you cannot add any other text"
)


def generate_answer(messages):
    completion = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0
    )
    return completion.choices[0].message.content


class Firm:
    def __init__(self, firm_id, system_prompt_template, user_prompt_template,
                 n_competitors, n_buyers, total_periods, cost_index):
        self.cost_index = cost_index
        self.id = firm_id
        self.n_competitors = n_competitors
        self.n_buyers = n_buyers
        self.total_periods = total_periods
        self.system_prompt = system_prompt_template.format(
            n_competitors=n_competitors, n_buyers=n_buyers, cost_index=cost_index)
        self.user_prompt_template = user_prompt_template
        self.p, self.q, self.cost = 0, 0, 0
        self.last_period_sold = 0
        self.sold = 0
        self.revenue = 0
        self.last_period_profit = 0
        self.cumulative_profit = 0
        self.last_period_revenue = 0
        self.price_history, self.quality_history = [], []
        self.sold_history, self.profit_history = [], []
        self.negative_streak = 0
        self.active = True

    def make_decision(self, market_info):
        user_prompt = self.user_prompt_template.format(
            firm_id=self.id, current_period=market_info['current_period'],
            total_periods=self.total_periods, price_history=self.price_history,
            quality_history=self.quality_history, sold_history=self.sold_history,
            last_period_profit=f"{self.last_period_profit:.2f}",
            cumulative_profit=f"{self.cumulative_profit:.2f}",
            competitor_prices=market_info['competitor_prices'],
            competitor_qualities=market_info['competitor_qualities'],
            competitor_sales=market_info['competitor_sales'])
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            response = generate_answer(messages).strip()
            lines = response.splitlines()
            decision_line = next((l for l in lines if l.strip().startswith("DECISION:")), None)
            if decision_line is None:
                raise ValueError(f"No DECISION line: {response}")
            match = re.match(r'DECISION:\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)', decision_line)
            if not match:
                raise ValueError(f"Malformed: {decision_line}")
            price, quality = float(match.group(1)), float(match.group(2))
            if not (0.1 <= price <= 1) or not (0 <= quality <= 1):
                raise ValueError(f"Out of bounds: p={price}, q={quality}")
            self.p = price
            self.q = quality
        except Exception as e:
            print(f"Error Firm {self.id}: {e}")
            self.p = round(np.random.uniform(0.1, 1), 2)
            self.q = round(np.random.uniform(0, 1), 2)

        self.cost = self.cost_index * self.q
        self.price_history.append(self.p)
        self.quality_history.append(self.q)


class Buyer:
    def __init__(self, buyer_id):
        self.id = buyer_id

    def purchase_decision(self, firms, temperature=BUYER_TEMPERATURE):
        active_firms = [f for f in firms if f.active]
        if not active_firms:
            return None
        utility = {f.id: f.q / f.p for f in active_firms if f.p > 0}
        if not utility:
            return None
        utilities = np.array(list(utility.values()))
        scaled = utilities / temperature
        exp_u = np.exp(scaled - np.max(scaled))
        probs = exp_u / np.sum(exp_u)
        return np.random.choice(list(utility.keys()), p=probs)


class MarketModel:
    def __init__(self, firm_prompt_configs, n_buyers=10, periods=10, cost_index=0.75):
        self.n_firms = len(firm_prompt_configs)
        self.n_buyers = n_buyers
        self.T = periods
        self.t = 0
        self.cost_index = cost_index
        self.firms = [
            Firm(i, cfg['system_template'], cfg['user_template'],
                 self.n_firms - 1, self.n_buyers, self.T, cost_index=cost_index)
            for i, cfg in enumerate(firm_prompt_configs)
        ]
        self.buyers = [Buyer(i) for i in range(n_buyers)]

    def run(self):
        for _ in range(self.T):
            active_firms = [f for f in self.firms if f.active]
            if not active_firms:
                return
            market_info = {
                "competitor_prices": [f.p for f in active_firms],
                "competitor_qualities": [f.q for f in active_firms],
                "competitor_sales": [f.last_period_sold for f in active_firms],
                "current_period": self.t + 1
            }
            with ThreadPoolExecutor(max_workers=len(active_firms)) as executor:
                futures = [executor.submit(f.make_decision, market_info) for f in active_firms]
                for fut in futures:
                    fut.result()
            for f in self.firms:
                f.last_period_sold = 0
                f.last_period_profit = 0
                f.last_period_revenue = 0
            for buyer in self.buyers:
                active = [f for f in self.firms if f.active]
                if not active:
                    return
                choice = buyer.purchase_decision(random.sample(active, len(active)))
                if choice is not None:
                    sel = next(f for f in self.firms if f.id == choice)
                    sel.sold += 1
                    sel.last_period_sold += 1
                    sel.revenue += sel.p
                    sel.last_period_revenue += sel.p
                    sel.cumulative_profit += sel.p - sel.cost
            for f in self.firms:
                if f.active:
                    f.last_period_profit = f.last_period_revenue - (f.cost * f.last_period_sold)
                    f.profit_history.append(f.last_period_profit)
                    f.sold_history.append(f.last_period_sold)
                    if f.last_period_profit < 0:
                        f.negative_streak += 1
                    else:
                        f.negative_streak = 0
                    if f.negative_streak >= 3:
                        f.active = False
                        print(f"Firm {f.id} exited (3 consecutive losses).")
            self.t += 1


def decode_individual(individual, cost_index):
    sys_bits, usr_bits = individual
    sys_prompt = "\n".join(p for b, p in zip(sys_bits, system_template_phrases) if b)
    sys_prompt += "\n" + MANDATORY_SYSTEM_LINE
    usr_prompt = "\n".join(p for b, p in zip(usr_bits, user_template_phrases) if b)
    usr_prompt += "\n" + MANDATORY_USER_LINE
    return {"system_template": sys_prompt, "user_template": usr_prompt}


def generate_random_individual():
    return ([random.choice([0, 1]) for _ in range(8)], [0])


def run_single(k, run_idx):
    seed = BASE_SEED + int(k * 100) + run_idx
    random.seed(seed)
    np.random.seed(seed)

    opt_cfg = decode_individual(OPTIMIZED_GENOME, k)
    base_cfgs = []
    for _ in range(N_BASELINE):
        while True:
            ind = generate_random_individual()
            if ind[0] != OPTIMIZED_GENOME[0]:
                break
        base_cfgs.append(decode_individual(ind, k))

    all_cfgs = [opt_cfg] + base_cfgs
    model = MarketModel(all_cfgs, n_buyers=N_BUYERS, periods=PERIODS, cost_index=k)
    model.run()

    opt_firm = model.firms[0]
    base_firms = model.firms[1:]

    return {
        "run": run_idx + 1,
        "k": k,
        "opt_profit": opt_firm.cumulative_profit,
        "opt_active": opt_firm.active,
        "base_profits": [f.cumulative_profit for f in base_firms],
        "base_active": [f.active for f in base_firms],
    }


def main():
    print("=" * 60)
    print("Cost Index Out-of-Distribution Test")
    print("=" * 60)
    print(f"Model: {MODEL}, T=0")
    print(f"Optimized genome: {''.join(str(b) for b in OPTIMIZED_GENOME[0])}")
    print(f"K values: {K_VALUES}")
    print(f"Runs per k: {N_RUNS}, Buyers: {N_BUYERS}, Periods: {PERIODS}")
    print()

    all_results = {}
    t0 = time.time()

    for k in K_VALUES:
        print(f"\n{'='*40}")
        print(f"  k = {k}")
        print(f"{'='*40}")
        results = []
        for run_idx in range(N_RUNS):
            print(f"  Run {run_idx + 1}/{N_RUNS}...", end=" ")
            r = run_single(k, run_idx)
            results.append(r)
            status = "active" if r["opt_active"] else "BANKRUPT"
            n_bb = sum(1 for a in r["base_active"] if not a)
            print(f"opt={status} profit={r['opt_profit']:+.0f}, base_bankrupt={n_bb}/{N_BASELINE}")

        opt_profits = [r["opt_profit"] for r in results]
        base_profits_flat = [p for r in results for p in r["base_profits"]]
        opt_bankrupt = sum(1 for r in results if not r["opt_active"])
        base_bankrupt = sum(1 for r in results for a in r["base_active"] if not a)

        all_results[k] = {
            "results": results,
            "opt_mean": float(np.mean(opt_profits)),
            "opt_std": float(np.std(opt_profits)),
            "base_mean": float(np.mean(base_profits_flat)),
            "base_std": float(np.std(base_profits_flat)),
            "opt_bankrupt": opt_bankrupt,
            "base_bankrupt": base_bankrupt,
        }

    total = time.time() - t0
    print(f"\nTotal time: {total / 60:.1f} minutes")

    # --- Summary ---
    lines = []
    lines.append("=" * 60)
    lines.append("Cost Index Out-of-Distribution Test")
    lines.append("=" * 60)
    lines.append(f"Model: {MODEL}, T=0")
    lines.append(f"Optimized genome: {''.join(str(b) for b in OPTIMIZED_GENOME[0])}")
    lines.append(f"Protocol: {N_RUNS} runs x (1 opt + {N_BASELINE} baseline), {N_BUYERS} buyers, {PERIODS} periods")
    lines.append("")
    lines.append(f"{'k':>5}  {'Opt profit':>14}  {'Base profit':>14}  {'Opt bankr':>10}  {'Base bankr':>11}")
    lines.append("-" * 60)
    for k in K_VALUES:
        s = all_results[k]
        marker = " <-- training" if k == 0.75 else ""
        lines.append(
            f"{k:5.2f}  {s['opt_mean']:+10.1f}±{s['opt_std']:5.0f}  "
            f"{s['base_mean']:+10.1f}±{s['base_std']:5.0f}  "
            f"{s['opt_bankrupt']:4d}/{N_RUNS:2d}      "
            f"{s['base_bankrupt']:4d}/{N_RUNS * N_BASELINE:3d}{marker}"
        )
    lines.append("")

    for k in K_VALUES:
        s = all_results[k]
        lines.append(f"--- k = {k} (per-run detail) ---")
        for r in s["results"]:
            n_bb = sum(1 for a in r["base_active"] if not a)
            opt_s = "active" if r["opt_active"] else "BANKRUPT"
            lines.append(
                f"  Run {r['run']:2d}  opt={opt_s:8s} profit={r['opt_profit']:+10.0f}  "
                f"base_bankrupt={n_bb}/{N_BASELINE}"
            )
        lines.append("")

    output = "\n".join(lines)
    print("\n" + output)
    (OUT_DIR / "results_summary.txt").write_text(output, encoding="utf-8")

    raw = {
        "config": {
            "model": MODEL,
            "temperature": 0,
            "optimized_genome": "".join(str(b) for b in OPTIMIZED_GENOME[0]),
            "k_values": K_VALUES,
            "n_runs": N_RUNS,
            "n_baseline": N_BASELINE,
            "n_buyers": N_BUYERS,
            "periods": PERIODS,
        },
        "results": {
            str(k): {key: val for key, val in s.items() if key != "results"}
            for k, s in all_results.items()
        },
        "runs": {str(k): s["results"] for k, s in all_results.items()},
    }
    (OUT_DIR / "results_raw.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )
    print(f"Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

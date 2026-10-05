"""Load the GA run logs and experiment outputs and compute the numbers quoted in the README.

Pure NumPy/SciPy on purpose: it can be unit-tested without matplotlib, and every
figure in ``make_figures.py`` is drawn from the dictionaries returned here.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats

# Display order: OpenAI models first, then Gemini, each from smaller/older to larger/newer.
MODEL_ORDER = [
    "gpt-3.5-turbo",
    "gpt-4o-mini",
    "gpt-5-nano",
    "gpt-5-mini",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
]

Genome = Tuple[int, ...]
Run = List[dict]  # one record per generation, as written by llm_abm_ga.logging_utils


# --------------------------------------------------------------------------- loading
def read_run(path) -> Run:
    """Read one JSON Lines GA log (one record per generation)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ga_runs(root) -> Dict[str, List[Run]]:
    """Read ``root/<model>/run_*.jsonl`` into ``{model: [run, ...]}`` in display order."""
    root = Path(root)
    found = {d.name: sorted(d.glob("run_*.jsonl")) for d in root.iterdir() if d.is_dir()}
    ordered = [m for m in MODEL_ORDER if m in found] + sorted(set(found) - set(MODEL_ORDER))
    return {model: [read_run(p) for p in found[model]] for model in ordered}


def load_json(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- genomes
def system_genome(individual) -> Genome:
    """The system-prompt bits of a logged individual ``[system_bits, user_bits]``."""
    return tuple(int(b) for b in individual[0])


def genome_str(genome: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in genome)


def hamming(a: Sequence[int], b: Sequence[int]) -> int:
    return sum(x != y for x, y in zip(a, b))


def gene_activation(run: Run) -> np.ndarray:
    """Share of the population with each gene switched on, shape ``(generations, genes)``."""
    return np.array(
        [np.mean([system_genome(ind) for ind in rec["population"]], axis=0) for rec in run]
    )


def final_best_genome(run: Run) -> Genome:
    """Best individual of the last generation."""
    return system_genome(run[-1]["best_individual"])


def convergence_summary(runs_by_model: Dict[str, List[Run]]) -> List[dict]:
    """Per model: each run's final best genome, whether the runs agree, and the final prompt density."""
    rows = []
    for model, runs in runs_by_model.items():
        genomes = [final_best_genome(r) for r in runs]
        distances = [hamming(a, b) for a, b in itertools.combinations(genomes, 2)]
        rows.append(
            {
                "model": model,
                "genomes": [genome_str(g) for g in genomes],
                "identical": len(set(genomes)) == 1,
                "mean_hamming": float(np.mean(distances)) if distances else 0.0,
                # Average share of components switched on in the final population.
                "final_density": float(np.mean([gene_activation(r)[-1].mean() for r in runs])),
            }
        )
    return rows


def component_counts(runs_by_model: Dict[str, List[Run]]) -> List[int]:
    """For each component, in how many of the final best genomes (one per run) it is switched on."""
    genomes = [final_best_genome(r) for runs in runs_by_model.values() for r in runs]
    return [int(c) for c in np.sum(genomes, axis=0)]


# --------------------------------------------------------------------------- GA vs random search
def random_search_vs_ga(runs_by_model: Dict[str, List[Run]]) -> dict:
    """Best of the random prompts of generation 1 vs. the best fitness reached in the whole run.

    Generation 1 is a random sample of the search space, so its best individual is a
    random-search baseline with *one* market evaluation of ``pop_size`` prompts; the GA
    gets one such evaluation per generation. The two are therefore not budget-matched.
    """
    rows = []
    for model, runs in runs_by_model.items():
        for i, run in enumerate(runs, start=1):
            gen1 = run[0]["best_fitness"]
            peak = max(rec["best_fitness"] for rec in run)
            rows.append(
                {
                    "model": model,
                    "run": i,
                    "gen1_best": gen1,
                    "ga_peak": peak,
                    "improved": peak > gen1 + 1e-9,
                    "gen1_mean": run[0]["mean_fitness"],
                    "last_mean": run[-1]["mean_fitness"],
                }
            )
    gen1 = np.array([r["gen1_best"] for r in rows])
    peak = np.array([r["ga_peak"] for r in rows])
    test = stats.wilcoxon(peak, gen1, alternative="greater")  # zero differences are dropped
    return {
        "runs": rows,
        "n_runs": len(rows),
        "n_improved": int(sum(r["improved"] for r in rows)),
        "not_improved": [f"{r['model']} #{r['run']}" for r in rows if not r["improved"]],
        "mean_gen1_best": float(gen1.mean()),
        "mean_ga_peak": float(peak.mean()),
        "relative_gain": float(peak.mean() / gen1.mean() - 1),
        "wilcoxon_p": float(test.pvalue),
        "mean_fitness_gen1": float(np.mean([r["gen1_mean"] for r in rows])),
        "mean_fitness_last": float(np.mean([r["last_mean"] for r in rows])),
    }


def per_model_gain(summary: dict) -> List[dict]:
    """Aggregate the per-run rows of :func:`random_search_vs_ga` by model."""
    out: Dict[str, dict] = {}
    for r in summary["runs"]:
        m = out.setdefault(r["model"], {"model": r["model"], "gen1": [], "peak": [], "improved": 0, "n": 0})
        m["gen1"].append(r["gen1_best"])
        m["peak"].append(r["ga_peak"])
        m["improved"] += int(r["improved"])
        m["n"] += 1
    return [
        {
            "model": m["model"],
            "mean_gen1_best": float(np.mean(m["gen1"])),
            "mean_ga_peak": float(np.mean(m["peak"])),
            "improved": m["improved"],
            "n": m["n"],
        }
        for m in out.values()
    ]


# --------------------------------------------------------------------------- optimized vs baseline
def cohens_d(a, b) -> float:
    """Cohen's d with the pooled standard deviation (ddof=1)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled)


def optimized_vs_baseline(path) -> dict:
    """One GA-optimized firm vs. nine random-prompt firms, repeated in independent markets."""
    raw = load_json(path)
    opt = np.array(raw["opt_runs"], dtype=float)
    base = np.array(raw["base_runs"], dtype=float)  # per market: mean profit of the baseline firms
    return {
        "params": raw["params"],
        "optimized": opt.tolist(),
        "baseline": base.tolist(),
        "n_runs": int(opt.size),
        "mean_optimized": float(opt.mean()),
        "sd_optimized": float(opt.std(ddof=1)),
        "mean_baseline": float(base.mean()),
        "sd_baseline": float(base.std(ddof=1)),
        "cohens_d": cohens_d(opt, base),
        "wins": int((opt > base).sum()),
        "wilcoxon_p": float(stats.wilcoxon(opt, base, alternative="greater").pvalue),
    }


# --------------------------------------------------------------------------- other experiments
def bankruptcy_summary(path) -> dict:
    """Bankruptcy counts of the optimized firm and of the random-prompt baselines."""
    raw = load_json(path)
    runs = raw["runs"]
    base_active = np.array([r["base_active"] for r in runs], dtype=bool)
    return {
        "config": raw["config"],
        "n_runs": len(runs),
        "opt_bankrupt": int(sum(not r["opt_active"] for r in runs)),
        "base_bankrupt": int((~base_active).sum()),
        "base_firms": int(base_active.size),
        "mean_profit_optimized": float(np.mean([r["opt_profit"] for r in runs])),
        "mean_profit_baseline": float(np.mean([r["base_profits"] for r in runs])),
    }


def cost_sensitivity_summary(path) -> dict:
    """Optimized genome deployed at cost indices other than the one it was evolved at."""
    raw = load_json(path)
    cfg = raw["config"]
    rows = [{"k": float(k), **v} for k, v in raw["results"].items()]
    return {
        "config": cfg,
        "rows": sorted(rows, key=lambda r: r["k"]),
        "n_opt": cfg["n_runs"],
        "n_base": cfg["n_runs"] * cfg["n_baseline"],
    }

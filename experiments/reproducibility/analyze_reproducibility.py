"""
Compare the re-run GA logs with the original gpt-4o-mini runs.

Reads:
  - re-runs from experiments/reproducibility/data/run_{i}.jsonl
  - original runs from results/ga_runs/gpt-4o-mini/

Produces results_summary.txt and results_raw.json next to this script.

Usage (offline):
    python experiments/reproducibility/analyze_reproducibility.py
"""

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEW_DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
ORIG_DATA_DIR = ROOT / "results" / "ga_runs" / "gpt-4o-mini"
OUT_DIR = pathlib.Path(__file__).resolve().parent


def read_run(log_path: pathlib.Path) -> dict:
    """Read a JSONL log and extract per-generation stats + final best individual."""
    generations = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                generations.append(json.loads(line))

    final = generations[-1]
    best = final["best_individual"]
    system_bits = tuple(best[0])

    trajectory = [(g["generation"], g["best_fitness"], g["mean_fitness"]) for g in generations]

    return {
        "n_generations": len(generations),
        "final_best_genotype": system_bits,
        "final_best_fitness": final["best_fitness"],
        "trajectory": trajectory,
    }


def bits_to_str(bits: tuple) -> str:
    return "".join(str(b) for b in bits)


def hamming(a: tuple, b: tuple) -> int:
    return sum(x != y for x, y in zip(a, b))


def main():
    new_runs = []
    for i in range(1, 4):
        path = NEW_DATA_DIR / f"run_{i:02d}.jsonl"
        if not path.exists():
            print(f"WARNING: missing {path}")
            continue
        new_runs.append(read_run(path))

    orig_runs = []
    for i in range(1, 4):
        path = ORIG_DATA_DIR / f"run_{i:02d}.jsonl"
        if not path.exists():
            print(f"WARNING: missing original {path}")
            continue
        orig_runs.append(read_run(path))

    if not new_runs or not orig_runs:
        print("ERROR: insufficient data to compare")
        return

    lines = []
    lines.append("=" * 60)
    lines.append("Reproducibility Analysis")
    lines.append("=" * 60)
    lines.append(f"Model: gpt-4o-mini, T=0")
    lines.append("")

    lines.append("--- Original runs (results/ga_runs/gpt-4o-mini) ---")
    for i, r in enumerate(orig_runs):
        lines.append(f"  Run {i+1}: genotype={bits_to_str(r['final_best_genotype'])}, "
                      f"fitness={r['final_best_fitness']:.1f}")

    lines.append("")
    lines.append("--- New runs (re-run) ---")
    for i, r in enumerate(new_runs):
        lines.append(f"  Run {i+1}: genotype={bits_to_str(r['final_best_genotype'])}, "
                      f"fitness={r['final_best_fitness']:.1f}")

    lines.append("")
    lines.append("--- Genotype comparison ---")

    orig_genotypes = [r["final_best_genotype"] for r in orig_runs]
    new_genotypes = [r["final_best_genotype"] for r in new_runs]
    all_genotypes = orig_genotypes + new_genotypes

    from collections import Counter
    geno_counts = Counter(all_genotypes)
    dominant = geno_counts.most_common(1)[0]
    lines.append(f"  Most frequent genotype across all 6 runs: {bits_to_str(dominant[0])} "
                  f"({dominant[1]}/6 runs)")

    orig_match = sum(1 for g in orig_genotypes if g == dominant[0])
    new_match = sum(1 for g in new_genotypes if g == dominant[0])
    lines.append(f"    Original: {orig_match}/3, Re-run: {new_match}/3")

    lines.append("")
    lines.append("--- Hamming distances (new vs original) ---")
    for i, nr in enumerate(new_runs):
        for j, orr in enumerate(orig_runs):
            h = hamming(nr["final_best_genotype"], orr["final_best_genotype"])
            lines.append(f"  New run {i+1} vs Original run {j+1}: Hamming = {h}/8")

    lines.append("")
    lines.append("--- Cross-run Hamming distances (within new runs) ---")
    for i in range(len(new_runs)):
        for j in range(i + 1, len(new_runs)):
            h = hamming(new_runs[i]["final_best_genotype"],
                        new_runs[j]["final_best_genotype"])
            lines.append(f"  New run {i+1} vs New run {j+1}: Hamming = {h}/8")

    lines.append("")
    lines.append("--- Fitness trajectories (best fitness per generation) ---")
    lines.append("  Original runs:")
    for i, r in enumerate(orig_runs):
        traj = ", ".join(f"{bf:.0f}" for _, bf, _ in r["trajectory"])
        lines.append(f"    Run {i+1}: [{traj}]")
    lines.append("  New runs:")
    for i, r in enumerate(new_runs):
        traj = ", ".join(f"{bf:.0f}" for _, bf, _ in r["trajectory"])
        lines.append(f"    Run {i+1}: [{traj}]")

    output = "\n".join(lines)
    print(output)
    (OUT_DIR / "results_summary.txt").write_text(output, encoding="utf-8")

    raw = {
        "config": {
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "n_runs": len(new_runs),
            "n_generations": 15,
            "pop_size": 50,
            "n_buyers": 80000,
            "periods": 10,
        },
        "original_runs": [
            {
                "genotype": bits_to_str(r["final_best_genotype"]),
                "fitness": r["final_best_fitness"],
                "trajectory": [(g, bf, mf) for g, bf, mf in r["trajectory"]],
            }
            for r in orig_runs
        ],
        "new_runs": [
            {
                "genotype": bits_to_str(r["final_best_genotype"]),
                "fitness": r["final_best_fitness"],
                "trajectory": [(g, bf, mf) for g, bf, mf in r["trajectory"]],
            }
            for r in new_runs
        ],
        "dominant_genotype": bits_to_str(dominant[0]),
        "dominant_count": dominant[1],
    }
    (OUT_DIR / "results_raw.json").write_text(
        json.dumps(raw, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()

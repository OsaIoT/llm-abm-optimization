"""Re-create the README figures and tables from the committed logs. No API key needed.

    pip install -e ".[analysis]"
    python analysis/make_figures.py

Writes ``docs/images/*_{light,dark}.png`` and ``results/summary.md`` (+ ``summary.json``).
Every number quoted in the README comes from ``results/summary.md``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python analysis/make_figures.py` without a package install
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from analysis import ga_logs as gl  # noqa: E402
from analysis import style  # noqa: E402
from llm_abm_ga.prompts.components import system_template_phrases  # noqa: E402

MARKER_AREA = 50  # points^2, about 9 px across
RING = 1.5  # surface-coloured ring around markers, about 2 px


def fmt_p(p: float) -> str:
    """Coarse p-value for the text: the exact digits of an approximate test vary with the SciPy version."""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def thousands(value, _pos=None) -> str:
    return f"{value:,.0f}".replace("-", "−")


def marker_handle(color: str, label: str, theme: style.Theme) -> Line2D:
    return Line2D(
        [], [], marker="o", linestyle="", markersize=7, markerfacecolor=color,
        markeredgecolor=theme.surface, markeredgewidth=RING, label=label,
    )


def header(fig, title: str, subtitle: str, theme: style.Theme) -> None:
    """Left-aligned title and (possibly multi-line) subtitle, placed in inches from the top edge."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.10 / h, title, fontsize=10.5, fontweight="bold", color=theme.ink, va="top")
    fig.text(0.01, 1 - 0.40 / h, subtitle, fontsize=8.5, color=theme.ink_secondary, va="top", linespacing=1.5)


# --------------------------------------------------------------------------- figures
def fig_convergence(runs_by_model, theme):
    """Small multiples: share of each generation's population that has each component on."""
    style.apply(theme)
    models = list(runs_by_model)
    fig, axes = plt.subplots(2, 4, figsize=(7.6, 4.9), gridspec_kw={"hspace": 0.5, "wspace": 0.16})
    flat = axes.ravel()
    mesh = None
    for idx, (ax, model) in enumerate(zip(flat, models)):
        act = np.mean([gl.gene_activation(r) for r in runs_by_model[model]], axis=0)
        n_gen, n_genes = act.shape
        mesh = ax.pcolormesh(
            np.arange(n_gen + 1), np.arange(n_genes + 1), act.T, cmap=theme.cmap(),
            vmin=0, vmax=1, edgecolors=theme.surface, linewidth=0.6,
        )
        ax.set_ylim(n_genes, 0)  # component 0 on top
        ax.set_xlim(0, n_gen)
        ax.set_xticks([g - 0.5 for g in (1, 5, 10, 15) if g <= n_gen])
        ax.set_xticklabels([str(g) for g in (1, 5, 10, 15) if g <= n_gen], fontsize=8)
        if idx % 4 == 0:
            ax.set_yticks(np.arange(n_genes) + 0.5)
            ax.set_yticklabels([str(i) for i in range(n_genes)], fontsize=8)
        else:
            ax.set_yticks([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(model, loc="left", fontsize=8.5, color=theme.ink, pad=4)

    key = flat[len(models)]
    key.axis("off")
    cax = key.inset_axes([0.0, 0.50, 1.0, 0.09])
    cbar = matplotlib.pyplot.colorbar(mesh, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    cbar.ax.set_xticklabels(["0%", "50%", "100%"], fontsize=8)
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)
    key.text(0.0, 0.70, "Share of the population\nwith the component on", transform=key.transAxes,
             fontsize=8, color=theme.ink_secondary, va="bottom")

    fig.supxlabel("generation", fontsize=8.5, color=theme.ink_secondary, y=0.05)
    fig.supylabel("prompt component (index)", fontsize=8.5, color=theme.ink_secondary, x=0.03)
    header(fig, "Which prompt components survive selection",
           "Mean over 3 independent GA runs per model; 50 prompts per generation", theme)
    fig.subplots_adjust(top=0.84, left=0.08, right=0.99, bottom=0.12)
    return fig


def fig_ga_vs_random(summary, theme):
    """Dumbbell per run: best of the 50 random prompts of generation 1 -> best the GA ever found."""
    style.apply(theme)
    rows = summary["runs"]
    models = list(dict.fromkeys(r["model"] for r in rows))
    gap, y, ypos, centers = 0.9, 0.0, {}, []
    for model in models:
        ys = []
        for r in (r for r in rows if r["model"] == model):
            ypos[(model, r["run"])] = y
            ys.append(y)
            y += 1
        centers.append(float(np.mean(ys)))
        y += gap
    last_y = y - gap - 1

    height = 0.22 * y + 2.3
    fig, ax = plt.subplots(figsize=(7.4, height))
    ys = np.array([ypos[(r["model"], r["run"])] for r in rows])
    gen1 = np.array([r["gen1_best"] for r in rows])
    peak = np.array([r["ga_peak"] for r in rows])
    for yy, a, b in zip(ys, gen1, peak):
        ax.plot([a, b], [yy, yy], color=theme.muted, lw=1.5, zorder=1)
    ax.scatter(gen1, ys, s=MARKER_AREA, color=theme.blue_low, edgecolors=theme.surface, linewidths=RING, zorder=2)
    ax.scatter(peak, ys, s=MARKER_AREA, color=theme.blue_high, edgecolors=theme.surface, linewidths=RING, zorder=3)

    ax.set_ylim(last_y + 0.8, -0.8)
    ax.set_yticks(centers)
    ax.set_yticklabels(models, fontsize=8.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-90, max(peak.max(), gen1.max()) * 1.04)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.axvline(0, color=theme.axis, lw=1.0, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Fitness of the best firm (cumulative profit over 10 periods)", fontsize=8.5)

    handles = [
        marker_handle(theme.blue_low, "Best of the 50 random prompts (generation 1)", theme),
        marker_handle(theme.blue_high, "Best prompt found by the GA (any of 15 generations)", theme),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.95 / height), ncol=1,
               fontsize=8.5, labelcolor=theme.ink_secondary, handletextpad=0.4, labelspacing=0.3)
    header(fig, "GA vs. random search, 21 independent runs (7 models x 3)",
           f"Mean best fitness {summary['mean_gen1_best']:,.0f} \u2192 {summary['mean_ga_peak']:,.0f}; "
           f"{summary['n_improved']} of {summary['n_runs']} runs improved "
           "(a dark dot alone means no gain).\n"
           "Budgets differ: 1 market evaluation for random search, 15 for the GA.", theme)
    fig.subplots_adjust(top=1 - 1.6 / height, left=0.2, right=0.985, bottom=0.6 / height)
    return fig


def fig_optimized_vs_baseline(summary, theme):
    """Paired dots per market: the GA-optimized firm vs. the mean of nine random-prompt firms."""
    style.apply(theme)
    opt = np.array(summary["optimized"])
    base = np.array(summary["baseline"])
    order = np.argsort(-opt)  # best market on top
    opt, base = opt[order], base[order]
    y = np.arange(len(opt))

    height = 5.6
    fig, ax = plt.subplots(figsize=(7.4, height))
    for yy, a, b in zip(y, base, opt):
        ax.plot([a, b], [yy, yy], color=theme.muted, lw=1.5, zorder=1)
    ax.scatter(base, y, s=MARKER_AREA, color=theme.orange, edgecolors=theme.surface, linewidths=RING, zorder=3)
    ax.scatter(opt, y, s=MARKER_AREA, color=theme.blue, edgecolors=theme.surface, linewidths=RING, zorder=3)
    ax.axvline(0, color=theme.axis, lw=1.0, zorder=0)
    for mean, color in ((summary["mean_optimized"], theme.blue), (summary["mean_baseline"], theme.orange)):
        ax.plot([mean, mean], [-0.7, len(opt) - 0.4], color=color, lw=1.0, zorder=0)  # starts below its label
        ax.text(mean, -0.95, f"mean {thousands(mean)}", ha="center", va="bottom", fontsize=8, color=theme.ink_secondary)

    ax.set_ylim(len(opt) - 0.4, -1.6)
    ax.set_yticks([])
    ax.set_ylabel("20 independent markets, best first", fontsize=8.5)
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Cumulative profit after 10 periods", fontsize=8.5)

    handles = [
        marker_handle(theme.blue, "GA-optimized firm", theme),
        marker_handle(theme.orange, "Mean of the 9 random-prompt firms", theme),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 1 - 0.78 / height), ncol=2, fontsize=8.5,
               labelcolor=theme.ink_secondary, handletextpad=0.4, columnspacing=1.6)
    header(fig, "GA-optimized prompt vs. random prompts, gpt-3.5-turbo",
           f"Genome 11100110. Above the baseline mean in {summary['wins']} of {summary['n_runs']} markets; "
           f"Cohen's d = {summary['cohens_d']:.2f}", theme)
    fig.subplots_adjust(top=1 - 1.2 / height, left=0.04, right=0.985, bottom=0.55 / height)
    return fig


# --------------------------------------------------------------------------- tables
def components_on(genome: str) -> str:
    return ",".join(str(i) for i, b in enumerate(genome) if b == "1") or "-"


def markdown_summary(s: dict) -> str:
    """The tables quoted in the README (a 'table view' of the figures)."""
    out = ["# Summary of the committed results", "",
           "Generated by `python analysis/make_figures.py`; do not edit by hand.", ""]

    n_genomes = sum(len(r["genomes"]) for r in s["convergence"])
    out += ["## Prompt component library", "",
            f"The last column counts the final best genomes (one per run, {n_genomes} in total) with the component on.", "",
            "| Index | Component | On in final best genomes |", "|---|---|---|"]
    out += [f"| {i} | {text} | {n}/{n_genomes} |"
            for i, (text, n) in enumerate(zip(system_template_phrases, s["component_counts"]))]

    out += ["", "## Best genome of each run's final generation", "",
            "Genomes are bit strings over the 8 components (`1` = component on).", "",
            "| Model | Run 1 | Run 2 | Run 3 | Same in all runs | Mean Hamming distance | Components on in final population |",
            "|---|---|---|---|---|---|---|"]
    for r in s["convergence"]:
        g = r["genomes"]
        out.append(
            f"| {r['model']} | `{g[0]}` | `{g[1]}` | `{g[2]}` | {'yes' if r['identical'] else 'no'} "
            f"| {r['mean_hamming']:.2f} | {r['final_density']:.0%} |"
        )
    n_same = sum(r["identical"] for r in s["convergence"])
    out += ["", f"{n_same} of {len(s['convergence'])} models reach the same best genome in all 3 runs."]

    rs = s["random_search"]
    out += ["", "## GA vs. random search", "",
            "Random search = the best of the 50 random prompts of generation 1 (one market evaluation). "
            "GA = the best fitness over all 15 generations (15 evaluations).", "",
            "| Model | Mean best, generation 1 | Mean best, GA | Runs improved |", "|---|---|---|---|"]
    for r in s["per_model_gain"]:
        out.append(f"| {r['model']} | {r['mean_gen1_best']:,.0f} | {r['mean_ga_peak']:,.0f} | {r['improved']}/{r['n']} |")
    out += ["", f"All runs: {rs['mean_gen1_best']:,.1f} → {rs['mean_ga_peak']:,.1f} "
            f"({rs['relative_gain']:+.1%}); {rs['n_improved']}/{rs['n_runs']} runs improved "
            f"(no gain: {', '.join(rs['not_improved'])}); one-sided Wilcoxon signed-rank {fmt_p(rs['wilcoxon_p'])}. "
            f"Mean population fitness: {rs['mean_fitness_gen1']:,.0f} (generation 1) → "
            f"{rs['mean_fitness_last']:,.0f} (generation 15)."]

    ob = s["optimized_vs_baseline"]
    out += ["", "## Optimized vs. random-prompt firms (gpt-3.5-turbo, genome `11100110`)", "",
            f"{ob['n_runs']} independent markets, each with 1 optimized firm and "
            f"{ob['params']['n_baseline_per_run']} random-prompt firms "
            f"({ob['params']['buyers']:,} buyers, {ob['params']['periods']} periods).", "",
            "| | Mean cumulative profit | SD |", "|---|---|---|",
            f"| Optimized firm | {ob['mean_optimized']:,.0f} | {ob['sd_optimized']:,.0f} |",
            f"| Mean of the random-prompt firms | {ob['mean_baseline']:,.0f} | {ob['sd_baseline']:,.0f} |", "",
            f"Cohen's d = {ob['cohens_d']:.2f}; optimized > baseline mean in {ob['wins']}/{ob['n_runs']} markets "
            f"(one-sided Wilcoxon signed-rank {fmt_p(ob['wilcoxon_p'])})."]

    bk = s["bankruptcy"]
    out += ["", "## Bankruptcy (3 consecutive losing periods)", "",
            f"{bk['n_runs']} markets with 1 optimized + {bk['base_firms'] // bk['n_runs']} random-prompt firms; "
            f"firms are played by `{bk['config']['model']}` at temperature {bk['config']['temperature']}. "
            "The optimized genome is `11100110`, the one evolved for gpt-3.5-turbo (a transfer setting).", "",
            "| | Bankruptcies | Mean cumulative profit |", "|---|---|---|",
            f"| Optimized firm | {bk['opt_bankrupt']}/{bk['n_runs']} | {bk['mean_profit_optimized']:,.0f} |",
            f"| Random-prompt firms | {bk['base_bankrupt']}/{bk['base_firms']} "
            f"({bk['base_bankrupt'] / bk['base_firms']:.1%}) | {bk['mean_profit_baseline']:,.0f} |"]

    cs = s["cost_sensitivity"]
    out += ["", "## Cost index the genome was not evolved at", "",
            f"`{cs['config']['model']}` plays genome `{cs['config']['optimized_genome']}` (evolved at k = 0.75); "
            f"{cs['n_opt']} markets per k, each with 1 optimized + {cs['config']['n_baseline']} random-prompt firms.", "",
            "| k | Optimized: mean profit | Optimized: bankruptcies | Random: mean profit | Random: bankruptcies |",
            "|---|---|---|---|---|"]
    for r in cs["rows"]:
        out.append(
            f"| {r['k']:.2f} | {r['opt_mean']:,.0f} | {r['opt_bankrupt']}/{cs['n_opt']} "
            f"| {r['base_mean']:,.0f} | {r['base_bankrupt']}/{cs['n_base']} |"
        )
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- main
def build_summary(results: Path, experiments: Path) -> dict:
    runs = gl.load_ga_runs(results / "ga_runs")
    random_search = gl.random_search_vs_ga(runs)
    return {
        "convergence": gl.convergence_summary(runs),
        "component_counts": gl.component_counts(runs),
        "random_search": random_search,
        "per_model_gain": gl.per_model_gain(random_search),
        "optimized_vs_baseline": gl.optimized_vs_baseline(results / "optimized_vs_baseline" / "log_fig_8.json"),
        "bankruptcy": gl.bankruptcy_summary(experiments / "bankruptcy" / "results_raw.json"),
        "cost_sensitivity": gl.cost_sensitivity_summary(experiments / "cost_sensitivity" / "results_raw.json"),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--experiments", type=Path, default=ROOT / "experiments")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "images")
    parser.add_argument("--summary-dir", type=Path, default=None, help="Where summary.md/json go (default: --results)")
    args = parser.parse_args(argv)
    summary_dir = args.summary_dir or args.results

    summary = build_summary(args.results, args.experiments)
    runs = gl.load_ga_runs(args.results / "ga_runs")
    args.out.mkdir(parents=True, exist_ok=True)

    for theme in style.THEMES:
        figures = {
            "convergence": fig_convergence(runs, theme),
            "ga_vs_random_search": fig_ga_vs_random(summary["random_search"], theme),
            "optimized_vs_baseline": fig_optimized_vs_baseline(summary["optimized_vs_baseline"], theme),
        }
        for name, fig in figures.items():
            path = args.out / f"{name}_{theme.name}.png"
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=theme.surface)
            plt.close(fig)
            print("wrote", path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote summary.md and summary.json to {summary_dir}")


if __name__ == "__main__":
    main()

"""The numbers quoted in the README must follow from the committed logs.

If a log or the analysis code changes, these tests fail instead of the README silently drifting.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import ga_logs as gl
from analysis import make_figures

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXPERIMENTS = ROOT / "experiments"


@pytest.fixture(scope="module")
def runs():
    return gl.load_ga_runs(RESULTS / "ga_runs")


# ----------------------------------------------------------------------------- helpers
def test_hamming_and_genome_str():
    assert gl.hamming((0, 1, 0, 1), (0, 0, 1, 1)) == 2
    assert gl.genome_str((1, 0, 0, 1, 1, 1, 0, 0)) == "10011100"


def test_cohens_d_on_a_toy_example():
    assert gl.cohens_d([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)
    assert gl.cohens_d([1, 2, 3], [2, 3, 4]) == pytest.approx(-1.0)


# ----------------------------------------------------------------------------- data integrity
def test_every_log_has_the_documented_shape(runs):
    assert list(runs) == gl.MODEL_ORDER
    for model, model_runs in runs.items():
        assert len(model_runs) == 3, model
        for run in model_runs:
            assert [rec["generation"] for rec in run] == list(range(1, 16))
            for rec in run:
                assert len(rec["population"]) == len(rec["fitness_scores"]) == 50
                # 8 system bits; the single user bit maps to a blank phrase, so mutation can flip it harmlessly.
                assert all(len(ind[0]) == 8 and ind[1] in ([0], [1]) for ind in rec["population"])


def test_logged_summaries_agree_with_the_raw_fitness_scores(runs):
    for model_runs in runs.values():
        for run in model_runs:
            for rec in run:
                scores = rec["fitness_scores"]
                assert rec["best_fitness"] == pytest.approx(max(scores))
                assert rec["mean_fitness"] == pytest.approx(np.mean(scores))
                assert rec["best_individual"] == rec["population"][scores.index(max(scores))]


def test_gene_activation_is_a_share(runs):
    act = gl.gene_activation(runs["gpt-4o-mini"][0])
    assert act.shape == (15, 8)
    assert act.min() >= 0 and act.max() <= 1


# ----------------------------------------------------------------------------- README numbers
def test_convergence_summary(runs):
    rows = {r["model"]: r for r in gl.convergence_summary(runs)}
    assert {m for m, r in rows.items() if r["identical"]} == {
        "gpt-3.5-turbo", "gpt-5-mini", "gemini-2.0-flash", "gemini-3-flash-preview",
    }
    assert rows["gpt-3.5-turbo"]["genomes"] == ["11100110"] * 3
    assert rows["gpt-4o-mini"]["genomes"] == ["01010100", "10011100", "10011100"]
    assert rows["gpt-4o-mini"]["mean_hamming"] == pytest.approx(2.0)


def test_random_search_vs_ga(runs):
    s = gl.random_search_vs_ga(runs)
    assert s["n_runs"] == 21 and s["n_improved"] == 18
    assert s["mean_gen1_best"] == pytest.approx(447.7, abs=0.1)
    assert s["mean_ga_peak"] == pytest.approx(1190.3, abs=0.1)
    assert s["relative_gain"] == pytest.approx(1.659, abs=0.001)
    assert s["wilcoxon_p"] < 1e-3
    assert sorted(s["not_improved"]) == ["gemini-2.5-flash #1", "gemini-2.5-flash #3", "gpt-5-nano #1"]


def test_optimized_vs_baseline():
    s = gl.optimized_vs_baseline(RESULTS / "optimized_vs_baseline" / "log_fig_8.json")
    assert s["n_runs"] == 20 and s["wins"] == 20
    assert s["mean_optimized"] == pytest.approx(14731, abs=1)
    assert s["mean_baseline"] == pytest.approx(-8558, abs=1)
    assert s["cohens_d"] == pytest.approx(2.23, abs=0.01)


def test_bankruptcy_summary_matches_the_stored_summary():
    path = EXPERIMENTS / "bankruptcy" / "results_raw.json"
    s = gl.bankruptcy_summary(path)
    stored = json.loads(path.read_text())["summary"]
    assert (s["opt_bankrupt"], s["base_bankrupt"], s["base_firms"]) == (0, 52, 180)
    assert s["opt_bankrupt"] == stored["opt_bankrupt_count"]
    assert s["base_bankrupt"] == stored["base_bankrupt_count"]


def test_cost_sensitivity_summary():
    s = gl.cost_sensitivity_summary(EXPERIMENTS / "cost_sensitivity" / "results_raw.json")
    assert [r["k"] for r in s["rows"]] == [0.3, 0.5, 0.75, 0.9]
    assert s["rows"][-1]["opt_bankrupt"] == 5 and s["rows"][-1]["base_bankrupt"] == 37


def assert_close(actual, expected, path="summary"):
    """Recursive comparison; floats to 1e-6, and only an order-of-magnitude check on the Wilcoxon p-values."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, float):
        if path.endswith("wilcoxon_p"):
            assert actual < 1e-3, path
        else:
            assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), path
    else:
        assert actual == expected, path


def test_the_committed_summary_is_up_to_date():
    fresh = json.loads(json.dumps(make_figures.build_summary(RESULTS, EXPERIMENTS)))
    committed = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    assert_close(fresh, committed)


def test_make_figures_writes_every_figure(tmp_path):
    make_figures.main(["--out", str(tmp_path / "img"), "--summary-dir", str(tmp_path / "out")])
    names = sorted(p.name for p in (tmp_path / "img").iterdir())
    assert names == [
        f"{fig}_{theme}.png"
        for fig in ("convergence", "ga_vs_random_search", "optimized_vs_baseline")
        for theme in ("dark", "light")
    ]
    assert (tmp_path / "out" / "summary.md").exists()


def test_component_counts(runs):
    # Component 5 (cost of production) is on in every final best genome; component 7 (explicit goal) in only 5.
    assert gl.component_counts(runs) == [13, 12, 9, 10, 10, 21, 11, 5]

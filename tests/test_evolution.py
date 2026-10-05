"""End-to-end check of the GA with a fake LLM: no API key, no network, a few hundred ms."""

import json
import random

import numpy as np

from llm_abm_ga.agents import firm as firm_mod
from llm_abm_ga.ga.evolution import evolve_prompts
from llm_abm_ga.prompts.components import system_template_phrases, user_template_phrases

# Index of the component that explains how buyers choose ("quality-to-price ratio").
BUYER_RULE_GENE = next(i for i, p in enumerate(system_template_phrases) if "quality-to-price" in p)


def fake_llm(messages):
    """A prompt that explains the buyers' rule gets a sensible offer, any other a greedy one."""
    if "quality-to-price" in messages[0]["content"]:
        return "DECISION: (0.7, 0.6)"
    return "DECISION: (1.0, 0.1)"  # priced out: buyers pick on quality/price


def test_ga_learns_to_keep_the_component_that_matters(monkeypatch, tmp_path):
    monkeypatch.setattr(firm_mod, "generate_answer", fake_llm)
    random.seed(0)
    np.random.seed(0)
    log_file = tmp_path / "evolution_log.jsonl"
    n_generations, pop_size = 6, 30

    population = evolve_prompts(
        system_template_phrases,
        user_template_phrases,
        n_generations=n_generations,
        pop_size=pop_size,
        n_buyers=500,
        periods=3,
        tournament_k=3,
        elite_size=2,
        mutation_rate=0.1,
        log_file=str(log_file),
    )

    assert len(population) == pop_size

    records = [json.loads(line) for line in log_file.read_text().splitlines()]
    assert [r["generation"] for r in records] == list(range(1, n_generations + 1))
    assert set(records[0]) == {
        "generation", "population", "fitness_scores", "mean_fitness", "best_fitness", "best_individual",
    }
    # Only prompts that explain the buyers' rule can earn money, so every generation's best has it...
    assert all(r["best_individual"][0][BUYER_RULE_GENE] == 1 for r in records)
    assert all(r["best_fitness"] > 0 for r in records)
    # ...and selection spreads that component through the population.
    def share(pop):
        return np.mean([ind[0][BUYER_RULE_GENE] for ind in pop])

    assert share(population) >= 0.6
    assert share(population) > share([tuple(i) for i in records[0]["population"]])

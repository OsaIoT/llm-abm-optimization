import random

import pytest

from llm_abm_ga.ga import operators as ops


def test_population_has_the_requested_size_and_distinct_genomes():
    random.seed(0)
    pop = ops.generate_population_counts(50, 8, 1)
    assert len(pop) == 50
    assert len({(tuple(s), tuple(u)) for s, u in pop}) == 50
    assert all(len(s) == 8 and u == [0] for s, u in pop)


def test_population_larger_than_the_search_space_raises_instead_of_looping():
    with pytest.raises(ValueError):
        ops.generate_population_counts(17, 4, 1)  # only 2**4 = 16 genomes exist


def test_legacy_generate_population_signature():
    pop = ops.generate_population(10, ["a"] * 8, [" "])
    assert len(pop) == 10


def test_crossover_keeps_the_genes_both_parents_agree_on():
    parent1 = ([1, 1, 0, 0], [0])
    parent2 = ([1, 0, 0, 1], [0])
    for seed in range(50):
        random.seed(seed)
        child_system, _ = ops.crossover(parent1, parent2)
        assert child_system[0] == 1 and child_system[2] == 0


def test_mutation_rate_extremes():
    ind = ([1, 0, 1, 0, 1, 0, 1, 0], [0])
    assert ops.mutate(ind, 0.0) == ind
    flipped_system, flipped_user = ops.mutate(ind, 1.0)
    assert flipped_system == [1 - b for b in ind[0]]
    assert flipped_user == [1]


def test_elitism_returns_the_fittest_in_order():
    pop = [([i], [0]) for i in range(5)]
    fitness = [3, 9, 1, 7, 5]
    assert ops.apply_elitism(pop, fitness, 2) == [([1], [0]), ([3], [0])]


def test_tournament_selection_has_population_size_and_favours_fit_individuals():
    random.seed(0)
    pop = [([i], [0]) for i in range(20)]
    selected = ops.tournament_selection(pop, list(range(20)), k=5)
    assert len(selected) == 20
    mean_index = sum(ind[0][0] for ind in selected) / len(selected)
    assert mean_index > 12  # a random pick would average 9.5


def test_next_generation_keeps_the_elites_and_the_population_size():
    random.seed(0)
    pop = [([i], [0]) for i in range(20)]
    fitness = [float(i) for i in range(20)]
    nxt = ops.select_next_generation(pop, fitness, k=3, elite_size=2)
    assert len(nxt) == 20
    assert nxt[:2] == ops.apply_elitism(pop, fitness, 2)


def test_fitness_is_cumulative_profit():
    class Firm:
        profit_history = [1.0, -2.0, 5.5]

    class Idle:
        profit_history: list = []

    assert ops.compute_fitness(Firm()) == pytest.approx(4.5)
    assert ops.compute_fitness(Idle()) == 0.0
    with pytest.raises(AttributeError):
        ops.compute_fitness(object())

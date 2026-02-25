from __future__ import annotations

import random
from typing import List, Tuple

from ..market.model import MarketModel
from ..prompts.templates import decode_individual
from ..logging_utils import log_generation_data, setup_logger

from .operators import (
    generate_population_counts,
    select_next_generation,
    crossover,
    mutate,
    compute_fitness,
)

Individual = Tuple[List[int], List[int]]


def evolve_prompts(
    system_phrases: List[str],
    user_phrases: List[str],
    *,
    n_generations: int = 20,
    pop_size: int = 50,
    n_buyers: int = 80000,
    periods: int = 10,
    tournament_k: int = 6,   # come vecchio
    elite_size: int = 5,     # come vecchio
    mutation_rate: float = 0.1,
    log_file: str | None = None,
) -> List[Individual]:
    """
    Legacy-style evolve_prompts, compatible with the new directory.

    - Uses generate_population_counts(pop_size, len(system), len(user))
    - Uses compute_fitness(firm) (expects firm.profit_history)
    - If log_file is None, creates one (old behaviour)
    """

    # Initial population (coerente col tuo generate_population_counts)
    population: List[Individual] = generate_population_counts(
        pop_size, len(system_phrases), len(user_phrases)
    )

    # Old behaviour: always log (create a default log file if not provided)
    if log_file is None:
        log_file = setup_logger("results/logs")

    for gen in range(n_generations):
        print(f"\n Generation {gen + 1}")

        # Decode prompts
        decoded_prompts = [
            decode_individual(ind, system_phrases, user_phrases)
            for ind in population
        ]

        # Run market simulation
        model = MarketModel(decoded_prompts, n_buyers=n_buyers, periods=periods)
        model.run()

        # Compute fitness (coerente con compute_fitness(firm))
        fitnesses: List[float] = [compute_fitness(firm) for firm in model.firms]

        gen_best_fitness = max(fitnesses)
        gen_best_individual = population[fitnesses.index(gen_best_fitness)]

        print(f"Best fitness this generation: {gen_best_fitness:.2f}")
        print(f"All fitnesses: {fitnesses}")

        # Log generation (old behaviour: always logs)
        log_generation_data(log_file, gen + 1, population, fitnesses)

        # Selection
        selected = select_next_generation(
            population, fitnesses, k=tournament_k, elite_size=elite_size
        )

        # Crossover + Mutation to form new population
        next_gen: List[Individual] = selected[:elite_size]  # preserve elite
        while len(next_gen) < pop_size:
            parent1, parent2 = random.sample(selected, 2)
            child = crossover(parent1, parent2)
            mutated_child = mutate(child, mutation_rate)
            next_gen.append(mutated_child)

        population = next_gen

    return population
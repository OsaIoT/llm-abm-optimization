"""
Genetic algorithm operators for evolving prompt configurations.

Legacy-compatible version:
- keep generate_population(pop_size, system_template_phrases, user_template_phrases)
- keep compute_fitness(firm) -> uses firm.profit_history

Also provides optional helper variants:
- generate_population_counts(pop_size, n_system_phrases, n_user_phrases)
- compute_fitness_from_profits(profits)
"""

from __future__ import annotations

import random
from typing import List, Tuple, Iterable, Any

Individual = Tuple[List[int], List[int]]


def generate_individual(n_system_phrases: int, n_user_phrases: int) -> Individual:
    system_bits = [random.choice([0, 1]) for _ in range(n_system_phrases)]
    # keep user bits disabled by default
    user_bits = [0] * n_user_phrases
    return (system_bits, user_bits)


# --- Legacy signature ---
def generate_population(
    pop_size: int,
    system_template_phrases: List[str],
    user_template_phrases: List[str],
) -> List[Individual]:
    """
    Generate a population of unique individuals.

    This keeps the SAME signature and behaviour as the old script:
    - it derives n_system and n_user from the phrase lists
    - it enforces uniqueness via a 'seen' set
    """
    n_system = len(system_template_phrases)
    n_user = len(user_template_phrases)

    population: List[Individual] = []
    seen = set()

    while len(population) < pop_size:
        individual = generate_individual(n_system, n_user)
        individual_key = (tuple(individual[0]), tuple(individual[1]))

        if individual_key not in seen:
            seen.add(individual_key)
            population.append(individual)

    return population


# --- Optional helper ---
def generate_population_counts(pop_size, n_system_phrases, n_user_phrases):
    population = []
    seen = set()
    while len(population) < pop_size:
        individual = generate_individual(n_system_phrases, n_user_phrases)
        key = (tuple(individual[0]), tuple(individual[1]))
        if key not in seen:
            seen.add(key)
            population.append(individual)
    return population


def tournament_selection(population: List[Individual], fitnesses: List[float], k: int = 3) -> List[Individual]:
    selected: List[Individual] = []
    combined = list(zip(population, fitnesses))
    for _ in range(len(population)):
        tournament = random.sample(combined, k)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


def apply_elitism(population: List[Individual], fitnesses: List[float], elite_size: int = 2) -> List[Individual]:
    sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)
    return [ind for ind, _ in sorted_pop[:elite_size]]


def select_next_generation(
    population: List[Individual],
    fitnesses: List[float],
    k: int = 3,
    elite_size: int = 0,
) -> List[Individual]:
    if elite_size == 0:
        return tournament_selection(population, fitnesses, k)
    elite = apply_elitism(population, fitnesses, elite_size)
    rest = tournament_selection(population, fitnesses, k)
    return elite + rest[: len(population) - elite_size]


def crossover(parent1: Individual, parent2: Individual) -> Individual:
    system1, user1 = parent1
    system2, user2 = parent2
    child_system = [random.choice([s1, s2]) for s1, s2 in zip(system1, system2)]
    child_user = [random.choice([u1, u2]) for u1, u2 in zip(user1, user2)]
    return (child_system, child_user)


def mutate(individual: Individual, mutation_rate: float = 0.05) -> Individual:
    system_bits, user_bits = individual
    mutated_system = [
        1 - bit if random.random() < mutation_rate else bit
        for bit in system_bits
    ]
    mutated_user = [
        1 - bit if random.random() < mutation_rate else bit
        for bit in user_bits
    ]
    return (mutated_system, mutated_user)


# --- Legacy signature  ---
def compute_fitness(firm: Any) -> float:
    """
    Legacy-compatible fitness function.
    Expects firm.profit_history, as in the old code.
    """
    profits = getattr(firm, "profit_history", None)
    if profits is None:
        raise AttributeError("compute_fitness expects an object with .profit_history")

    if len(profits) == 0:
        return 0.0

    return float(sum(profits))


# --- Optional helper ---
def compute_fitness_from_profits(profits: Iterable[float]) -> float:
    """Convenience wrapper if you already have a profit iterable."""
    return float(sum(profits))
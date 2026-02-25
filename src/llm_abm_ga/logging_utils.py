"""
Utilities for logging the progress of the genetic algorithm.

This module centralises the creation of log files and the writing of
per-generation summaries. Logs are written in JSON Lines format
(one JSON object per line) to make downstream processing easy with
tools like ``pandas`` or ``jq``.
"""

from __future__ import annotations

import json
import os
from typing import Any, List, Tuple


def setup_logger(log_dir: str) -> str:
    """Create the logging directory if necessary and return a log file path.

    The log file name is based on the directory name and will always
    end in ``evolution_log.jsonl``. If the directory does not exist it
    will be created. You may override the directory by passing a
    different value to this function.

    Args:
        log_dir: Directory where the log file should live.
    Returns:
        The full path to the log file.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "evolution_log.jsonl")
    return log_file


def log_generation_data(
    log_file: str,
    generation: int,
    population: List[Tuple[List[int], List[int]]],
    fitness_scores: List[float],
) -> None:
    """Append a summary of the current generation to the log file.

    Each call writes a single JSON object on its own line. The object
    includes the generation number, the encoded population, raw fitness
    scores and simple summary statistics (mean and best fitness).

    Args:
        log_file: The path to the JSONL file where data should be logged.
        generation: The current generation number (1-indexed).
        population: A list of individuals, where each individual is a tuple of
            two lists of bits (system bits and user bits).
        fitness_scores: Fitness score for each individual in the population.
    """
    if not fitness_scores:
        raise ValueError("Fitness scores list is empty; cannot log generation data.")
    if len(population) != len(fitness_scores):
        raise ValueError(
            "Population and fitness scores must have the same length."
        )
    mean_fitness = sum(fitness_scores) / len(fitness_scores)
    best_fitness = max(fitness_scores)
    best_index = fitness_scores.index(best_fitness)
    data: dict[str, Any] = {
        "generation": generation,
        "population": population,
        "fitness_scores": fitness_scores,
        "mean_fitness": mean_fitness,
        "best_fitness": best_fitness,
        "best_individual": population[best_index],
    }
    with open(log_file, "a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write("\n")

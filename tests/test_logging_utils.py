import json

import pytest

from llm_abm_ga.logging_utils import log_generation_data, setup_logger


def test_generation_summary_roundtrip(tmp_path):
    log = tmp_path / "log.jsonl"
    population = [([1, 0], [0]), ([0, 1], [0])]
    log_generation_data(str(log), 1, population, [1.0, 3.0])
    log_generation_data(str(log), 2, population, [2.0, 2.0])
    first, second = [json.loads(line) for line in log.read_text().splitlines()]
    assert first["generation"] == 1
    assert first["mean_fitness"] == 2.0 and first["best_fitness"] == 3.0
    assert first["best_individual"] == [[0, 1], [0]]
    assert second["generation"] == 2


def test_generation_summary_validates_its_input(tmp_path):
    log = str(tmp_path / "log.jsonl")
    with pytest.raises(ValueError):
        log_generation_data(log, 1, [], [])
    with pytest.raises(ValueError):
        log_generation_data(log, 1, [([1], [0])], [1.0, 2.0])


def test_setup_logger_creates_the_directory(tmp_path):
    path = setup_logger(str(tmp_path / "a" / "b"))
    assert (tmp_path / "a" / "b").is_dir()
    assert path.endswith("evolution_log.jsonl")

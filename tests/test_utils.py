import random

import numpy as np
import pytest

from llm_abm_ga.utils import parse_decision, seed_everything, to_float


@pytest.mark.parametrize(
    "text, expected",
    [("0.5", 0.5), (" 2/4 ", 0.5), ("1", 1.0), ("10/4", 2.5)],
)
def test_to_float_accepts_decimals_and_fractions(text, expected):
    assert to_float(text) == pytest.approx(expected)


def test_to_float_rejects_garbage():
    with pytest.raises(ValueError):
        to_float("abc")


def test_parse_decision_plain():
    assert parse_decision("DECISION: (0.8, 0.7)") == (0.8, 0.7)


def test_parse_decision_accepts_fractions():
    assert parse_decision("DECISION: (3/4, 1/2)") == (0.75, 0.5)


def test_parse_decision_ignores_text_before_the_decision_line():
    assert parse_decision("Let me think about it.\nDECISION: (0.6, 0.4)") == (0.6, 0.4)


def test_parse_decision_accepts_the_bounds():
    assert parse_decision("DECISION: (0.1, 0)") == (0.1, 0.0)
    assert parse_decision("DECISION: (1, 1)") == (1.0, 1.0)


@pytest.mark.parametrize(
    "reply",
    [
        "I would rather not answer",  # no DECISION line
        "DECISION: 0.5, 0.5",  # missing parentheses
        "decision: (0.5, 0.5)",  # the keyword is case-sensitive
        "DECISION: (0.05, 0.5)",  # price below 0.1
        "DECISION: (1.2, 0.5)",  # price above 1
        "DECISION: (0.5, 1.5)",  # quality above 1
        "DECISION: (0.5.1, 0.5)",  # not a number
    ],
)
def test_parse_decision_rejects_invalid_replies(reply):
    with pytest.raises(ValueError):
        parse_decision(reply)


def test_seed_everything_makes_python_and_numpy_reproducible():
    seed_everything(123)
    first = (random.random(), np.random.rand())
    seed_everything(123)
    assert (random.random(), np.random.rand()) == first

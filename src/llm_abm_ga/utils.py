"""
General-purpose utilities for the market simulation and genetic algorithm.

This module contains helper functions for parsing LLM responses and
converting strings to numeric values. Extracting the DECISION line
and validating its format happens here to keep parsing logic separate
from the ``Firm`` class.
"""

from __future__ import annotations

import random
import re
from fractions import Fraction
from typing import Tuple

import numpy as np


def to_float(s: str) -> float:
    """Convert a numeric string into a floating point value.

    This function understands simple fractions (e.g. ``"2/3"`` or ``"10/4"``)
    as well as plain decimal representations. Leading and trailing
    whitespace is ignored.

    Args:
        s: A string representing a number or fraction.
    Returns:
        A floating point number.
    Raises:
        ValueError: If the input string cannot be interpreted as a number.
    """
    s = s.strip()
    if "/" in s:
        # Fractional input such as "2/3" or "10/4".
        return float(Fraction(s))
    return float(s)


# Regular expression matching the DECISION line produced by the LLM.
_DECISION_RE = re.compile(r"DECISION:\s*\(\s*([0-9./]+)\s*,\s*([0-9./]+)\s*\)")


def parse_decision(text: str) -> Tuple[float, float]:
    """Extract a price and quality decision from an LLM response.

    The LLM is expected to include a line of the form ``DECISION: (p, q)``
    where ``p`` and ``q`` are numeric strings. Fractions are supported.
    Values are validated to lie within the expected ranges (price between
    0.1 and 1.0 inclusive, quality between 0 and 1.0 inclusive).

    Args:
        text: The full text response returned by the LLM.
    Returns:
        A tuple ``(price, quality)`` with both values as floats.
    Raises:
        ValueError: If no DECISION line is found, if the line is
            malformed or if either number is out of bounds.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    decision_line = next((ln for ln in lines if ln.startswith("DECISION:")), None)
    if not decision_line:
        raise ValueError(f"No DECISION line found in response: {text!r}")
    match = _DECISION_RE.match(decision_line)
    if not match:
        raise ValueError(f"Malformed DECISION line: {decision_line!r}")
    price_str, quality_str = match.groups()
    price = to_float(price_str)
    quality = to_float(quality_str)
    # Validate bounds
    if not (0.1 <= price <= 1.0):
        raise ValueError(f"Price out of bounds: {price}")
    if not (0.0 <= quality <= 1.0):
        raise ValueError(f"Quality out of bounds: {quality}")
    return price, quality


def seed_everything(seed: int) -> None:
    """Seed Python's and NumPy's random generators.

    The GA operators, the buyers and the rule-based firms all draw from these.
    LLM replies are not covered: hosted models are not fully deterministic,
    even at temperature 0 (see ``experiments/reproducibility``).
    """
    random.seed(seed)
    np.random.seed(seed)

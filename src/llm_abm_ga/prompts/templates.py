"""
Functions for constructing full prompts from genetic encodings.

Genetic individuals consist of two binary lists indicating which phrases
to include from the system and user phrase libraries. This module
assembles those phrases, formats the system prompt with market
parameters and appends mandatory instruction lines.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

from .components import (
    system_template_phrases,
    user_template_phrases,
    MANDATORY_SYSTEM_LINE,
    MANDATORY_USER_LINE,
)


def decode_individual(
    individual: Tuple[List[int], List[int]],
    system_phrases: List[str] = system_template_phrases,
    user_phrases: List[str] = user_template_phrases,
) -> Dict[str, str]:
    """Convert a binary genome into system and user prompt strings.

    The genome is a tuple ``(system_bits, user_bits)``. Each bit indicates
    whether the corresponding phrase in the library should be included.
    Mandatory lines are always appended to ensure constraints and
    formatting requirements are met.

    Args:
        individual: Tuple of lists of bits representing phrase selection.
        system_phrases: Optional override of system phrase library.
        user_phrases: Optional override of user phrase library.
    Returns:
        A dict with ``'system_template'`` and ``'user_template'`` keys
        containing the constructed prompt strings.
    """
    system_bits, user_bits = individual
    # Build the system prompt from selected phrases.
    system_prompt_lines: List[str] = [
        phrase for bit, phrase in zip(system_bits, system_phrases) if bit
    ]
    system_prompt = "\n".join(system_prompt_lines) if system_prompt_lines else ""
    system_prompt += "\n" + MANDATORY_SYSTEM_LINE
    # Build the user prompt from selected phrases. In most cases this will
    # be empty as user phrases are disabled in the default genome.
    user_prompt_lines: List[str] = [
        phrase for bit, phrase in zip(user_bits, user_phrases) if bit
    ]
    user_prompt = "\n".join(user_prompt_lines) if user_prompt_lines else ""
    user_prompt += "\n" + MANDATORY_USER_LINE
    return {
        "system_template": system_prompt,
        "user_template": user_prompt,
    }

"""
Subpackage containing definitions of prompt components and helpers for
constructing full system and user prompts.

Import ``components`` for the base phrase lists and mandatory lines, and
``templates.decode_individual`` to build complete prompt strings from
bit-encoded genomes.
"""

from .components import (
    system_template_phrases,
    user_template_phrases,
    MANDATORY_SYSTEM_LINE,
    MANDATORY_USER_LINE,
)
from .templates import decode_individual

__all__ = [
    "system_template_phrases",
    "user_template_phrases",
    "MANDATORY_SYSTEM_LINE",
    "MANDATORY_USER_LINE",
    "decode_individual",
]

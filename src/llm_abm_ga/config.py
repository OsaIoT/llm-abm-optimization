"""
Centralised configuration for the market simulation and genetic algorithm.

This version supports both OpenAI and Google Gemini:
- Provider, model, and API keys are loaded from .env (if python-dotenv is installed) or from environment
- temperature: 0
"""

from __future__ import annotations

import os

# Load .env automatically if python-dotenv is installed.
# If not installed, it will just use environment variables as-is.
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore

    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass


def _env_flag(name: str, default: str = "0") -> bool:
    """Convert an environment variable into a boolean flag."""
    val = os.getenv(name, default)
    return str(val).lower() in {"1", "true", "yes"}


# ---- LLM Configuration ----
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").lower()
"""Choose the provider: 'openai' or 'gemini'."""

OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
"""API key used to authenticate with OpenAI. Must be provided at runtime."""

GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
"""API key used to authenticate with Google Gemini."""

LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
"""The specific model to use (e.g., 'gpt-4o-mini', 'gemini-2.0-flash')."""

LLM_TEMPERATURE: float = 0.0
"""Fixed temperature used by the project."""

MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "30"))
"""Maximum tokens allowed for the LLM response to prevent reasoning/CoT."""

# Economic parameters
COST_INDEX: float = float(os.getenv("COST_INDEX", "0.75"))
"""The cost multiplier for converting quality into production cost.
Each unit of quality incurs a production cost of ``COST_INDEX * quality``.
"""

BUYER_TEMPERATURE: float = float(os.getenv("BUYER_TEMPERATURE", "0.1")) # has to be > 0
"""Temperature parameter controlling buyer decision noise.
A lower temperature makes buyers more deterministic in selecting the
firm with the highest quality-to-price ratio. A higher temperature
introduces more randomness into the softmax choice.
"""


# ---- Logging configuration ----
ENABLE_LOGGING: bool = _env_flag("ENABLE_LOGGING", "0")
"""Enable detailed logging of firm decisions. Set to ``1`` or ``true`` to enable.
    The output is the copy of the received prompt and the resulting decision of
    each firm, written to results/logs/firm_logs.txt"""
LOG_DIRECTORY: str = os.getenv("LOG_DIRECTORY", "results/logs")


# ---- Genetic algorithm defaults ----
DEFAULT_GENERATIONS: int = int(os.getenv("DEFAULT_GENERATIONS", "30"))
"""Default number of generations for the evolutionary algorithm."""

DEFAULT_POP_SIZE: int = int(os.getenv("DEFAULT_POP_SIZE", "50"))
"""Default population size for the evolutionary algorithm."""

DEFAULT_TOURNAMENT_K: int = int(os.getenv("DEFAULT_TOURNAMENT_K", "6")) # must be smaller than population
"""Default tournament size for selection."""

DEFAULT_ELITE_SIZE: int = int(os.getenv("DEFAULT_ELITE_SIZE", "5"))
"""Default number of elite individuals preserved each generation."""

DEFAULT_MUTATION_RATE: float = float(os.getenv("DEFAULT_MUTATION_RATE", "0.1"))
"""Default mutation rate for flipping bits in the genome."""

DEFAULT_N_BUYERS: int = int(os.getenv("DEFAULT_N_BUYERS", "80000"))
"""Default number of buyers in the market during GA simulations."""

DEFAULT_PERIODS: int = int(os.getenv("DEFAULT_PERIODS", "10"))
"""Default number of time periods in each simulation."""


def require_api_key() -> None:
    """Exit with a helpful message if the selected provider has no API key.

    The entry-point scripts call this up front so that a missing key fails fast
    instead of in the middle of a simulation. Only the key of the provider
    selected through ``LLM_PROVIDER`` is required.
    """
    keys = {"openai": OPENAI_API_KEY, "gemini": GEMINI_API_KEY}
    if LLM_PROVIDER not in keys:
        raise SystemExit(f"Unsupported LLM_PROVIDER '{LLM_PROVIDER}'. Use 'openai' or 'gemini'.")
    if not keys[LLM_PROVIDER]:
        raise SystemExit(
            f"{LLM_PROVIDER.upper()}_API_KEY not set. Put it in .env (project root) "
            "or in the environment variables."
        )

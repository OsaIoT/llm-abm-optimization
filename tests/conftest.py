"""Shared pytest setup. No test needs an API key: the LLM is always stubbed."""

import os

# Plots are saved to files in the tests: never open a window.
os.environ.setdefault("MPLBACKEND", "Agg")

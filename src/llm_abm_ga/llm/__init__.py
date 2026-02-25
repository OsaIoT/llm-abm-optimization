"""
Subpackage providing LLM integration.

Currently this package only exposes a thin wrapper around the OpenAI
API. Additional providers or fallback strategies could be added in
future versions without changing the rest of the codebase.
"""

__all__ = ["client"]

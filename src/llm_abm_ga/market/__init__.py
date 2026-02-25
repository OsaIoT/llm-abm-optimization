"""
Subpackage for the market simulation.

The ``model`` module defines the ``MarketModel`` class which brings
firms and buyers together and orchestrates the simulation over
multiple time periods.
"""

from .model import MarketModel

__all__ = ["MarketModel"]

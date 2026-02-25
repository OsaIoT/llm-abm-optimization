"""
Subpackage containing agent classes for the market simulation.

Agents encapsulate the decision-making logic of firms and buyers. They
interact through the ``MarketModel`` to simulate supply and demand.
"""

from .firm import Firm
from .buyer import Buyer
from .rule_based_firm import RuleBasedFirm

__all__ = ["Firm", "Buyer", "RuleBasedFirm"]
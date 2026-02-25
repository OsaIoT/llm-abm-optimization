"""
Buyer agent for the market simulation.

Each buyer chooses between active firms based on the softmax of
quality-to-price ratios. Buyers do not store state beyond their
identifier.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .firm import Firm
from ..config import BUYER_TEMPERATURE


class Buyer:
    """Represents a single buyer in the market simulation.

    Buyers are identified by an integer ``id`` and make purchasing
    decisions using a softmax over each firm's quality-to-price ratio.
    """

    def __init__(self, buyer_id: int) -> None:
        self.id = buyer_id

    def purchase_decision(
        self,
        firms: Iterable["Firm"],
        *,
        temperature: float = BUYER_TEMPERATURE,
    ) -> Optional[int]:
        """Select a firm to purchase from based on quality-to-price ratio.

        Only firms that are marked as active participate in the choice.
        If no active firms exist or none have valid prices, ``None`` is
        returned indicating that the buyer does not purchase.

        Args:
            firms: An iterable of firm instances.
            temperature: Softmax temperature controlling randomness. A lower
                temperature yields more deterministic choices; a higher
                temperature increases randomness.
        Returns:
            The ``id`` of the selected firm, or ``None`` if no purchase is
            made.
        """
        active_firms = [firm for firm in firms if firm.active]
        if not active_firms:
            return None
        # Compute quality-to-price ratios. Skip firms with non-positive price.
        utility = {firm.id: (firm.q / firm.p) for firm in active_firms if firm.p > 0}
        if not utility:
            return None
        
        # Convert utilities to an array for softmax computation.
        util_values = np.array(list(utility.values()), dtype=float)
        scaled = util_values / temperature
        # Subtract the max for numerical stability.
        exp_util = np.exp(scaled - np.max(scaled))
        probabilities = exp_util / np.sum(exp_util)
        firm_ids = list(utility.keys())
        choice_index = np.random.choice(len(firm_ids), p=probabilities)
        return firm_ids[choice_index]

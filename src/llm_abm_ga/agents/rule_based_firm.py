"""
Rule-based baseline firm.
Decides (p, q) following simple heuristics based on past profit.
No LLM is involved.
"""

from __future__ import annotations
import os
import numpy as np
from typing import Dict, Any

from ..config import COST_INDEX, ENABLE_LOGGING, LOG_DIRECTORY

class RuleBasedFirm:
    # ---------- TUNABLE PARAMETERS ----------
    PRICE_MIN = 0.1
    PRICE_MAX = 1.0
    QUALITY_MIN = 0.0
    QUALITY_MAX = 1.0

    INIT_Q_LOW = 0.3
    INIT_Q_HIGH = 0.7
    INIT_MARGIN_LOW = 0.05
    INIT_MARGIN_HIGH = 0.2

    PRICE_STEP_LOSS = 0.05    
    PRICE_STEP_ZERO = -0.02   
    QUALITY_STEP_ZERO = 0.05  

    NOISE_PRICE = 0.01
    NOISE_QUALITY = 0.01

    ZERO_TOL = 1e-6
    MAX_NEGATIVE_STREAK = 3

    def __init__(
        self, 
        firm_id: int, 
        n_competitors: int, 
        n_buyers: int, 
        total_periods: int,
        cost_index: float = COST_INDEX
    ) -> None:
        self.id = firm_id
        self.n_competitors = n_competitors
        self.n_buyers = n_buyers
        self.total_periods = total_periods
        self.cost_index = cost_index

        # Initialize p,q: price = cost + margin
        q0 = np.random.uniform(self.INIT_Q_LOW, self.INIT_Q_HIGH)
        margin0 = np.random.uniform(self.INIT_MARGIN_LOW, self.INIT_MARGIN_HIGH)
        p0 = self.cost_index * q0 + margin0

        self.q = float(np.clip(q0, self.QUALITY_MIN, self.QUALITY_MAX))
        self.p = float(np.clip(p0, self.PRICE_MIN, self.PRICE_MAX))
        self.cost = self.cost_index * self.q

        # State Variables (compatible with Firm)
        self.last_period_sold = 0
        self.sold = 0
        self.revenue = 0.0
        self.last_period_profit = 0.0
        self.cumulative_profit = 0.0
        self.last_period_revenue = 0.0

        # History Tracking
        self.price_history = [self.p]
        self.quality_history = [self.q]
        self.sold_history = []
        self.profit_history = []

        self.negative_streak = 0
        self.active = True

    def deactivate(self, reason: str, current_period: int = None) -> None:
        """Fallback exit logic matching the LLM firm."""
        if not self.active:
            return
        self.active = False
        msg = f"RuleFirm {self.id} excluded from the market"
        if current_period is not None:
            msg += f" (period {current_period})"
        msg += f": {reason}"
        print(msg)
        
        if ENABLE_LOGGING:
            os.makedirs(LOG_DIRECTORY, exist_ok=True)
            log_path = os.path.join(LOG_DIRECTORY, "firm_logs.txt")
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(msg + "\n")
                log_file.write("=" * 80 + "\n\n")

    def _clip_and_update_cost(self) -> None:
        self.p = float(np.clip(self.p, self.PRICE_MIN, self.PRICE_MAX))
        self.q = float(np.clip(self.q, self.QUALITY_MIN, self.QUALITY_MAX))
        self.cost = self.cost_index * self.q

    def make_decision(self, market_info: Dict[str, Any]) -> None:
        if not self.active:
            return

        current_period = market_info["current_period"]

        # First period: keep initialized p0, q0
        if current_period == 1 or len(self.profit_history) == 0:
            self._clip_and_update_cost()
            return

        profit = self.last_period_profit
        noise_p = np.random.uniform(-self.NOISE_PRICE, self.NOISE_PRICE)
        noise_q = np.random.uniform(-self.NOISE_QUALITY, self.NOISE_QUALITY)

        if profit < -self.ZERO_TOL:
            self.p += self.PRICE_STEP_LOSS + noise_p
            self.q -= self.QUALITY_STEP_ZERO + noise_q
        elif abs(profit) <= self.ZERO_TOL:
            self.p += self.PRICE_STEP_ZERO + noise_p
            self.q += self.QUALITY_STEP_ZERO + noise_q
        else:
            self.p += noise_p
            self.q += noise_q

        self._clip_and_update_cost()
        self.price_history.append(self.p)
        self.quality_history.append(self.q)
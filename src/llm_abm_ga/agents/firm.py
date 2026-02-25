"""
Firm agent for the market simulation.

A firm represents a producer in the market who sets a price and
quality for their product each period. Firms receive feedback via
sales and profits and can exit the market if they experience
consecutive periods of negative profit. Decision making is delegated to
an LLM through the ``generate_answer`` function, and responses are
parsed using utilities in ``llm_abm_ga.utils``.
"""

from __future__ import annotations

import os
from typing import Dict, Any, List

import numpy as np

from ..config import COST_INDEX, ENABLE_LOGGING, LOG_DIRECTORY
from ..llm.client import generate_answer
from ..utils import parse_decision


class Firm:
    def __init__(
        self,
        firm_id: int,
        system_prompt_template: str,
        user_prompt_template: str,
        n_competitors: int,
        n_buyers: int,
        total_periods: int,
        cost_index: float = COST_INDEX,
    ) -> None:
        # (mantengo struttura simile al vecchio)
        self.cost_index = cost_index
        self.id = firm_id
        self.n_competitors = n_competitors
        self.n_buyers = n_buyers
        self.total_periods = total_periods

        # Format system prompt
        self.system_prompt = system_prompt_template.format(
            n_competitors=self.n_competitors,
            n_buyers=self.n_buyers,
            cost_index=self.cost_index,
        )
        self.user_prompt_template = user_prompt_template

        # State Variables
        self.p, self.q, self.cost = 0.0, 0.0, 0.0
        self.last_period_sold = 0
        self.sold = 0
        self.revenue = 0.0
        self.last_period_profit = 0.0
        self.cumulative_profit = 0.0
        self.last_period_revenue = 0.0

        # History Tracking
        self.price_history: List[float] = []
        self.quality_history: List[float] = []
        self.sold_history: List[int] = []
        self.profit_history: List[float] = []

        self.negative_streak = 0
        self.active = True

    def make_decision(self, market_info: Dict[str, Any]) -> None:
        # already out -> skip
        if not self.active:
            return

        user_prompt = self.user_prompt_template.format(
            firm_id=self.id,
            current_period=market_info["current_period"],
            total_periods=self.total_periods,
            price_history=self.price_history,
            quality_history=self.quality_history,
            sold_history=self.sold_history,
            last_period_profit=f"{self.last_period_profit:.2f}",
            cumulative_profit=f"{self.cumulative_profit:.2f}",
            competitor_prices=market_info["competitor_prices"],
            competitor_qualities=market_info["competitor_qualities"],
            competitor_sales=market_info["competitor_sales"],
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = generate_answer(messages).strip()

            # Parse decision (supports fractions) + bounds validation
            # (expects: DECISION: (price, quality))
            from ..utils import parse_decision  # local import to avoid circular issues
            price, quality = parse_decision(response)

            # success → update state
            self.p = price
            self.q = quality
            self.cost = self.cost_index * self.q

            # logging/explanation: follow old logic (capture anything before DECISION)
            lines = response.splitlines()
            decision_idx = next(
                (i for i, ln in enumerate(lines) if ln.strip().startswith("DECISION:")),
                None,
            )
            explanation = "\n".join(lines[:decision_idx]) if decision_idx is not None else ""

            if ENABLE_LOGGING:
                self.log_decision(
                    market_info=market_info,
                    user_prompt=user_prompt,
                    explanation=explanation,
                )

            # track history ONLY on valid decision (old behavior)
            self.price_history.append(self.p)
            self.quality_history.append(self.q)

        except Exception as e:
            # parsing (or API) failed -> firm exits market (as requested)
            print(f"Error parsing LLM output for Firm {self.id}: {e}")
            self.active = False
            # clear current decision so it cannot be used downstream
            self.p, self.q, self.cost = 0.0, 0.0, 0.0
            print(f"Firm {self.id} has been eliminated from the market (invalid LLM output).")

            return

    def log_decision(self, market_info: Dict[str, Any], user_prompt: str, explanation: str) -> None:
        os.makedirs(LOG_DIRECTORY, exist_ok=True)
        log_path = os.path.join(LOG_DIRECTORY, "firm_logs.txt")

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Firm {self.id}, Period {market_info['current_period']}:\n")
            log_file.write(f"System Prompt:\n{self.system_prompt}\n")
            log_file.write(f"User Prompt:\n{user_prompt}\n")
            log_file.write(f"Explanation:\n{explanation}\n")
            log_file.write(f"Decision: ({self.p}, {self.q})\n")
            log_file.write("=" * 80 + "\n\n")


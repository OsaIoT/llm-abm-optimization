"""
Market model orchestrating interactions between firms and buyers.

The ``MarketModel`` class manages a collection of firms and buyers and
simulates their interactions over a series of time periods. Firms make
pricing and quality decisions, buyers select firms based on
quality-to-price ratios, and profits are computed accordingly. Firms
may exit the market after consecutive losses.
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from ..agents.firm import Firm
from ..agents.buyer import Buyer
from ..agents.rule_based_firm import RuleBasedFirm


class MarketModel:
    """Simulate a competitive market with multiple firms and buyers."""

    def __init__(
        self,
        firm_prompt_configs: Optional[List[Dict[str, str]]] = None,
        *,
        n_rule_firms: int = 0, # Add parameter for rule-based firms
        n_buyers: int = 10,
        periods: int = 10,
    ) -> None:
        
        firm_prompt_configs = firm_prompt_configs or []
        self.n_llm_firms = len(firm_prompt_configs)
        self.n_rule_firms = n_rule_firms
        self.n_firms = self.n_llm_firms + self.n_rule_firms
        self.n_buyers = n_buyers
        self.T = periods
        self.t = 0

        self.firms = []
        current_id = 0

        # Initialize LLM Firms
        for config in firm_prompt_configs:
            self.firms.append(
                Firm(
                    firm_id=current_id,
                    system_prompt_template=config["system_template"],
                    user_prompt_template=config["user_template"],
                    n_competitors=self.n_firms - 1,
                    n_buyers=self.n_buyers,
                    total_periods=self.T,
                )
            )
            current_id += 1

        # Initialize Rule-Based Firms
        for _ in range(self.n_rule_firms):
            self.firms.append(
                RuleBasedFirm(
                    firm_id=current_id,
                    n_competitors=self.n_firms - 1,
                    n_buyers=self.n_buyers,
                    total_periods=self.T,
                )
            )
            current_id += 1

        self.buyers: List[Buyer] = [Buyer(i) for i in range(n_buyers)]

    def run(self) -> None:
        """Execute the market simulation over ``self.T`` periods."""
        for _ in range(self.T):
            print(f"Time Period {self.t + 1}")

            firm_by_id = {f.id: f for f in self.firms}

            # Active firms (used both for competitor info and for decisions)
            active_firms = [firm for firm in self.firms if firm.active]
            if not active_firms:
                print("Collapsed market: no active firms.")
                return

            # Market info prior to firm decisions (only active firms)
            competitor_prices = [firm.p for firm in active_firms]
            competitor_qualities = [firm.q for firm in active_firms]
            competitor_sales = [firm.last_period_sold for firm in active_firms]
            market_info: Dict[str, Any] = {
                "competitor_prices": competitor_prices,
                "competitor_qualities": competitor_qualities,
                "competitor_sales": competitor_sales,
                "current_period": self.t + 1,
            }

            # Firms make decisions in parallel
            with ThreadPoolExecutor(max_workers=len(active_firms)) as executor:
                futures = [
                    executor.submit(firm.make_decision, market_info) for firm in active_firms
                ]
                for future in futures:
                    future.result()

            # Reset per-period stats before buyer purchases
            for firm in self.firms:
                firm.last_period_sold = 0
                firm.last_period_profit = 0.0
                firm.last_period_revenue = 0.0

            # Buyer purchases (only among active firms)
            for buyer in self.buyers:
                active_firms = [firm for firm in self.firms if firm.active]
                if not active_firms:
                    print("Collapsed market: no active firms.")
                    return

                shuffled_firms = random.sample(active_firms, len(active_firms))
                choice = buyer.purchase_decision(shuffled_firms)

                if choice is not None:
                    selected_firm = firm_by_id[choice]
                    selected_firm.sold += 1
                    selected_firm.last_period_sold += 1
                    selected_firm.revenue += selected_firm.p
                    selected_firm.last_period_revenue += selected_firm.p
                    selected_firm.cumulative_profit += selected_firm.p - selected_firm.cost

            # Update profits, histories, and negative streaks
            for firm in self.firms:
                if firm.active:
                    firm.last_period_profit = (
                        firm.last_period_revenue - (firm.cost * firm.last_period_sold)
                    )
                    firm.profit_history.append(firm.last_period_profit)
                    firm.sold_history.append(firm.last_period_sold)

                    if firm.last_period_profit < 0:
                        firm.negative_streak += 1
                    else:
                        firm.negative_streak = 0

                    if firm.negative_streak >= 3:
                        firm.active = False
                        print(f"Firm {firm.id} has exited the market (3 consecutive losses).")

            self.t += 1
import random

import numpy as np
import pytest

from llm_abm_ga.agents import firm as firm_mod
from llm_abm_ga.market.model import MarketModel

MARKET_INFO = {
    "competitor_prices": [],
    "competitor_qualities": [],
    "competitor_sales": [],
    "current_period": 1,
}


def test_rule_based_market_runs_and_records_history():
    random.seed(0)
    np.random.seed(0)
    model = MarketModel(n_rule_firms=3, n_buyers=300, periods=5)
    model.run()
    assert len(model.firms) == 3
    assert 0 < sum(f.sold for f in model.firms) <= 300 * 5
    for f in model.firms:
        assert len(f.profit_history) == len(f.sold_history)
        if f.active:
            assert len(f.profit_history) == 5


def test_firm_exits_after_three_consecutive_losses():
    model = MarketModel(n_rule_firms=1, n_buyers=50, periods=6)
    firm = model.firms[0]
    firm.make_decision = lambda market_info: None  # freeze the strategy
    firm.p, firm.q, firm.cost = 0.2, 1.0, 0.75  # sells below cost
    model.run()
    assert firm.active is False
    assert firm.profit_history == [pytest.approx(-0.55 * 50)] * 3


def make_llm_firm():
    return firm_mod.Firm(
        firm_id=0,
        system_prompt_template="{n_competitors} {n_buyers} {cost_index}",
        user_prompt_template=" ",
        n_competitors=3,
        n_buyers=100,
        total_periods=10,
    )


def test_llm_firm_parses_the_decision_and_derives_its_cost(monkeypatch):
    monkeypatch.setattr(firm_mod, "generate_answer", lambda messages: "DECISION: (0.8, 0.6)")
    f = make_llm_firm()
    f.make_decision(MARKET_INFO)
    assert (f.p, f.q) == (0.8, 0.6)
    assert f.cost == pytest.approx(0.75 * 0.6)
    assert f.active


@pytest.mark.parametrize("reply", ["I refuse to answer", "DECISION: (5, 0.5)"])
def test_llm_firm_exits_the_market_on_an_invalid_reply(monkeypatch, reply):
    monkeypatch.setattr(firm_mod, "generate_answer", lambda messages: reply)
    f = make_llm_firm()
    f.make_decision(MARKET_INFO)
    assert f.active is False
    assert (f.p, f.q, f.cost) == (0.0, 0.0, 0.0)


def test_llm_firm_exits_when_the_api_call_fails(monkeypatch):
    def boom(messages):
        raise RuntimeError("network down")

    monkeypatch.setattr(firm_mod, "generate_answer", boom)
    f = make_llm_firm()
    f.make_decision(MARKET_INFO)
    assert f.active is False

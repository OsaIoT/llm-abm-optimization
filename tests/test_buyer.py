from types import SimpleNamespace

import numpy as np

from llm_abm_ga.agents.buyer import Buyer


def firm(firm_id, price, quality, active=True):
    return SimpleNamespace(id=firm_id, p=price, q=quality, active=active)


def test_buyer_prefers_the_best_quality_to_price_ratio():
    np.random.seed(0)
    buyer = Buyer(0)
    firms = [firm(0, 0.5, 0.9), firm(1, 0.9, 0.5)]  # ratios 1.8 vs 0.56
    picks = [buyer.purchase_decision(firms) for _ in range(300)]
    assert picks.count(0) >= 295


def test_buyer_ignores_inactive_firms():
    firms = [firm(0, 0.5, 0.9, active=False), firm(1, 0.9, 0.5)]
    assert {Buyer(0).purchase_decision(firms) for _ in range(20)} == {1}


def test_buyer_does_not_buy_without_a_valid_offer():
    assert Buyer(0).purchase_decision([]) is None
    assert Buyer(0).purchase_decision([firm(0, 0.0, 0.5)]) is None  # non-positive price
    assert Buyer(0).purchase_decision([firm(0, 0.5, 0.5, active=False)]) is None


def test_high_temperature_makes_choices_more_random():
    np.random.seed(0)
    firms = [firm(0, 0.5, 0.9), firm(1, 0.9, 0.5)]
    picks = {Buyer(0).purchase_decision(firms, temperature=100.0) for _ in range(200)}
    assert picks == {0, 1}

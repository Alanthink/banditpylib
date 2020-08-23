import numpy as np
import pytest

from .ordinary_mnl_bandit import OrdinaryMNLBandit, \
    search, search_best_assortment, MeanReward, CvarReward


class TestOrdinaryMNLBandit:
  """Tests in ordinary mnl bandit"""

  def test_search_unrestricted(self):
    results = []
    search(results, 3, 1, [])
    assert results == [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3]]

  def test_search_restricted(self):
    results = []
    search(results, 3, 1, [], 1)
    assert results == [[1], [2], [3]]

  def test_search_best_assortment(self):
    reward = MeanReward()
    reward.set_abstraction_params(np.array([1, 1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 0.45, 0.8, 0.9, 1.0]))
    best_revenue, best_assortment = search_best_assortment(product_num=4,
                                                           reward=reward)
    assert best_assortment == [2, 3, 4]
    assert best_revenue == pytest.approx(0.675, 1e-8)

  def test_cvar_calculation(self):
    reward = CvarReward(alpha=0.5)
    reward.set_abstraction_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # calculate cvar under percentile 0.25
    cvar_alpha = reward.calc([2, 3])
    # upper bound of cvar is 1
    assert cvar_alpha <= 1

    # equivalent to MeanReward
    reward = CvarReward(alpha=2.0)
    reward.set_abstraction_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # calculate cvar under percentile 0.25
    cvar_alpha = reward.calc([1, 2, 3])
    # upper bound of cvar is 1
    assert cvar_alpha == 0.75

  def test_regret(self):
    abstraction_params = np.array(
        [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5])
    revenues = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    card_limit = 1
    bandit = OrdinaryMNLBandit(abstraction_params, revenues, card_limit)
    bandit.reset()
    # serve best assortment [1] for 3 times
    bandit.feed([([1], 3)])
    assert bandit.regret() == 0.0

  def test_one_product(self):
    abstraction_params = [1.0, 0.0]
    revenues = [0.0, 1.0]
    bandit = OrdinaryMNLBandit(abstraction_params, revenues)
    bandit.reset()
    # always get no purchase
    assert set(bandit.feed([([1], 5)])[0][1]) == {0}

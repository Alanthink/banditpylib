import numpy as np
import pytest

import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Actions
from banditpylib.learners import MaxReward
from .ordinary_mnl_bandit import OrdinaryMNLBandit, \
    search, search_best_assortment, MeanReward, CvarReward, \
    local_search_best_assortment


class TestOrdinaryMNLBandit:
  """Tests in ordinary mnl bandit"""
  def test_search_unrestricted(self):
    results = []
    search(assortments=results,
           product_num=3,
           next_product_id=1,
           assortment=set())
    assert results == [{1, 2, 3}, {1, 2}, {1, 3}, {1}, {2, 3}, {2}, {3}]

  def test_search_restricted(self):
    results = []
    search(assortments=results,
           product_num=3,
           next_product_id=1,
           assortment=set(),
           card_limit=1)
    assert results == [{1}, {2}, {3}]

  def test_search_best_assortment(self):
    reward = MeanReward()
    reward.set_preference_params(np.array([1, 1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 0.45, 0.8, 0.9, 1.0]))
    best_revenue, best_assortment = search_best_assortment(reward=reward)
    assert best_assortment == {2, 3, 4}
    assert best_revenue == pytest.approx(0.675, 1e-8)

    reward = CvarReward(0.7)
    reward.set_preference_params(np.array([1, 0.7, 0.8, 0.5, 0.2]))
    reward.set_revenues(np.array([0, 0.7, 0.8, 0.9, 1.0]))
    best_revenue, best_assortment = search_best_assortment(reward=reward)
    assert best_assortment == {1, 2, 3, 4}
    assert best_revenue == pytest.approx(0.41, 1e-2)

  def test_local_search_best_assortment(self):
    reward = MeanReward()
    reward.set_preference_params(
        np.array([
            1, 0.41796638065213765, 0.640233815546399, 0.8692331344533714,
            0.6766260654759216, 0.7388897283940284
        ]))
    reward.set_revenues(
        np.array([
            0, 0.6741252008863028, 0.16873112382668315, 0.3056694804088398,
            0.9369761261650644, 0.9444412097177997
        ]))
    product_num = len(reward.revenues) - 1
    _, best_assortment = search_best_assortment(reward=reward, card_limit=3)
    _, local_best_assortment = local_search_best_assortment(
        reward=reward,
        random_neighbors=10,
        card_limit=3,
        init_assortment=best_assortment)
    assert set(local_best_assortment).issubset(set(range(1, product_num + 1)))
    # When reward is MeanReward, local search guarantees to output the best
    # assortment.
    assert best_assortment == local_best_assortment

  def test_cvar_calculation(self):
    reward = CvarReward(alpha=0.5)
    reward.set_preference_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # Calculate cvar under percentile 0.25
    cvar_alpha = reward.calc({2, 3})
    # Upper bound of cvar is 1
    assert cvar_alpha <= 1

    # Equivalent to MeanReward
    reward = CvarReward(alpha=1.0)
    reward.set_preference_params(np.array([1, 1, 1, 1]))
    reward.set_revenues(np.array([0, 1, 1, 1]))
    # Calculate cvar under percentile 0.25
    cvar_alpha = reward.calc({1, 2, 3})
    # Upper bound of cvar is 1
    assert cvar_alpha == 0.75

  def test_regret(self):
    preference_params = np.array(
        [1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5])
    revenues = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    card_limit = 1
    bandit = OrdinaryMNLBandit(preference_params, revenues, card_limit)
    bandit.reset()
    # Serve best assortment {1} for 3 times
    bandit.feed(
        text_format.Parse(
            """
      arm_pulls_pairs {
        arm {
          set {
            id: 1
          }
        }
        pulls: 3
      }
      """, Actions()))
    assert bandit.regret(MaxReward()) == 0.0

  def test_one_product(self):
    preference_params = [1.0, 0.0]
    revenues = [0.0, 1.0]
    bandit = OrdinaryMNLBandit(preference_params, revenues)
    bandit.reset()
    # Always get no purchase
    assert set(
        bandit.feed(
            text_format.Parse(
                """
      arm_pulls_pairs {
        arm {
          set {
            id: 1
          }
        }
        pulls: 5
      }
      """, Actions())).arm_rewards_pairs[0].customer_feedbacks) == {0}

from unittest.mock import MagicMock

import numpy as np

from banditpylib.bandits import MeanReward
from .eps_greedy import EpsGreedy


class TestEpsGreedy:
  """Test epsilon greedy policy"""
  def test_random_assortment(self):
    revenues = np.array([0, 0.7, 0.8, 0.9, 1.0])
    horizon = 100
    reward = MeanReward()
    card_limit = 3
    learner = EpsGreedy(revenues=revenues,
                        horizon=horizon,
                        reward=reward,
                        card_limit=card_limit)
    random_assortment = learner.select_ramdom_assort()
    assert random_assortment != []
    assert len(random_assortment) <= card_limit

  def test_simple_run(self):
    revenues = np.array([0, 0.45, 0.8, 0.9, 1.0])
    horizon = 100
    reward = MeanReward()
    learner = EpsGreedy(revenues=revenues, horizon=horizon, reward=reward)
    mock_random_assortment = [2, 3, 4]
    learner.select_ramdom_assort = MagicMock(
        return_value=mock_random_assortment)
    learner.reset()
    assert learner.actions() == [([2, 3, 4], 1)]

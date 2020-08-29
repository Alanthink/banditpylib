from unittest.mock import MagicMock

import numpy as np

from banditpylib.bandits import CvarReward
from .ucb import UCB


class TestUCB:
  """Test UCB policy"""
  def test_simple_run(self):
    revenues = np.array([0, 0.7, 0.8, 0.9, 1.0])
    horizon = 100
    reward = CvarReward(0.7)
    learner = UCB(revenues=revenues, horizon=horizon, reward=reward)

    learner.reset()
    mock_preference_params = np.array([1, 1, 1, 1, 1])
    learner.UCB = MagicMock(return_value=mock_preference_params)
    assert learner.actions() == [([1, 2, 3, 4], 1)]

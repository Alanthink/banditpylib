from unittest.mock import MagicMock

import numpy as np

from banditpylib.bandits import CvarReward
from .ucb import RiskAwareUCB


class TestRiskAwareUCB:
  """Test risk-aware UCB policy"""
  def test_simple_run(self):
    revenues = np.array([0, 0.7, 0.8, 0.9, 1.0])
    horizon = 100
    reward = CvarReward(0.7)
    learner = RiskAwareUCB(revenues=revenues, horizon=horizon, reward=reward)

    learner.reset()
    mock_abstraction_params = np.array([1, 1, 1, 1, 1])
    learner.UCB = MagicMock(return_value=mock_abstraction_params)
    assert learner.actions() == [([2, 3, 4], 1)]

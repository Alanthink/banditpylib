from unittest.mock import MagicMock

import numpy as np

from banditpylib.bandits import CvarReward
from .ts import ThompsonSampling


class TestThompsonSampling:
  """Test thompson sampling policy"""
  def test_simple_run(self):
    revenues = np.array([0, 0.7, 0.8, 0.9, 1.0])
    horizon = 100
    reward = CvarReward(0.7)
    learner = ThompsonSampling(revenues=revenues,
                               horizon=horizon,
                               reward=reward)

    # test warm start
    learner.reset()
    assert learner.actions() == [({1}, 1)]

    learner.reset()
    learner.within_warm_start = MagicMock(return_value=False)
    mock_preference_params = np.array([1, 1, 1, 1, 1])
    learner.correlated_sampling = MagicMock(
        return_value=mock_preference_params)
    assert learner.actions() == [({1, 2, 3, 4}, 1)]

from unittest.mock import MagicMock

import numpy as np

from .ucb import UCB


class TestUCB:
  """Test UCB policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 30
    learner = UCB(arm_num=arm_num)
    learner.reset()
    mock_ucb = np.array([1.2, 1, 1, 1, 1])
    learner.UCB = MagicMock(return_value=mock_ucb)

    # during the first 5 time steps, each arm is pulled once
    for time in range(1, arm_num + 1):
      assert learner.actions() == [((time - 1) % arm_num, 1)]
      learner.update(([np.array([0])], ))
    # for the left time steps, arm 0 is always the choice
    for _ in range(arm_num + 1, horizon + 1):
      assert learner.actions() == [(0, 1)]
      learner.update(([np.array([0])], ))

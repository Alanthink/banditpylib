from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.bandits import LinearBandit
from .linucb import LinUCB


class TestLinUCB:
  """Test LinUCB Policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 10
    features = [
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([0, 1])
    ]
    learner = LinUCB(arm_num, features, 0.1, 1e-3)
    learner.reset()
    mock_ucb = np.array([1.2, 1, 1, 1, 1])
    # pylint: disable=protected-access
    learner._LinUCB__LinUCB = MagicMock(return_value=mock_ucb)

    # Always 0th arm is picked
    # not the most efficient test
    for time in range(1, horizon + 1):
      assert learner.actions().SerializeToString() == text_format.Parse(
          """
            arm_pulls_pairs <
              arm <
                id: 0
              >
              pulls: 1
            >
            """, Actions()).SerializeToString()
      learner.update(
          text_format.Parse(
              """
            arm_rewards_pairs <
              arm <
                id: 0
              >
              rewards: 0
            >
            """, Feedback()))

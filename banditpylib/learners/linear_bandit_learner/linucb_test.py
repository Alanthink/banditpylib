from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.data_pb2 import Context, Actions, Feedback
from .linucb import LinUCB


class TestLinUCB:
  """Test LinUCB Policy"""
  def test_simple_run(self):
    horizon = 10
    features = [
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([1, 0]),
        np.array([0, 1])
    ]
    learner = LinUCB(features, 0.1, 1e-3)
    learner.reset()
    mock_ucb = np.array([1.2, 1, 1, 1, 1])
    # pylint: disable=protected-access
    learner._LinUCB__LinUCB = MagicMock(return_value=mock_ucb)

    # Always 0th arm is picked
    # not the most efficient test
    for _ in range(1, horizon + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
              """
            arm_pulls <
              arm <
                id: 0
              >
              times: 1
            >
            """, Actions()).SerializeToString()
      learner.update(
          text_format.Parse(
              """
            arm_feedbacks <
              arm <
                id: 0
              >
              rewards: 0
            >
            """, Feedback()))

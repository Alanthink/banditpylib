from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.data_pb2 import Context, Actions, Feedback
from .ucbv import UCBV


class TestUCBV:
  """Test UCBV policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 30
    learner = UCBV(arm_num=arm_num)
    learner.reset()
    mock_ucbv = np.array([1.2, 1, 1, 1, 1])
    # pylint: disable=protected-access
    learner._UCBV__UCBV = MagicMock(return_value=mock_ucbv)

    # during the first 5 time steps, each arm is pulled once
    for time in range(1, arm_num + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
              """
        arm_pulls_pairs <
          arm <
            id: {arm_id}
          >
          pulls: 1
        >
        """.format(arm_id=time - 1), Actions()).SerializeToString()
      learner.update(
          text_format.Parse(
              """
        arm_feedbacks <
          arm <
            id: {arm_id}
          >
          rewards: 0
        >
        """.format(arm_id=time - 1), Feedback()))
    # for the left time steps, arm 0 is always the choice
    for _ in range(arm_num + 1, horizon + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
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
        arm_feedbacks <
          arm <
            id: 0
          >
          rewards: 0
        >
        """, Feedback()))

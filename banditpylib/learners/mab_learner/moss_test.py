from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.data_pb2 import Context, Actions, Feedback
from .moss import MOSS


class TestMOSS:
  """Test MOSS policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 30
    learner = MOSS(arm_num=arm_num, horizon=horizon)
    learner.reset()
    mock_moss = np.array([1.2, 1, 1, 1, 1])
    # pylint: disable=protected-access
    learner._MOSS__MOSS = MagicMock(return_value=mock_moss)

    # during the first 5 time steps, each arm is pulled once
    for time in range(1, arm_num + 1):
      assert learner.actions(
          Context()).SerializeToString() == text_format.Parse(
              """
        arm_pulls <
          arm <
            id: {arm_id}
          >
          times: 1
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

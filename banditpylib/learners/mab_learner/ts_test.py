from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

from banditpylib.data_pb2 import Context, Actions
from .ts import ThompsonSampling


class TestThompsonSampling:
  """Test thompson sampling policy"""
  def test_simple_run(self):
    ts_learner = ThompsonSampling(arm_num=4)
    ts_learner.reset()
    # pylint: disable=protected-access
    ts_learner._ThompsonSampling__sample_from_beta_prior = MagicMock(
        return_value=1)
    # always pull arm 1
    assert ts_learner.actions(
        Context()).SerializeToString() == text_format.Parse(
            """
        arm_pulls <
          arm <
            id: 1
          >
          times: 1
        >
        """, Actions()).SerializeToString()

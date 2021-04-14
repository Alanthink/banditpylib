from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.bandits import CvarReward
from banditpylib.data_pb2 import Actions
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

    # Test warm start
    learner.reset()
    assert learner.actions().SerializeToString() == text_format.Parse(
        """
      arm_pulls_pairs {
        arm {
          ids: 1
        }
        pulls: 1
      }
      """, Actions()).SerializeToString()

    learner.reset()
    # pylint: disable=protected-access
    learner._ThompsonSampling__within_warm_start = MagicMock(
        return_value=False)
    mock_preference_params = np.array([1, 1, 1, 1, 1])
    learner._ThompsonSampling__correlated_sampling = MagicMock(
        return_value=mock_preference_params)
    assert learner.actions().SerializeToString() == text_format.Parse(
        """
      arm_pulls_pairs {
        arm {
          ids: 1
          ids: 2
          ids: 3
          ids: 4
        }
        pulls: 1
      }
      """, Actions()).SerializeToString()

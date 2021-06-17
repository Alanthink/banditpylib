from unittest.mock import MagicMock

import google.protobuf.text_format as text_format

import numpy as np

from banditpylib.bandits import MeanReward
from banditpylib.data_pb2 import Actions, Context
from .eps_greedy import EpsGreedy


class TestEpsGreedy:
  """Test epsilon greedy policy"""
  def test_random_assortment(self):
    revenues = np.array([0, 0.7, 0.8, 0.9, 1.0])
    reward = MeanReward()
    card_limit = 3
    learner = EpsGreedy(revenues=revenues,
                        reward=reward,
                        card_limit=card_limit)
    # pylint: disable=protected-access
    random_assortment = learner._EpsGreedy__select_ramdom_assort()
    assert random_assortment != set()
    assert len(random_assortment) <= card_limit

  def test_simple_run(self):
    revenues = np.array([0, 0.45, 0.8, 0.9, 1.0])
    reward = MeanReward()
    learner = EpsGreedy(revenues=revenues, reward=reward)
    learner.reset()
    mock_random_assortment = {2, 3, 4}
    # pylint: disable=protected-access
    learner._EpsGreedy__select_ramdom_assort = MagicMock(
        return_value=mock_random_assortment)
    assert learner.actions(Context()).SerializeToString() == text_format.Parse(
        """
      arm_pulls_pairs {
        arm {
          set {
            id: 2
            id: 3
            id: 4
          }
        }
        pulls: 1
      }
      """, Actions()).SerializeToString()

import numpy as np

from banditpylib.learners import BestArmId
from .linear_bandit import LinearBandit


class TestLinearBandit:
  """Test linear bandit"""
  def test_linear_bandit(self):
    features = np.array([[0, 1], [1, 0]])
    theta = np.array([1, 0])
    linear_bandit = LinearBandit(features, theta)
    linear_bandit.reset()
    assert linear_bandit.regret(BestArmId(best_arm=0)) == 1

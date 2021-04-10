import numpy as np

from banditpylib.arms import BernoulliArm
from .eps_greedy import EpsGreedy


class TestEpsGreedy:
  """Test epsilon greedy policy"""
  def test_simple_run(self):
    means = [0, 0.5, 0.7, 1]
    arms = [BernoulliArm(mean) for mean in means]
    learner = EpsGreedy(arm_num=len(arms))
    learner.reset()

    for arm_id in range(len(arms)):
      assert learner.actions() == [(arm_id, 1)]
      learner.update(([np.array([0])], ))

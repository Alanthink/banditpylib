import numpy as np

from .uniform import Uniform


class TestUniform:
  """Test uniform policy"""
  def test_simple_run(self):
    arm_num = 5
    horizon = 30
    learner = Uniform(arm_num=arm_num, horizon=horizon)
    learner.reset()

    for time in range(1, horizon + 1):
      assert learner.actions() == [((time - 1) % arm_num, 1)]
      learner.update(([np.array([0])], ))

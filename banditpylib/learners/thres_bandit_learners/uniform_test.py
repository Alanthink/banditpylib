import numpy as np

from .uniform import Uniform


class TestUniform:
  """Test Uniform Sampling algorithm"""
  def test_actions(self):
    # test actions are in the right range
    arm_num = 10
    budget = 15
    apt = Uniform(arm_num=arm_num,
                  budget=budget,
                  theta=0.5,
                  eps=0)
    apt.reset()
    for _ in range(budget):
      actions = apt.actions()
      for (arm_id, _) in actions:
        assert 0 <= arm_id < arm_num
      apt.update([(np.array([1]), None)])

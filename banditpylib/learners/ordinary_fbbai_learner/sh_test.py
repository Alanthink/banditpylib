import numpy as np

from .sh import SH


class TestSH:
  """Test sequential halving policy"""
  def test_simple_run(self):
    arm_num = 5
    budget = 20
    learner = SH(arm_num=arm_num, budget=budget)
    learner.reset()

    while True:
      actions = learner.actions()
      if actions is None:
        break
      learner.update([(np.zeros(pulls), None) for (arm_id, pulls) in actions])
    assert learner.best_arm() in set(range(arm_num))

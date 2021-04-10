import numpy as np

from .uniform import Uniform


class TestUniform:
  """Test uniform policy"""
  def test_simple_run(self):
    arm_num = 5
    budget = 30
    learner = Uniform(arm_num=arm_num, budget=budget)
    learner.reset()

    while True:
      actions = learner.actions()
      if actions is None:
        break

      # pylint: disable=no-member
      learner.update([(np.random.random(pulls), None)
                      for (arm_id, pulls) in actions])

    assert learner.best_arm() in set(range(arm_num))

import numpy as np

from .lilucb_heur import LilUCBHeuristic


class TestLilUCBHeuristic:
  """Test exponential-gap elimination policy"""
  def test_simple_run(self):
    arm_num = 3
    confidence = 0.95
    learner = LilUCBHeuristic(arm_num=arm_num, confidence=confidence)
    learner.reset()

    while True:
      actions = learner.actions()
      if actions is None:
        break
      # pylint: disable=no-member
      learner.update([(np.random.normal(arm_id / arm_num, 1, pulls), None)
                      for (arm_id, pulls) in actions])
    assert learner.best_arm() in set(range(arm_num))

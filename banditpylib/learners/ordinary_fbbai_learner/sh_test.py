import numpy as np

from banditpylib.data_pb2 import Context, Feedback
from .sh import SH


class TestSH:
  """Test sequential halving policy"""
  def test_simple_run(self):
    arm_num = 5
    budget = 20
    learner = SH(arm_num=arm_num, budget=budget)
    learner.reset()

    while True:
      actions = learner.actions(Context())
      if not actions.arm_pulls_pairs:
        break

      feedback = Feedback()
      for arm_pulls_pair in actions.arm_pulls_pairs:
        arm_rewards_pair = feedback.arm_rewards_pairs.add()
        arm_rewards_pair.arm.id = arm_pulls_pair.arm.id
        arm_rewards_pair.rewards.extend(list(np.zeros(arm_pulls_pair.pulls)))
      learner.update(feedback)
    assert learner.best_arm in list(range(arm_num))

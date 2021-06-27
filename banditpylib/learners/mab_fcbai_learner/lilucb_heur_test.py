import numpy as np

from banditpylib.data_pb2 import Context, Feedback
from .lilucb_heur import LilUCBHeuristic


class TestLilUCBHeuristic:
  """Test exponential-gap elimination policy"""
  def test_simple_run(self):
    arm_num = 3
    confidence = 0.95
    learner = LilUCBHeuristic(arm_num=arm_num, confidence=confidence)
    learner.reset()

    while True:
      actions = learner.actions(Context())
      if not actions.arm_pulls_pairs:
        break

      feedback = Feedback()
      for arm_pulls_pair in actions.arm_pulls_pairs:
        arm_feedback = feedback.arm_feedbacks.add()
        arm_feedback.arm.id = arm_pulls_pair.arm.id
        arm_feedback.rewards.extend(
            list(
                np.random.normal(arm_pulls_pair.arm.id / arm_num, 1,
                                 arm_pulls_pair.pulls)))
      learner.update(feedback)

    assert learner.best_arm in list(range(arm_num))

import numpy as np

from banditpylib.data_pb2 import Context, Feedback
from .exp_gap import ExpGap


class TestExpGap:
  """Test exponential-gap elimination policy"""
  def test_simple_run(self):
    arm_num = 3
    confidence = 0.95
    learner = ExpGap(arm_num=arm_num, confidence=confidence)
    learner.reset()

    while True:
      actions = learner.actions(Context())
      if not actions.arm_pulls:
        break

      feedback = Feedback()
      for arm_pull in actions.arm_pulls:
        arm_feedback = feedback.arm_feedbacks.add()
        arm_feedback.arm.id = arm_pull.arm.id
        arm_feedback.rewards.extend(
            list(np.random.normal(arm_pull.arm.id / arm_num, 1,
                                  arm_pull.times)))
      learner.update(feedback)
    assert learner.best_arm in list(range(arm_num))

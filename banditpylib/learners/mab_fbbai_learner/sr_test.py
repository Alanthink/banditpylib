import numpy as np

from banditpylib.data_pb2 import Context, Feedback
from .sr import SR


class TestSR:
  """Test successive rejects policy"""
  def test_simple_run(self):
    arm_num = 5
    budget = 30
    learner = SR(arm_num=arm_num, budget=budget)
    learner.reset()

    while True:
      actions = learner.actions(Context())
      if not actions.arm_pulls:
        break

      feedback = Feedback()
      for arm_pull in actions.arm_pulls:
        arm_feedback = feedback.arm_feedbacks.add()
        arm_feedback.arm.id = arm_pull.arm.id
        arm_feedback.rewards.extend(list(np.random.random(arm_pull.times)))
      learner.update(feedback)
    assert learner.best_arm in list(range(arm_num))

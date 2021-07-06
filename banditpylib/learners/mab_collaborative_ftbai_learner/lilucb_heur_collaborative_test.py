import numpy as np

from banditpylib.data_pb2 import Feedback
from .lilucb_heur_collaborative import CentralizedLilUCBHeuristic


class TestLilUCBHeuristicCollaborative:
  """Test collaborative lilUCB-Heuristic elimination policy"""
  def test_simple_run(self):
    arm_num = 3
    confidence = 0.95
    learner = CentralizedLilUCBHeuristic(arm_num=arm_num,
                                         confidence=confidence,
                                         assigned_arms=np.arange(arm_num))
    learner.reset()

    while True:
      actions = learner.actions()
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

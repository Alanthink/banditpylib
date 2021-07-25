from typing import List, Dict

import numpy as np

from banditpylib.data_pb2 import Feedback
from .lilucb_heur_collaborative import CentralizedLilUCBHeuristic, assign_arms


class TestLilUCBHeuristicCollaborative:
  """Test collaborative lilUCB-heuristic policy"""
  def test_assign_arms(self):
    active_arms = [1, 2, 3, 4]
    active_agents = [4, 5, 6, 7, 8, 9]
    agent_arm_assignment = assign_arms(active_arms, active_agents)
    # Check that each agent is assigned at least one arm
    assert set(agent_arm_assignment.keys()) == set(active_agents)
    min_num_agents_per_arm = int(len(active_agents) / len(active_arms))
    arm_agent_assignment: Dict[int, List[int]] = {}
    for agent_id in active_agents:
      assert len(agent_arm_assignment[agent_id]) == 1
      arm_id = agent_arm_assignment[agent_id][0]
      if arm_id not in arm_agent_assignment:
        arm_agent_assignment[arm_id] = []
      arm_agent_assignment[arm_id].append(agent_id)

    for agents in arm_agent_assignment.items():
      assert len(agents) >= min_num_agents_per_arm

    active_arms = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    active_agents = [4, 5, 6]
    agent_arm_assignment = assign_arms(active_arms, active_agents)
    # Check that each agent is assigned at least one arm
    assert set(agent_arm_assignment.keys()) == set(active_agents)
    min_num_arms_per_agent = int(len(active_arms) / len(active_agents))
    for agent_id in active_agents:
      assert len(agent_arm_assignment[agent_id]) >= min_num_arms_per_agent

  def test_centralized_lilUCBHeuristic(self):
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

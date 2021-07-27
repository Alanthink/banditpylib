from typing import List, Dict

from .lilucb_heur_collaborative_utils import assign_arms, \
    get_num_pulls_per_round


class TestLilUCBHeuristicCollaborativeUtils:
  """Test utilities of collaborative lilUCB-heuristic policy"""
  def test_get_num_pulls_per_round(self):
    rounds = 3
    arm_num = 5
    num_agents = 10
    horizon = 2000

    num_pulls_per_round = get_num_pulls_per_round(rounds=rounds,
                                                  arm_num=arm_num,
                                                  num_agents=num_agents,
                                                  horizon=horizon)
    assert num_pulls_per_round == [1000, 1000, 0]

    rounds = 3
    arm_num = 20
    num_agents = 10
    horizon = 2000

    num_pulls_per_round = get_num_pulls_per_round(rounds=rounds,
                                                  arm_num=arm_num,
                                                  num_agents=num_agents,
                                                  horizon=horizon)
    assert num_pulls_per_round == [1000, 500, 500, 0]

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

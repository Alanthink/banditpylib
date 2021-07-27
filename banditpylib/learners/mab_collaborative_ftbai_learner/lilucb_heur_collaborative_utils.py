from typing import List, Dict
import math

import numpy as np

from banditpylib.data_pb2 import Feedback, Actions, Context

from banditpylib import argmax_or_min_tuple
from banditpylib.arms import PseudoArm
from banditpylib.learners.mab_fcbai_learner import MABFixedConfidenceBAILearner


class CentralizedLilUCBHeuristic(MABFixedConfidenceBAILearner):
  """LilUCB heuristic policy :cite:`jamieson2014lil`
  Modified implementation to supplement CollaborativeAgent
  along with additional functionality to work on only a subset of arms

  :param int arm_num: number of arms of the bandit
  :param float confidence: confidence level. It should be within (0, 1). The
    algorithm should output the best arm with probability at least this value.
  :param np.ndarray assigned_arms: arm indices the learner has to work with
  :param str name: alias name
  """
  def __init__(self,
               arm_num: int,
               confidence: float,
               assigned_arms: np.ndarray = None,
               name: str = None):
    assert np.max(assigned_arms) < arm_num and len(assigned_arms) <= arm_num, (
        "assigned arms should be a subset of [arm_num]\nReceived: " +
        str(assigned_arms))
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)
    if assigned_arms is not None:
      self.__assigned_arms = assigned_arms
    else:
      self.__assigned_arms = np.arange(arm_num)

  def _name(self) -> str:
    return 'lilUCB_heur_collaborative'

  def reset(self):
    # create only as many local arms as num_assigned_arms
    # entire algo behaves as if there are just num_assigned_arms in the bandit
    self.__pseudo_arms = [PseudoArm() for arm_id in self.__assigned_arms]
    # Parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1 + 10 / len(self.__assigned_arms)
    self.__eps = 0
    self.__delta = (1 - self.confidence) / 5
    # Total number of pulls used
    self.__total_pulls = 0
    self.__stage = 'initialization'
    self.__ucb = np.array([0.0] * len(self.__assigned_arms))

  def __confidence_radius(self, pulls: int) -> float:
    """
    Args:
      pulls: number of pulls

    Returns:
      confidence radius
    """
    if (1 + self.__eps) * pulls == 1:
      return math.inf
    return (1 + self.__beta) * (1 + math.sqrt(self.__eps)) * math.sqrt(
        2 * (1 + self.__eps) *
        math.log(math.log((1 + self.__eps) * pulls) / self.__delta) / pulls)

  def __update_ucb(self, arm_id: int):
    """
    Args:
      arm_id: index of the arm whose ucb has to be updated
    """
    self.__ucb[arm_id] = self.__pseudo_arms[arm_id].em_mean +\
      self.__confidence_radius(self.__pseudo_arms[arm_id].total_pulls)

  def actions(self, context: Context = None) -> Actions:
    del context
    if self.__stage == 'initialization':
      actions = Actions()  # default state is normal

      # 1 pull each for every assigned arm
      for arm_id in self.__assigned_arms:
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = arm_id
        arm_pull.times = 1
      return actions

    # self.__stage == 'main'
    actions = Actions()

    for pseudo_arm in self.__pseudo_arms:
      if pseudo_arm.total_pulls >= (
          1 + self.__a * (self.__total_pulls - pseudo_arm.total_pulls)):
        return actions

    arm_pull = actions.arm_pulls.add()

    # map local arm index to the bandits arm index
    arm_pull.arm.id = self.__assigned_arms[int(np.argmax(self.__ucb))]
    arm_pull.times = 1

    return actions

  def update(self, feedback: Feedback):
    for arm_feedback in feedback.arm_feedbacks:
      # reverse map from bandit index to local index
      pseudo_arm_index = np.where(
          self.__assigned_arms == arm_feedback.arm.id)[0][0]
      self.__pseudo_arms[pseudo_arm_index].update(
          np.array(arm_feedback.rewards))
      self.__update_ucb(pseudo_arm_index)
      self.__total_pulls += len(arm_feedback.rewards)

    if self.__stage == 'initialization':
      self.__stage = 'main'

  @property
  def best_arm(self) -> int:
    # map best arm local index to actual bandit index
    return self.__assigned_arms[argmax_or_min_tuple([
        (pseudo_arm.total_pulls, arm_id)
        for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
    ])]

  def get_total_pulls(self) -> int:
    return self.__total_pulls


def get_num_pulls_per_round(rounds: int, arm_num: int, num_agents: int,
                            horizon: int):
  """Calculate number of pulls used per round

  When `arm_num` > `num_agents`, the output result has one more round than the
  input. It is assumed there will be no communication after the first
  round within which the main goal is to eliminate assigned arms such that only
  one remains.

  Args:
    rounds: number of total rounds allowed
    arm_num: number of arms of the bandit
    num_agents: number of agents
    horizon: maximum number of pulls the agent can make
      (over all rounds combined)

  Returns:
    number of pulls used per round
  """
  num_pulls_per_round = []
  if arm_num > num_agents:
    pseudo_comm_rounds = rounds
    if pseudo_comm_rounds == 1:
      num_pulls_per_round.append(horizon)
    else:
      num_pulls_per_round.append(int(0.5 * horizon))
      num_pulls_per_round.extend(
          [int(0.5 * horizon /
               (pseudo_comm_rounds - 1))] * (pseudo_comm_rounds - 1))
  else:
    comm_rounds = rounds - 1
    num_pulls_per_round.extend([int(horizon / comm_rounds)] * comm_rounds)
  # For the last round, we always use 0 pulls.
  num_pulls_per_round.append(0)
  # Assign the remaining budget
  for i in range(horizon - sum(num_pulls_per_round)):
    num_pulls_per_round[i] += 1

  return num_pulls_per_round


def assign_arms(active_arms: List[int],
                agent_ids: List[int]) -> Dict[int, List[int]]:
  """Assign arms to agents to pull

  Args:
    active_arms: list of active arm ids
    agent_ids: list of agent ids

  Returns:
    arm assignment where key is agent id and value is assigned arms to this
      agent
  """
  if not active_arms:
    raise ValueError("No arms to assign.")

  if not agent_ids:
    raise ValueError("No agents to assign.")

  agent_arm_assignment: Dict[int, List[int]] = {}

  if len(active_arms) == 1:
    # Use -1 as the first arm id if there is only one active arm
    for agent_id in agent_ids:
      agent_arm_assignment[agent_id] = [-1, active_arms[0]]
    return agent_arm_assignment

  if len(active_arms) < len(agent_ids):
    # Number of arms is less than the number of agents
    min_num_agents_per_arm = int(len(agent_ids) / len(active_arms))
    arms_assign_list = active_arms * min_num_agents_per_arm
    if len(agent_ids) > len(arms_assign_list):
      arms_assign_list.extend(
          list(
              np.random.choice(active_arms,
                               len(agent_ids) - len(arms_assign_list))))
    np.random.shuffle(arms_assign_list)

    for i, agent_id in enumerate(agent_ids):
      agent_arm_assignment[agent_id] = [arms_assign_list[i]]

  else:
    # Number of arms is at least the number of agents
    min_num_arms_per_agent = int(len(active_arms) / len(agent_ids))
    agents_assign_list = agent_ids * min_num_arms_per_agent
    if len(active_arms) > len(agents_assign_list):
      agents_assign_list.extend(
          list(
              np.random.choice(agent_ids,
                               len(active_arms) - len(agents_assign_list))))
    np.random.shuffle(agents_assign_list)

    for i, arm_id in enumerate(active_arms):
      agent_id = agents_assign_list[i]
      if agent_id not in agent_arm_assignment:
        agent_arm_assignment[agent_id] = []
      agent_arm_assignment[agent_id].append(arm_id)

  return agent_arm_assignment

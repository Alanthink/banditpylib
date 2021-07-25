from typing import Optional, List, Tuple, Dict
import math

import numpy as np

from banditpylib import argmax_or_min_tuple
from banditpylib.data_pb2 import Feedback, Actions, Context
from banditpylib.arms import PseudoArm
from banditpylib.learners.mab_fcbai_learner import MABFixedConfidenceBAILearner

from .utils import MABCollaborativeFixedTimeBAIAgent, \
    MABCollaborativeFixedTimeBAIMaster, MABCollaborativeFixedTimeBAILearner


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

  def actions(self, context=None) -> Actions:
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


class LilUCBHeuristicAgent(MABCollaborativeFixedTimeBAIAgent):
  """Agent of collaborative learning

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: total number of agents
  :param Optional[str] name: alias name
  """

  # Stages within the agent
  UNASSIGNED = "unassigned"
  CENTRALIZED_LEARNING = "centralized_learning"
  LEARNING = "learning"
  COMMUNICATION = "communication"
  TERMINATION = "termination"

  def __init__(self,
               arm_num: int,
               rounds: int,
               horizon: int,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(name)
    self.__arm_num = arm_num
    self.__comm_rounds = rounds - 1
    self.__horizon = horizon
    self.__num_agents = num_agents
    self.reset()

  def _name(self) -> str:
    return "lilucb_heuristic_collaborative_agent"

  def reset(self):
    self.__id = np.random.rand() # pylint: disable=unused-private-member
    self.__round_index = 0

    # Calculate number of pulls used per round
    self.__num_pulls_per_round = []
    if self.__arm_num > self.__num_agents:
      if self.__comm_rounds == 1:
        self.__num_pulls_per_round.append(self.__horizon)
      else:
        self.__num_pulls_per_round.append(int(0.5 * self.__horizon))
        self.__num_pulls_per_round.extend(
            [int(0.5 * self.__horizon /
                 (self.__comm_rounds - 1))] * (self.__comm_rounds - 1))
    else:
      self.__num_pulls_per_round.extend(
          [int(self.__horizon / self.__comm_rounds)] * self.__comm_rounds)
    # For the last round, we always use 0 pulls.
    self.__num_pulls_per_round.append(0)
    # Assign the remaining budget
    for i in range(self.__horizon - sum(self.__num_pulls_per_round)):
      self.__num_pulls_per_round[i] += 1

    self.__stage = self.UNASSIGNED

  def set_input_arms(self, arms: List[int]):
    if self.__stage != self.UNASSIGNED:
      raise Exception("The agent is expected in stage unassigned. Got %s." %
                      self.__stage)

    if arms[0] < 0:
      # Terminate since there is only one active arm
      self.__best_arm = arms[1]
      self.__stage = self.TERMINATION
      return

    # Maintain informaiton of assigned arms
    self.__assigned_arms = np.array(arms)
    self.__assigned_arm_info: Dict[int, Tuple[float, int]] = {}
    for arm_id in arms:
      self.__assigned_arm_info[arm_id] = (0.0, 0)

    if self.__round_index == 0 and len(self.__assigned_arms) > 1:
      # Confidence of 0.99 suggested in the paper
      self.__central_algo = CentralizedLilUCBHeuristic(self.__arm_num, 0.99,
                                                       self.__assigned_arms)
      self.__central_algo.reset()
      self.__stage = self.CENTRALIZED_LEARNING
    else:
      if len(self.__assigned_arms) > 1:
        raise Exception("Got more than 1 arm in stage learning.")
      if self.__round_index == self.__comm_rounds:
        self.__best_arm = arms[0]
        self.__stage = self.TERMINATION
      else:
        self.__arm_to_broadcast = arms[0]
        self.__stage = self.LEARNING

  def actions(self, context: Context) -> Actions:
    del context

    if self.__stage == self.UNASSIGNED:
      raise Exception("%s: I can\'t act in stage unassigned." % self.name)

    if self.__stage == self.CENTRALIZED_LEARNING:
      if self.__round_index > 0:
        raise Exception("Expected centralized learning in round 0. Got %d." %
                        self.__round_index)

      if self.__central_algo.get_total_pulls(
      ) >= self.__num_pulls_per_round[0]:
        # Early stop the centralized algorithm when it uses more than horizon
        # / 2 pulls.
        self.__stage = self.COMMUNICATION
        self.__arm_to_broadcast = np.random.choice(self.__assigned_arms)
        actions = Actions()
        actions.state = Actions.WAIT
        return actions

      central_algo_actions = self.__central_algo.actions()
      if not central_algo_actions.arm_pulls:
        # Centralized algorithm terminates before using up horizon / 2 pulls
        self.__stage = self.COMMUNICATION
        self.__arm_to_broadcast = self.__central_algo.best_arm
        actions = Actions()
        actions.state = Actions.WAIT
        return actions
      return central_algo_actions
    elif self.__stage == self.LEARNING:
      actions = Actions()
      arm_pull = actions.arm_pulls.add()
      arm_pull.arm.id = self.__arm_to_broadcast
      arm_pull.times = self.__num_pulls_per_round[self.__round_index]
      return actions
    elif self.__stage == self.COMMUNICATION:
      actions = Actions()
      actions.state = Actions.WAIT
      return actions
    else:
      # self.__stage == self.TERMINATION
      actions = Actions()
      actions.state = Actions.STOP
      return actions

  def update(self, feedback: Feedback):
    if self.__stage not in [self.CENTRALIZED_LEARNING, self.LEARNING]:
      raise Exception("%s: I can\'t do update in stage not learning." %
                      self.name)

    for arm_feedback in feedback.arm_feedbacks:
      old_arm_info = self.__assigned_arm_info[arm_feedback.arm.id]
      new_arm_info = (
          (old_arm_info[0] * old_arm_info[1] + sum(arm_feedback.rewards)) /
          (old_arm_info[1] + len(arm_feedback.rewards)),
          old_arm_info[1] + len(arm_feedback.rewards))
      self.__assigned_arm_info[arm_feedback.arm.id] = new_arm_info

    if self.__stage == self.CENTRALIZED_LEARNING:
      self.__central_algo.update(feedback)
    else:
      # self.__stage == self.LEARNING
      self.__stage = self.COMMUNICATION

  @property
  def best_arm(self) -> int:
    if self.__stage != self.TERMINATION:
      raise Exception('%s: I don\'t have an answer yet.' % self.name)
    return self.__best_arm

  def broadcast(self) -> Dict[int, Tuple[float, int]]:
    if self.__stage != self.COMMUNICATION:
      raise Exception('%s: I can\'t broadcast in stage %s.'\
        % (self.name, self.__stage))

    # Complete the current round
    self.__round_index += 1
    self.__stage = self.UNASSIGNED

    message: Dict[int, Tuple[float, int]] = {}
    message[self.__arm_to_broadcast] = self.__assigned_arm_info[
        self.__arm_to_broadcast]
    return message


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


class LilUCBHeuristicMaster(MABCollaborativeFixedTimeBAIMaster):
  """Master of collaborative learning

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: number of agents
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               rounds: int,
               horizon: int,
               num_agents: int,
               name: Optional[str] = None):
    super().__init__(name)
    self.__arm_num = arm_num
    self.__comm_rounds = rounds - 1
    self.__T = horizon
    self.__num_agents = num_agents

  def _name(self) -> str:
    return "lilucb_heuristic_collaborative_master"

  def reset(self):
    self.__active_arms = list(range(self.__arm_num))

  def initial_arm_assignment(self) -> Dict[int, List[int]]:
    return assign_arms(self.__active_arms, list(range(self.__num_agents)))

  def elimination(
      self, messages: Dict[int, Dict[int,
                                     Tuple[float,
                                           int]]]) -> Dict[int, List[int]]:

    aggregate_messages: Dict[int, Tuple[float, int]] = {}
    for agent_id in messages.keys():
      message_from_agent = messages[agent_id]
      for arm_id in message_from_agent:
        if arm_id not in aggregate_messages:
          aggregate_messages[arm_id] = (0.0, 0)
        arm_info = message_from_agent[arm_id]
        new_pulls = aggregate_messages[arm_id][1] + arm_info[1]
        new_em_mean_reward = (aggregate_messages[arm_id][0] * \
            aggregate_messages[arm_id][1] + arm_info[0] * arm_info[1]) \
            / new_pulls
        aggregate_messages[arm_id] = (new_em_mean_reward, new_pulls)

    accumulated_arm_ids = np.array(list(aggregate_messages.keys()))
    accumulated_em_mean_rewards = np.array(
        list(map(lambda x: aggregate_messages[x][0],
                 aggregate_messages.keys())))

    # Elimination
    confidence_radius = np.sqrt(
        self.__comm_rounds *
        np.log(200 * self.__num_agents * self.__comm_rounds) /
        (self.__T * max(1, self.__num_agents / len(self.__active_arms))))
    highest_em_reward = np.max(accumulated_em_mean_rewards)
    self.__active_arms = list(
        accumulated_arm_ids[accumulated_em_mean_rewards >= (
            highest_em_reward - 2 * confidence_radius)])

    return assign_arms(self.__active_arms, list(messages.keys()))


class LilUCBHeuristicCollaborative(MABCollaborativeFixedTimeBAILearner):
  """Colaborative learner using lilucb heuristic as centralized policy

  :param int num_agents: number of agents
  :param int arm_num: number of arms of the bandit
  :param int rounds: number of total rounds allowed
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param Optional[str] name: alias name
  """
  def __init__(self,
               num_agents: int,
               arm_num: int,
               rounds: int,
               horizon: int,
               name: Optional[str] = None):
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    if rounds <= 2:
      raise ValueError('Number of rounds is expected at least 2. Got %d.' %
                       rounds)

    if horizon <= rounds - 1:
      raise ValueError(
          'Horizon is expected at least total rounds minus one. Got %d.' %
          horizon)

    super().__init__(agent=LilUCBHeuristicAgent(arm_num=arm_num,
                                                rounds=rounds,
                                                horizon=horizon,
                                                num_agents=num_agents),
                     master=LilUCBHeuristicMaster(arm_num=arm_num,
                                                  rounds=rounds,
                                                  horizon=horizon,
                                                  num_agents=num_agents),
                     num_agents=num_agents,
                     name=name)

  def _name(self) -> str:
    return 'lilucb_heuristic_collaborative'

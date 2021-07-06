# Implementation of the Collaborative Learning Algorithm

from typing import Optional, List, Tuple, Dict
import random
import math
from copy import deepcopy as dcopy

import numpy as np

from banditpylib.data_pb2 import Feedback, Actions
from banditpylib.arms import PseudoArm
from banditpylib import argmax_or_min_tuple
from banditpylib.learners.mab_fcbai_learner import MABFixedConfidenceBAILearner

from .utils import CollaborativeBAIAgent, CollaborativeBAIMaster

class LilUCBHeuristicCollaborative(MABFixedConfidenceBAILearner):
  """LilUCB heuristic policy :cite:`jamieson2014lil`
  Modified implementation to supplement CollaborativeAgent
  along with additional functionality to work on only a subset of arms

  :param int arm_num: number of arms of the bandit
  :param float confidence: confidence level. It should be within (0, 1). The
    algorithm should output the best arm with probability at least this value.
  :param np.ndarray assigned_arms: arm indices the learner has to work with
  :param str name: alias name
  """
  def __init__(self, arm_num: int, confidence: float,
    assigned_arms: np.ndarray = None, name: str = None):
    assert np.max(assigned_arms)<arm_num and len(assigned_arms)<=arm_num, (
      "assigned arms should be a subset of [arm_num]\nReceived: "
        + str(assigned_arms))
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

  def __ucb(self) -> np.ndarray:
    """
    Returns:
      upper confidence bound
    """
    return np.array([
        pseudo_arm.em_mean + self.__confidence_radius(pseudo_arm.total_pulls)
        for pseudo_arm in self.__pseudo_arms
    ])

  def actions(self, context=None) -> Actions:
    del context
    if self.__stage == 'initialization':
      actions = Actions() # default state is normal

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
    arm_pull.arm.id = self.__assigned_arms[int(np.argmax(self.__ucb()))]
    arm_pull.times = 1

    return actions

  def update(self, feedback: Feedback):
    for arm_feedback in feedback.arm_feedbacks:
      # reverse map from bandit index to local index
      pseudo_arm_index = np.where(
        self.__assigned_arms==arm_feedback.arm.id)[0][0]
      self.__pseudo_arms[pseudo_arm_index].update(
          np.array(arm_feedback.rewards))
      self.__total_pulls += len(arm_feedback.rewards)

    if self.__stage == 'initialization':
      self.__stage = 'main'

  @property
  def best_arm(self) -> int:
    # map best arm local index to actual bandit index
    return self.__assigned_arms[
      argmax_or_min_tuple([
        (pseudo_arm.total_pulls, arm_id)
        for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
      ])
    ]

  def get_total_pulls(self) -> int:
    return self.__total_pulls


class LilUCBHeuristicCollaborativeBAIAgent(CollaborativeBAIAgent):
  r"""Implementation of agent of the Collaborative Learning Algorithm\
  Uses LilUCBHeuristic as the central algorithm

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param Optional[str] name: alias name
  """

  def __init__(self, arm_num: int, rounds: int,
    horizon: int, name: Optional[str] = None):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    if rounds <= 1:
      raise ValueError('Number of rounds is expected at least 2. Got %d.' %
                       rounds)
    self.__arm_num = arm_num
    self.__comm_rounds = rounds - 1
    self.__horizon = horizon
    self.__num_pulls_learning = int(0.5 * horizon / self.__comm_rounds)

  def _name(self) -> str:
    return "lilUCBHeur_collaborative_agent"

  def reset(self):
    self.__round_num = 0
    self.__stage = "unassigned"
    # True if action forwarded from central algo
    self.__central_algo_action_taken = False

  def __complete_round(self):
    self.__round_num += 1
    self.__central_algo_action_taken = False
    if self.__round_num < self.__comm_rounds + 1:
      self.__stage = "unassigned"
    else:
      self.__stage = "termination"

  def set_input_arms(self, arms: List[int]):
    if arms[0] < 0:
      # terminate since there is only one active arm
      self.__learning_arm = arms[1]
      self.__stage = "termination"
      return

    self.__assigned_arms = np.array(arms)
    # confidence of 0.01 suggested in the paper
    self.__central_algo = LilUCBHeuristicCollaborative(self.__arm_num,
      0.99, self.__assigned_arms)
    self.__central_algo.reset()
    if self.__stage == "unassigned":
      self.__stage = "preparation"

  def actions(self, context=None) -> Actions:
    # a core assumption is all non-empty actions immediately receive feedback
    # and hence stage is changed here and not when feedback is received
    del context

    if self.__stage == "unassigned":
      raise Exception("No arms assigned to agent " + self.name)

    # in preparation:
    #   if only one arm is assigned, proceed to learning
    #   else if central_algo is running, forward its actions
    #   and get best arm when central_algo completes
    #   but interrupt central algo after T/2 pulls
    elif self.__stage == "preparation":
      if len(self.__assigned_arms) == 1:
        self.__stage = "learning"
        self.__learning_arm = self.__assigned_arms[0]
        return self.actions()
      if self.__central_algo.get_total_pulls() >= self.__horizon//2:
        self.__stage = "learning"
        # use whatever best_arm the central algo outputs
        self.__learning_arm = self.__central_algo.best_arm
        return self.actions()

      central_algo_actions = self.__central_algo.actions()
      if not central_algo_actions.arm_pulls:
        # central algo terminated before T/2 pulls
        self.__stage = "learning"
        self.__learning_arm = self.__central_algo.best_arm
        return self.actions()
      self.__central_algo_action_taken = True
      return central_algo_actions

    # in learning:
    #   if learning_arm is none, do no pulls and move to communication
    #   else pull learning_arm and move to communication
    elif self.__stage == "learning":
      actions = Actions()
      self.__stage = "communication"
      if self.__learning_arm is None:
        actions.state = Actions.WAIT
        return actions
      else:
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = self.__learning_arm # pylint: disable=protobuf-type-error
        arm_pull.times = self.__num_pulls_learning
        return actions

    elif self.__stage == "communication":
      actions = Actions()
      actions.state = Actions.WAIT
      return actions

    elif self.__stage == "termination":
      actions = Actions()
      actions.state = Actions.STOP
      return actions

    else:
      raise Exception(self.name + ": " + self.__stage +
        " does not allow actions to be played")

  def update(self, feedback: Feedback):
    self.__learning_mean = None # default in case learning_arm is None
    num_pulls = 0
    for arm_feedback in feedback.arm_feedbacks:
      num_pulls += len(arm_feedback.rewards)
    if self.__central_algo_action_taken:
      self.__central_algo.update(feedback)
    elif num_pulls>0:
      # non-zero pulls not by central_algo => learning step was done
      for arm_feedback in feedback.arm_feedbacks:
        if arm_feedback.arm.id == self.__learning_arm:
          self.__learning_mean = np.array(arm_feedback.rewards).mean()
          self.__pulls_used = len(arm_feedback.rewards)
    # else ignore feedback (which is empty)

    self.__central_algo_action_taken = False

  @property
  def best_arm(self) -> int:
    # returns arm that the agent chose (could be None)
    if self.__stage != "termination":
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__learning_arm

  def broadcast(self) -> Dict[int, Tuple[float, int]]:
    if self.__stage != "communication":
      raise Exception('%s: I can\'t broadcast in stage %s!'\
        % (self.name, self.__stage))
    return_dict = {}
    if self.__learning_arm:
      return_dict[self.__learning_arm] = (self.__learning_mean,
        self.__pulls_used)
    self.__complete_round()
    return return_dict

class LilUCBHeuristicCollaborativeBAIMaster(CollaborativeBAIMaster):
  r"""Implementation of master in Collaborative Learning Algorithm

  :param int arm_num: number of arms of the bandit
  :param int rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: number of agents
  :param Optional[str] name: alias name
  """

  def __init__(self, arm_num:int, rounds: int,
    horizon: int, num_agents: int, name: Optional[str] = None):
    super().__init__(name)
    if arm_num <= 1:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       arm_num)
    if rounds <= 1:
      raise ValueError('Number of rounds is expected at least 2. Got %d.' %
                       rounds)
    self.__arm_num = arm_num
    self.__comm_rounds = rounds - 1
    self.__T = horizon
    self.__num_agents = num_agents

  def _name(self) -> str:
    return "lilUCBHeur_collaborative_master"

  def reset(self):
    self.__active_arms = list(range(self.__arm_num))

  def __assign_arms(self, agent_ids: List[int]) -> Dict[int, List[int]]:
    agent_arm_assignment: Dict[int, List[int]] = {}

    def random_round(x: float) -> int:
      # if x = 19.3
      # rounded to 19 with probability 0.7
      # rounded to 20 with probability 0.3
      frac_x = x - int(x)
      if np.random.uniform() > frac_x:
        return int(x)
      return int(x) + 1

    if len(self.__active_arms) == 1:
      # use -1 as the first arm if there is only 1 active arm
      agent_arm_assignment = {}
      for agent_id in agent_ids:
        agent_arm_assignment[agent_id] = [-1, self.__active_arms[0]]
      return agent_arm_assignment

    if len(self.__active_arms) < len(agent_ids):
      arms_assign_list = []
      num_agents_per_arm = len(agent_ids) / len(self.__active_arms)

      # match agent to all but the last arm
      for arm in self.__active_arms[:-1]:
        arms_assign_list += [[arm]] * \
          random_round(num_agents_per_arm)

      # match remaining agents to last arm
      arms_assign_list += [[self.__active_arms[-1]]] * \
        (len(agent_ids) - len(arms_assign_list))
      random.shuffle(arms_assign_list)
      for i, agent_id in enumerate(agent_ids):
        agent_arm_assignment[agent_id] = arms_assign_list[i]

    else:
      active_arms_copy = dcopy(self.__active_arms)
      random.shuffle(active_arms_copy)
      for agent_id in agent_ids:
        agent_arm_assignment[agent_id] = []

      # assign atleast 1 arm per agent
      for i, arm in enumerate(active_arms_copy[:len(agent_ids)]):
        agent_arm_assignment[agent_ids[i]].append(arm)

      # assign remaining arms
      num_arms_per_agent = (len(active_arms_copy) - len(agent_ids))\
      / len(agent_ids)
      i = len(agent_ids)
      for agent_id in agent_ids[:-1]:
        if i >= len(active_arms_copy):
          break
        num_arms = random_round(num_arms_per_agent)
        agent_arm_assignment[agent_id] += active_arms_copy[i: i + num_arms]
        i+= num_arms
      if i < len(active_arms_copy):
        agent_arm_assignment[agent_ids[-1]] += active_arms_copy[i:]

    return agent_arm_assignment

  def initial_arm_assignment(self) -> Dict[int, List[int]]:
    return self.__assign_arms(list(range(self.__num_agents)))

  def elimination(self, agent_ids: List[int],
    messages: Dict[int, Dict[int, Tuple[float, int]]]) -> Dict[int, List[int]]:

    aggregate_messages: Dict[int, Tuple[float, int]] = {}
    for agent_id in messages.keys():
      message_from_agent = messages[agent_id]
      for arm_id in message_from_agent:
        if arm_id not in aggregate_messages:
          aggregate_messages[arm_id] = (0.0, 0)
        arm_info = message_from_agent[arm_id]
        new_pulls = aggregate_messages[arm_id][1] + arm_info[1]
        new_em_mean_reward = (aggregate_messages[arm_id][1] * \
         aggregate_messages[arm_id][0] + arm_info[1] * arm_info[0]) / new_pulls
        aggregate_messages[arm_id] = (new_em_mean_reward, new_pulls)

    accumulated_arm_ids = np.array(list(aggregate_messages.keys()))
    accumulated_em_mean_rewards = np.array(
      list(map(lambda x: aggregate_messages[x][0], aggregate_messages.keys())))

    # elimination
    confidence_radius = np.sqrt(
      self.__comm_rounds * np.log(200 * self.__num_agents * self.__comm_rounds)
      / (self.__T * max(1, self.__num_agents / len(self.__active_arms)))
    )
    highest_em_reward = np.max(accumulated_em_mean_rewards)
    self.__active_arms = list(
      accumulated_arm_ids[accumulated_em_mean_rewards >=
        highest_em_reward - 2 * confidence_radius]
    )

    return self.__assign_arms(agent_ids)

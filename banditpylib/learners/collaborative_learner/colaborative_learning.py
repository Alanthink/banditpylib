# Implementation of the Collaborative Learning Algorithm

from typing import Optional, List, Tuple
import random
from copy import deepcopy as dcopy

import numpy as np

from banditpylib.data_pb2 import Feedback, Actions

from .lilucb_heur_collaborative import LilUCBHeuristicCollaborative
from .utils import CollaborativeBAIAgent, CollaborativeBAIMaster

class LilUCBHeuristicCollaborativeBAIAgent(CollaborativeBAIAgent):
  r"""Implementation of agent of the Collaborative Learning Algorithm
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
    self.__arm_num = arm_num
    self.__rounds = rounds
    self.__horizon = horizon
    self.__num_pulls_learning = int(0.5 * horizon / rounds)

  def _name(self) -> str:
    return "lilUCBHeur_collaborative_agent"

  def reset(self):
    self.__total_pulls = 0
    self.__round_pulls = 0
    self.__round_num = 0
    self.__stage = "unassigned"
    # True if action forwarded from central algo
    self.__central_algo_action_taken = False

  def complete_round(self):
    self.__round_num += 1
    self.__round_pulls = 0
    self.__central_algo_action_taken = False
    if self.__round_num < self.__rounds + 1:
      self.__stage = "unassigned"
    else:
      self.__stage = "termination"

  def assign_arms(self, arms: List[int]):
    if self.__stage != "unassigned":
      raise Exception('%s: I can\'t be assigned arms in stage %s!'\
        % (self.name, self.__stage))
    self.__assigned_arms = np.array(arms)
    # confidence of 0.01 suggested in the paper
    self.__central_algo = LilUCBHeuristicCollaborative(self.__arm_num,
      0.99, self.__assigned_arms)
    self.__central_algo.reset()
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
        self.__i_l_r = self.__assigned_arms[0]
        return self.actions()
      if self.__central_algo.get_total_pulls() >= self.__horizon//2:
        self.__stage = "learning"
        self.__i_l_r = None
        return self.actions()

      central_algo_actions = self.__central_algo.actions()
      if not central_algo_actions.arm_pulls:
        # central algo terminated before T/2 pulls
        self.__stage = "learning"
        self.__i_l_r = self.__central_algo.best_arm
        return self.actions()
      self.__central_algo_action_taken = True
      return central_algo_actions

    # in learning:
    #   if i_l_r is none, do no pulls and move to communication
    #   else pull i_l_r for a fixed number of times and move to communication
    elif self.__stage == "learning":
      actions = Actions()
      self.__stage = "communication"
      if self.__i_l_r is None:
        actions.state = Actions.WAIT
        return actions
      else:
        arm_pull = actions.arm_pulls.add()
        arm_pull.arm.id = self.__i_l_r # pylint: disable=protobuf-type-error
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
    # update total pulls
    num_pulls = 0
    for arm_feedback in feedback.arm_feedbacks:
      num_pulls += len(arm_feedback.rewards)
    self.__total_pulls += num_pulls
    self.__round_pulls += num_pulls

    # handle feedback
    self.__p_l_r = None # default in case i_l_r is None
    if self.__central_algo_action_taken:
      self.__central_algo.update(feedback)
    elif num_pulls>0:
      # non-zero pulls not by central_algo => learning step was done
      for arm_feedback in feedback.arm_feedbacks:
        if arm_feedback.arm.id == self.__i_l_r:
          self.__p_l_r = np.array(arm_feedback.rewards).mean()
    # else ignore feedback (which is empty)

    self.__central_algo_action_taken = False

  @property
  def best_arm(self) -> int:
    # returns arm that the agent chose (could be None)
    if self.__stage != "termination":
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__i_l_r

  @property
  def stage(self) -> str:
    return self.__stage

  def broadcast(self) -> Tuple[int, float, int]:
    if self.__stage != "communication":
      raise Exception('%s: I can\'t broadcast in stage %s!'\
        % (self.name, self.__stage))
    return self.__i_l_r, self.__p_l_r, self.__round_pulls

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
    self.__arm_num = arm_num
    self.__R = rounds
    self.__T = horizon
    self.__num_agents = num_agents

  def _name(self) -> str:
    return "lilUCBHeur_collaborative_master"

  def reset(self):
    self.__active_arms = list(range(self.__arm_num))

  def assign_arms(self, num_running_agents: int) -> List[List[int]]:
    # assumption: no agent has terminated
    # valid only for this particular algorithm
    self.__stage = "assign_arms"
    arms_assign_list = []

    def random_round(x: float) -> int:
      # x = 19.3
      # rounded to 19 with probability 0.7
      # rounded to 20 with probability 0.3
      frac_x = x - int(x)
      if np.random.uniform() > frac_x:
        return int(x)
      return int(x) + 1

    if len(self.__active_arms) < num_running_agents:
      num_agents_per_arm = num_running_agents / len(self.__active_arms)
      for arm in self.__active_arms[:-1]:
        arms_assign_list += [[arm]] * \
          random_round(num_agents_per_arm)
      arms_assign_list += [[self.__active_arms[-1]]] * \
        (num_running_agents - len(arms_assign_list))
      random.shuffle(arms_assign_list)
    else:
      __active_arms_copy = dcopy(self.__active_arms)
      random.shuffle(__active_arms_copy)
      for _ in range(num_running_agents):
        arms_assign_list.append([])

      for i, arm in enumerate(__active_arms_copy[:num_running_agents]):
        arms_assign_list[i].append(arm)
      for arm in __active_arms_copy[num_running_agents:]:
        agent_idx = int(np.random.randint(num_running_agents))
        arms_assign_list[agent_idx].append(arm)
    return arms_assign_list

  def elimination(self, i_l_r_list, p_l_r_list):
    s_tilde_r = np.array(list(set(i_l_r_list)))
    q_tilde_r = np.zeros_like(s_tilde_r, dtype="float64")
    i_l_r_list = np.array(i_l_r_list)
    p_l_r_list = np.array(p_l_r_list)

    for i, i_l_r in enumerate(s_tilde_r):
      q_tilde_r[i] = p_l_r_list[i_l_r_list == i_l_r].mean()
    # elimination
    confidence_radius = np.sqrt(
      self.__R * np.log(200 * self.__num_agents * self.__R) /
      (self.__T * max(1, self.__num_agents / len(self.__active_arms)))
    )
    best_q_i = np.max(q_tilde_r)
    self.__active_arms = list(
      s_tilde_r[q_tilde_r >= best_q_i - 2 * confidence_radius]
    )

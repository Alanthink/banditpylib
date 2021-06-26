from typing import Tuple

import numpy as np
from banditpylib.data_pb2 import Feedback, Actions

from .utils import CollaborativeLearner
from .collaborative_master import CollaborativeMaster
from .lilucb_heur_collaborative import LilUCBHeuristicCollaborative


class CollaborativeAgent(CollaborativeLearner):
  r"""One individual agent of the Collaborative Learning Algorithm

  :param int arm_num: number of arms of the bandit
  :param int num_rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int time_horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: total number of agents involved
  :param CollaborativeMaster master: the master controlling all agents
  :param str name: alias name
  """

  def __init__(self, arm_num: int, num_rounds: int,
    time_horizon: int, num_agents: int,
    master: CollaborativeMaster, name: str = None):
    super().__init__(arm_num=arm_num, name=name, rounds=num_rounds,
      horizon=time_horizon, num_agents=num_agents, master=master)
    self.__num_pulls_learning = int(0.5 * time_horizon / num_rounds)

  def _name(self) -> str:
    return 'collaborative_agent'

  def reset(self):
    self.__total_pulls = 0
    self.__round_pulls = 0
    self.__round_num = 0
    self.__stage = "unassigned"

    self.__central_algo_action_taken = False
    # True if action forwarded from central algo

  def assign_arms(self, arms):
    if self.__stage != "unassigned":
      raise Exception('%s: I can\'t be assigned arms in stage %s!'\
        % (self.name, self.__stage))
    self.__assigned_arms = np.array(arms)
    # confidence of 0.01 suggested in the paper
    self.__central_algo = LilUCBHeuristicCollaborative(self.arm_num,
      0.99, self.__assigned_arms)
    self.__central_algo.reset()
    self.__stage = "preparation"

  def complete_round(self):
    # completes round
    self.__round_num += 1
    self.__round_pulls = 0
    self.__central_algo_action_taken = False
    if self.__round_num < self.rounds + 1:
      self.__stage = "unassigned"
    else:
      self.__stage = "termination"

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
      if self.__central_algo.get_total_pulls() >= self.horizon//2:
        self.__stage = "learning"
        self.__i_l_r = None
        return self.actions()

      central_algo_actions = self.__central_algo.actions()
      if not central_algo_actions.arm_pulls_pairs:
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
        arm_pulls_pair = actions.arm_pulls_pairs.add()
        arm_pulls_pair.arm.id = self.__i_l_r # pylint: disable=protobuf-type-error
        arm_pulls_pair.pulls = self.__num_pulls_learning
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
    for arm_rewards_pair in feedback.arm_rewards_pairs:
      num_pulls += len(arm_rewards_pair.rewards)
    self.__total_pulls += num_pulls
    self.__round_pulls += num_pulls

    # handle feedback
    self.__p_l_r = None # default in case i_l_r is None
    if self.__central_algo_action_taken:
      self.__central_algo.update(feedback)
    elif num_pulls>0:
      # non-zero pulls not by central_algo => learning step was done
      for arm_rewards_pair in feedback.arm_rewards_pairs:
        if arm_rewards_pair.arm.id == self.__i_l_r:
          self.__p_l_r = np.array(arm_rewards_pair.rewards).mean()
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
    # broadcasts learnt information in the current round
    if self.__stage != "communication":
      raise Exception('%s: I can\'t broadcast in stage %s!'\
        % (self.name, self.__stage))
    return self.__i_l_r, self.__p_l_r, self.__round_pulls

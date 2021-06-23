from typing import Tuple

import numpy as np
from banditpylib.data_pb2 import Feedback, Actions

from .utils import CollaborativeLearner
from .lilucb_heur_collaborative import LilUCBHeuristicCollaborative

class CollaborativeAgent(CollaborativeLearner):
  r"""One individual agent of the Collaborative Learning Algorithm

  :param int arm_num: number of arms of the bandit
  :param int num_rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int time_horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: total number of agents involved
  :param str name: alias name
  """

  def __init__(self, arm_num: int, num_rounds: int,
    time_horizon: int, num_agents: int, name: str = None):
    super().__init__(arm_num=arm_num, name=name, rounds=num_rounds,
      horizon=time_horizon, num_agents=num_agents)
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

  def get_stage(self) -> str:
    return self.__stage

  def broadcast(self) -> Tuple[int, float, int]:
    # broadcasts learnt information in the current round
    return self.__i_l_r, self.__p_l_r, self.__round_pulls


# class MyCollaborativeMaster(CollaborativeMaster):
#   r"""Controlling master of the Collaborative Learning Algorithm

#   :param int arm_num: number of arms of the bandit
#   :param int num_rounds: number of rounds of communication allowed
#     (this agent uses one more)
#   :param int time_horizon: maximum number of pulls the agent can make
#     (over all rounds combined)
#   :param int num_agents: number of agents
#   :param str name: alias name
#   """

#   def __init__(self, arm_num: int, num_rounds: int,
#     time_horizon: int, num_agents: int, name: str = None):
#     super().__init__(arm_num=arm_num, num_agents=num_agents,
#       agent_class=CollaborativeAgent, name=name)
#     self.__R = num_rounds
#     self.__T = time_horizon
#     self.__agents = []
#     for i in range(num_agents):
#       if name is None:
#         agent_name = "collaborative_agent_" + str(i)
#       else:
#         agent_name = name + "_agent_" + str(i)
#       self.__agents.append(self.agent_class(
#         arm_num, num_rounds, time_horizon, agent_name))

#   def __assign_arms(self):
#     self.__stage = "assign_arms"
#     arms_assign_list = []

#     def random_round(x: float) -> int:
#       # x = 19.3
#       # rounded to 19 with probability 0.7
#       # rounded to 20 with probability 0.3
#       frac_x = x - int(x)
#       if np.random.uniform() > frac_x:
#         return int(x)
#       return int(x) + 1

#     if len(self.__active_arms) < len(self.__agents):
#       num_agents_per_arm = len(self.__agents) / len(self.__active_arms)
#       for arm in self.__active_arms[:-1]:
#         arms_assign_list += [[arm]] * \
#           random_round(num_agents_per_arm)
#       arms_assign_list += [[self.__active_arms[-1]]] * \
#         (len(self.__agents) - len(arms_assign_list))
#       random.shuffle(arms_assign_list)
#     else:
#       __active_arms_copy = dcopy(self.__active_arms)
#       random.shuffle(__active_arms_copy)
#       for _ in range(len(self.__agents)):
#         arms_assign_list.append([])

#       for i, arm in enumerate(__active_arms_copy[:len(self.__agents)]):
#         arms_assign_list[i].append(arm)
#       for arm in __active_arms_copy[len(self.__agents):]:
#         agent_idx = int(np.random.randint(len(self.__agents)))
#         arms_assign_list[agent_idx].append(arm)

#     for i, agent in enumerate(self.__agents):
#       agent.assign_arms(arms_assign_list[i])

#   def reset(self):
#     self.__active_arms = list(range(self.arm_num))
#     for agent in self.__agents:
#       agent.reset()
#     self.__round_num = 0
#     self.__total_pulls = 0
#     self.__assign_arms()

#   def _name(self) -> str:
#     return 'collaborative_master'

#   def update(self, feedback: Feedback):
#     if self.__stage == "preparation_learning":
#       self.__agents[self.__current_agent_idx].update(feedback)

#     else:
#       raise Exception("%s: cannot update with this feedback" % self.name)

#   def iterable_actions(self, context=None) \
#     -> Iterable[Actions]:
#     while self.__round_num < self.__R + 1 \
#       and self.__total_pulls < self.__T and \
#       len(self.__active_arms)>1:

#       # preparation and learning
#       self.__stage = "preparation_learning"
#       # waiting for communication
#       waiting_agents = [False] * len(self.__agents)
#       stopped_agents = [False] * len(self.__agents) # terminated
#       while sum(waiting_agents) + sum(stopped_agents) != len(self.__agents):
#         for i, agent in enumerate(self.__agents):
#           self.__current_agent_idx = i # to be used in update
#           if not (waiting_agents[i] or stopped_agents[i]):
#             actions = agent.actions(context)
#             if actions.state == Actions.WAIT:
#               waiting_agents[i] = True
#             elif actions.state == Actions.STOP:
#               stopped_agents[i] = True
#             else:
#               yield actions
#       # other stages
#       self.__communication_aggregation_elimination()

#       # complete_round routine
#       self.__round_num += 1
#       for agent in self.__agents:
#         agent.complete_round()
#       self.__assign_arms()

#   def __communication_aggregation_elimination(self):
#     self.__stage = "communication_aggregation_elimination"
#     # communication and aggregation
#     i_l_r_list, p_l_r_list, pulls_used_list = [], [], []
#     for agent in self.__agents:
#       i_l_r, p_l_r, pulls_used = agent.broadcast()
#       pulls_used_list.append(pulls_used)
#       if i_l_r is not None:
#         i_l_r_list.append(i_l_r)
#         p_l_r_list.append(p_l_r)

#     self.__total_pulls += max(pulls_used_list)

#     s_tilde_r = np.array(list(set(i_l_r_list)))
#     q_tilde_r = np.zeros_like(s_tilde_r, dtype="float64")
#     i_l_r_list = np.array(i_l_r_list)
#     p_l_r_list = np.array(p_l_r_list)

#     for i, i_l_r in enumerate(s_tilde_r):
#       q_tilde_r[i] = p_l_r_list[i_l_r_list == i_l_r].mean()
#     # elimination
#     confidence_radius = np.sqrt(
#       self.__R * np.log(200 * len(self.__agents) * self.__R) /
#       (self.__T * max(1, len(self.__agents) / len(self.__active_arms)))
#     )
#     best_q_i = np.max(q_tilde_r)
#     self.__active_arms = list(
#       s_tilde_r[q_tilde_r >= best_q_i - 2 * confidence_radius]
#     )

#   @property
#   def best_arm(self):
#     if len(self.__active_arms) == 1:
#       return self.__active_arms[0]
#     raise Exception('%s: I don\'t have an answer yet!' % self.name)

#   @property
#   def data(self):
#     return (self.__round_num, self.__total_pulls)

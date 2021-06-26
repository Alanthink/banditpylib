from typing import Tuple
import random
from copy import deepcopy as dcopy

import numpy as np

class CollaborativeMaster():
  r"""Controlling master of the Collaborative Learning Algorithm

  :param int arm_num: number of arms of the bandit
  :param int num_rounds: number of rounds of communication allowed
    (this agent uses one more)
  :param int time_horizon: maximum number of pulls the agent can make
    (over all rounds combined)
  :param int num_agents: number of agents
  :param str name: alias name
  """

  def __init__(self, arm_num:int, num_rounds: int,
    time_horizon: int, num_agents: int, name: str = None):
    self.__arm_num = arm_num
    self.__R = num_rounds
    self.__T = time_horizon
    self.__num_agents = num_agents
    self.__name = name

  @property
  def name(self) -> str:
    if self.__name is None:
      return "collaborative_master"
    return self.__name

  def reset(self):
    self.__active_arms = list(range(self.__arm_num))

  def assign_arms(self, agents):
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

    if len(self.__active_arms) < len(agents):
      num_agents_per_arm = len(agents) / len(self.__active_arms)
      for arm in self.__active_arms[:-1]:
        arms_assign_list += [[arm]] * \
          random_round(num_agents_per_arm)
      arms_assign_list += [[self.__active_arms[-1]]] * \
        (len(agents) - len(arms_assign_list))
      random.shuffle(arms_assign_list)
    else:
      __active_arms_copy = dcopy(self.__active_arms)
      random.shuffle(__active_arms_copy)
      for _ in range(len(agents)):
        arms_assign_list.append([])

      for i, arm in enumerate(__active_arms_copy[:len(agents)]):
        arms_assign_list[i].append(arm)
      for arm in __active_arms_copy[len(agents):]:
        agent_idx = int(np.random.randint(len(agents)))
        arms_assign_list[agent_idx].append(arm)

    for i, agent in enumerate(agents):
      agent.assign_arms(arms_assign_list[i])

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
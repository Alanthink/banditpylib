from typing import List
from copy import deepcopy as dcopy
import random

import numpy as np
from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, Arm, Actions
from banditpylib.learners import IdentifyBestArm
from banditpylib.learners.collaborative_learner import CollaborativeLearner
from .utils import Protocol


class CollaborativeLearningProtocol(Protocol):
  """Collaborative learning protocol

  This protocol is used to simulate the multi-agent game
  as discussed in the paper. It runs in rounds. During each round,
  the protocol runs the following steps in sequence:

  * fetch the state of the environment and ask each learner for actions;
  * send the actions to the enviroment for execution;
  * update each learner with the corresponding feedback of the environment;
  * repeat the above steps until every agent enters the WAIT state;
  * receive information broadcasted from every agent,
    and use it to decide the parameters of the next round.

  The simulation stopping criteria is:

  * every agent enters STOP state;

  Algorithm is guaranteed to stop before total number of time-steps
  achieve `horizon`.

  .. todo::
    extend protocol to compare different types of agents
    or different num_agent/round/horizon combos


  :param Bandit bandit: bandit environment
  :param List[CollaborativeLearner] agents: agent classes to be used
  :param List[int] num_agents: number of agents per class

  .. note::
    During a timestep, a learner may want to perform multiple actions, which is
    so-called batched learner. In this case, eah action counts as a timestep
    used.
  """
  def __init__(self,
               bandit: Bandit, agents: List[CollaborativeLearner]):
    super().__init__(bandit=bandit, learners=agents)

  @property
  def name(self) -> str:
    return 'collaborative_learning_protocol'

  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    def assign_arms():
      # function to assign arms to agents
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

      if len(self.__active_arms) < len(self.__agents):
        num_agents_per_arm = len(self.__agents) / len(self.__active_arms)
        for arm in self.__active_arms[:-1]:
          arms_assign_list += [[arm]] * \
            random_round(num_agents_per_arm)
        arms_assign_list += [[self.__active_arms[-1]]] * \
          (len(self.__agents) - len(arms_assign_list))
        random.shuffle(arms_assign_list)
      else:
        active_arms_copy = dcopy(self.__active_arms)
        random.shuffle(active_arms_copy)
        for _ in range(len(self.__agents)):
          arms_assign_list.append([])

        for i, arm in enumerate(active_arms_copy[:len(self.__agents)]):
          arms_assign_list[i].append(arm)
        for arm in active_arms_copy[len(self.__agents):]:
          agent_idx = int(np.random.randint(len(self.__agents)))
          arms_assign_list[agent_idx].append(arm)

      for i, agent in enumerate(self.__agents):
        agent.assign_arms(arms_assign_list[i])

    def communication_aggregation_elimination():
      self.__stage = "communication_aggregation_elimination"
      # communication and aggregation
      i_l_r_list, p_l_r_list, pulls_used_list = [], [], []
      for agent in self.__agents:
        i_l_r, p_l_r, pulls_used = agent.broadcast()
        pulls_used_list.append(pulls_used)
        if i_l_r is not None:
          i_l_r_list.append(i_l_r)
          p_l_r_list.append(p_l_r)

      self.__total_pulls += max(pulls_used_list)

      s_tilde_r = np.array(list(set(i_l_r_list)))
      q_tilde_r = np.zeros_like(s_tilde_r, dtype="float64")
      i_l_r_list = np.array(i_l_r_list)
      p_l_r_list = np.array(p_l_r_list)

      for i, i_l_r in enumerate(s_tilde_r):
        q_tilde_r[i] = p_l_r_list[i_l_r_list == i_l_r].mean()
      # elimination
      confidence_radius = np.sqrt(
        self.__R * np.log(200 * len(self.__agents) * self.__R) /
        (self.__T * max(1, len(self.__agents) / len(self.__active_arms)))
      )
      best_q_i = np.max(q_tilde_r)
      self.__active_arms = list(
        s_tilde_r[q_tilde_r >= best_q_i - 2 * confidence_radius]
      )

    def add_data():
      data_item = trial.data_items.add()
      data_item.rounds = self.__round_num
      data_item.total_actions = self.__total_pulls
      if len(self.__active_arms) == 1:
        arm = Arm()
        arm.id = self.__active_arms[0]
        goal = IdentifyBestArm(best_arm=arm)
        data_item.regret = self.bandit.regret(goal)
      else:
        data_item.regret = 1


    # Initialization
    self.__bandits = []
    self.__agents = []
    for _ in range(self.current_learner.num_agents):
      self.__bandits.append(dcopy(self.bandit))
      self.__agents.append(dcopy(self.current_learner))
      # reset all bandits and agents
      self.__bandits[-1].reset()
      self.__agents[-1].reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = self.current_learner.name

    self.__R = self.current_learner.rounds
    self.__T = self.current_learner.horizon

    self.__active_arms = list(range(self.current_learner.arm_num))
    self.__round_num = 0
    self.__total_pulls = 0
    assign_arms()

    while self.__round_num < self.__R + 1 and self.__total_pulls < self.__T\
      and len(self.__active_arms)>1:

      # preparation and learning
      self.__stage = "preparation_learning"
      waiting_agents = [False] * len(self.__agents) # waiting for communication
      stopped_agents = [False] * len(self.__agents) # terminated
      while sum(waiting_agents) + sum(stopped_agents) != len(self.__agents):
        for i, agent in enumerate(self.__agents):
          self.__current_agent_idx = i # to be used in update
          if not (waiting_agents[i] or stopped_agents[i]):
            actions = agent.actions(
              self.__bandits[self.__current_agent_idx].context)
            if actions.state == Actions.WAIT:
              waiting_agents[i] = True
            elif actions.state == Actions.STOP:
              stopped_agents[i] = True
            else:
              feedback = self.__bandits[self.__current_agent_idx].feed(actions)
              self.__agents[self.__current_agent_idx].update(feedback)

      # other stages
      communication_aggregation_elimination()

      # complete_round routine
      self.__round_num += 1
      for agent in self.__agents:
        agent.complete_round()
      assign_arms()

    add_data()

    return trial.SerializeToString()

from typing import List, Tuple
from copy import deepcopy as dcopy
import random

import numpy as np

from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, CollaborativeActions
from banditpylib.learners.collaborative_learner import CollaborativeLearner
from .utils import Protocol


class CollaborativeLearningProtocol(Protocol):
  """Collaborative learning protocol

  This protocol is used to simulate the multi-agent game as discussed in the paper. It runs in
  rounds. During each round, the protocol runs the following steps in sequence:

  * fetch the state of the environment and ask each learner for actions;
  * send the actions to the enviroment for execution;
  * update each learner with the corresponding feedback of the environment;
  * repeat the above steps until every agent enters the WAIT state;
  * receive information broadcasted from every agent,
    and use it to decide the parameters of the next round.

  The simulation stopping criteria is:

  * every agent enters STOP state;
  
  Algorithm is guaranteed to stop before total number of time-steps achieve `horizon`.

  .. todo::
    extend protocol to compare different types of agents
    or different num_agent/round/horizon combos


  :param Bandit bandit: bandit environment
  :param CollaborativeLearner agent: agent class to be used
  :param int num_agents: number of such agents
  :param int rounds: number of times communication is allowed
    (this particular algorithm uses one extra communication round)
  :param int horizon: horizon of the game (i.e., total number of actions each
    leaner can make)

  .. note::
    During a timestep, a learner may want to perform multiple actions, which is
    so-called batched learner. In this case, eah action counts as a timestep
    used.
  """
  def __init__(self,
               bandit: Bandit,
               agent: CollaborativeLearner,
               num_agents: int,
               rounds: int,
               horizon: int):
    super().__init__(bandit=bandit, learners=[agent])
    self.__horizon = horizon
    self.__rounds = rounds
    self.__num_agents = num_agents

  @property
  def name(self) -> str:
    return 'collaborative_learning_protocol'

  def __random_round(self, x: float) -> int:
    # x = 19.3
    # rounded to 19 with probability 0.7
    # rounded to 20 with probability 0.3
    frac_x = x - int(x)
    if np.random.uniform() > frac_x:
        return int(x)
    return int(x) + 1

  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    # Reset the bandit environment and the learner
    self.bandit.reset()
    self.__agents = []
    for _ in range(self.__num_agents):
      self.__agents.append(dcopy(self.current_learner))
      self.__agents[-1].reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = self.current_learner.name
    current_round = 0
    # Number of time_steps used per round = max(number of actions by any agent)
    total_actions = 0

    def add_data():
      data_item = trial.data_items.add()
      data_item.rounds = current_round
      data_item.total_actions = total_actions
      data_item.regret = self.bandit.regret(self.__agents[0].goal)

    self.active_arms = list(range(self.bandit.arm_num))
    while current_round < self.__rounds + 1 and total_actions < self.__horizon and \
      len(self.active_arms)>1:
      # assigning arms
      arms_assign_list = []

      if len(self.active_arms) < self.__num_agents:
        num_agents_per_arm = self.__num_agents / len(self.active_arms)
        for arm in self.active_arms[:-1]:
          arms_assign_list += [[arm]] * \
            self.__random_round(num_agents_per_arm)
        arms_assign_list += [[self.active_arms[-1]]] * \
          (self.__num_agents - len(arms_assign_list))
        random.shuffle(arms_assign_list)
      else:
        active_arms_copy = dcopy(self.active_arms)
        random.shuffle(active_arms_copy)
        for _ in range(self.__num_agents):
          arms_assign_list.append([])

        for i, arm in enumerate(active_arms_copy[:self.__num_agents]):
          arms_assign_list[i].append(arm)
        for arm in active_arms_copy[self.__num_agents:]:
          agent_idx = np.random.randint(self.__num_agents)
          arms_assign_list[agent_idx].append(arm)

      for i, agent in enumerate(self.__agents):
        agent.assign_arms(arms_assign_list[i])

      # preparation and learning
      waiting_agents = [False] * self.__num_agents # waiting for communication
      stopped_agents = [False] * self.__num_agents # terminated
      while sum(waiting_agents) + sum(stopped_agents) != self.__num_agents:
        for i, agent in enumerate(self.__agents):
          if not (waiting_agents[i] or stopped_agents[i]):
            actions = agent.actions(self.bandit.context)
            if actions.state == CollaborativeActions.WAIT:
              waiting_agents[i] = True
            elif actions.state == CollaborativeActions.STOP:
              stopped_agents[i] = True
            else:
              feedback = self.bandit.feed(actions)
              agent.update(feedback)

      # communication and aggregation
      i_l_r_list, p_l_r_list, pulls_used_list = [], [], []
      for agent in self.__agents:
        i_l_r, p_l_r, pulls_used = agent.broadcast()
        pulls_used_list.append(pulls_used)
        if i_l_r is not None:
          i_l_r_list.append(i_l_r)
          p_l_r_list.append(p_l_r)

      total_actions += max(pulls_used_list)

      s_tilde_r = np.array(list(set(i_l_r_list)))
      q_tilde_r = np.zeros_like(s_tilde_r)
      i_l_r_list = np.array(i_l_r_list)
      p_l_r_list = np.array(p_l_r_list)

      for i, i_l_r in enumerate(s_tilde_r):
        q_tilde_r[i] = np.mean(
          p_l_r_list[i_l_r_list == i_l_r]
        )

      # elimination
      confidence_radius = np.sqrt(
        self.__rounds * np.log(200 * self.__num_agents * self.__rounds) /
        (self.__horizon * max(1, self.__num_agents / len(self.active_arms)))
      )
      best_q_i = np.max(q_tilde_r)
      self.active_arms = list(
        s_tilde_r[q_tilde_r >= best_q_i - 2 * confidence_radius]
      )

      for agent in self.__agents:
        agent.complete_round()

      add_data()

    return trial.SerializeToString()
            

    while total_actions < self.__horizon:
      
      actions = self.current_learner.actions(self.bandit.context)

      # Stop the game if no actions are returned by the learner
      if not actions.arm_pulls_pairs:
        break

      # Record intermediate regrets
      if rounds in self.__intermediate_regrets:
        add_data()

      feedback = self.bandit.feed(actions)
      self.current_learner.update(feedback)

      for arm_pulls_pair in actions.arm_pulls_pairs:
        total_actions += arm_pulls_pair.pulls
      rounds += 1

    # Record final regret
    add_data()
    return trial.SerializeToString()

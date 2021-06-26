from typing import List
from copy import deepcopy as dcopy

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

    # initialization
    agents = []
    bandits = []
    master = self.current_learner.master
    master.reset()
    for _ in range(self.current_learner.num_agents):
      agents.append(dcopy(self.current_learner))
      agents[-1].reset()
      bandits.append(dcopy(self.bandit))
      bandits[-1].reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = self.current_learner.name

    round_num, total_pulls_used = 0, 0

    # stages of the algorithms
    stopped_agents = [False] * len(agents) # terminated
    while True:
      # using only non-stopped agents
      running_agents = []
      for i, agent in enumerate(agents):
        if not stopped_agents[i]:
          running_agents.append(agent)
      if len(running_agents) == 0:
        break

      master.assign_arms(running_agents)
      waiting_agents = [False] * len(agents) # waiting for communication

      # preparation and learning
      for i, agent in enumerate(agents):
        while True:
          actions = agent.actions(bandits[i].context)
          if actions.state == Actions.WAIT:
            waiting_agents[i] = True
            break
          elif actions.state == Actions.STOP:
            stopped_agents[i] = True
            break
          else:
            feedback = bandits[i].feed(actions)
            agent.update(feedback)
      if sum(stopped_agents) == len(agents):
        break

      # communication and aggregation
      i_l_r_list, p_l_r_list, pulls_used_list = [], [], []
      for i, agent in enumerate(agents):
        if waiting_agents[i]:
          i_l_r, p_l_r, pulls_used = agent.broadcast()
          pulls_used_list.append(pulls_used)
          if i_l_r is not None:
            i_l_r_list.append(i_l_r)
            p_l_r_list.append(p_l_r)

      # send info to master for elimination
      # todo: handle empty i_l_r_list
      master.elimination(i_l_r_list, p_l_r_list)

      for i, agent in enumerate(agents):
        agent.complete_round()
        if agent.stage == "termination":
          stopped_agents[i] = True

      round_num += 1
      total_pulls_used += max(pulls_used_list)

    # add data
    data_item = trial.data_items.add()
    data_item.rounds = round_num
    data_item.total_actions = total_pulls_used
    total_regret = 0.0
    for i, agent in enumerate(agents):
      arm = Arm()
      best_arm = agent.best_arm
      if best_arm is None:
        total_regret += 1
      else:
        arm.id = best_arm
        goal = IdentifyBestArm(best_arm=arm)
        total_regret += bandits[i].regret(goal)
    data_item.regret = total_regret/len(agents)

    return trial.SerializeToString()

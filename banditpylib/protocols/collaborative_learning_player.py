from typing import List, Tuple
from copy import deepcopy as dcopy
import random

import numpy as np

from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial, CollaborativeActions
from banditpylib.learners.collaborative_learner import CollaborativeMaster
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
               master: CollaborativeMaster,
               rounds: int,
               horizon: int):
    super().__init__(bandit=bandit, learners=[master])
    self.__horizon = horizon
    self.__rounds = rounds

  @property
  def name(self) -> str:
    return 'collaborative_learning_protocol'

  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    if debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    # Reset the bandit environment and the learner
    self.bandit.reset()
    self.current_learner.reset()

    trial = Trial()
    trial.bandit = self.bandit.name
    trial.learner = self.current_learner.name
    current_round = 0
    # Number of time_steps used per round = max(number of actions by any agent)
    total_actions = 0

    def add_data():
      data_item = trial.data_items.add()
      data_item.rounds = self.current_learner.num_rounds_completed
      data_item.total_actions = self.current_learner.total_pulls
      try:
        print(self.current_learner.best_arm)
        data_item.regret = self.bandit.regret(self.current_learner.goal)
      except:
        data_item.regret = 1

    while self.current_learner.num_rounds_completed < self.__rounds + 1 \
      and self.current_learner.total_pulls < self.__horizon and \
      self.current_learner.num_active_arms>1:
      
      # Record intermediate regrets
      # if self.current_learner.num_rounds_completed in self.__intermediate_regrets:
      #   add_data()

      for actions in self.current_learner.preparation_learning():
        feedback = self.bandit.feed(actions)
        self.current_learner.update(feedback)
      self.current_learner.complete_round()

    add_data()

    return trial.SerializeToString()

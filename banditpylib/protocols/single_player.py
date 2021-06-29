from typing import List

import numpy as np

from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial
from banditpylib.learners import Learner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This protocol is used to simulate the ordinary single-player game. It runs in
  rounds. During each round, the protocol runs the following steps in sequence:

  * fetch the state of the environment and ask the learner for actions;
  * send the actions to the enviroment for execution;
  * update the learner with the feedback of the environment.

  The simulation stopping criteria is one of the following two:

  * no actions are returned by the learner;
  * total number of actions achieve `horizon`.


  :param Bandit bandit: bandit environment
  :param List[Learner] learner: learners to be compared with
  :param List[int] intermediate_regrets: a list of rounds. If set, the regrets
    after these rounds will be recorded
  :param int horizon: horizon of the game (i.e., total number of actions a
    leaner can make)

  .. note::
    During a round, a learner may want to perform multiple actions, which is
    so-called batched learner. The total number of rounds shows how often the
    learner wants to communicate with the bandit environment which is at most
    `horizon`.
  """
  def __init__(self,
               bandit: Bandit,
               learners: List[Learner],
               intermediate_regrets: List[int] = None,
               horizon: int = np.inf):  # type: ignore
    super().__init__(bandit=bandit, learners=learners)
    self.__intermediate_regrets = \
        intermediate_regrets if intermediate_regrets is not None else []
    self.__horizon = horizon

  @property
  def name(self) -> str:
    return 'single_player_protocol'

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
    rounds = 0
    # Number of actions the learner has made
    total_actions = 0

    def add_result():
      result = trial.results.add()
      result.rounds = rounds
      result.total_actions = total_actions
      result.regret = self.bandit.regret(self.current_learner.goal)

    while total_actions < self.__horizon:
      actions = self.current_learner.actions(self.bandit.context)

      # Stop the game if no actions are returned by the learner
      if not actions.arm_pulls:
        break

      # Record intermediate regrets
      if rounds in self.__intermediate_regrets:
        add_result()

      feedback = self.bandit.feed(actions)
      self.current_learner.update(feedback)

      for arm_pull in actions.arm_pulls:
        total_actions += arm_pull.times
      rounds += 1

    # Record final regret
    add_result()
    return trial.SerializeToString()

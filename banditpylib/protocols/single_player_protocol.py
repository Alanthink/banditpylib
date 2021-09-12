from typing import List, cast

import numpy as np

from absl import logging

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial
from banditpylib.learners import Learner, SinglePlayerLearner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This class defines the communication protocol for the ordinary single-player
  game. The game runs in rounds and during each round, the protocol runs the
  following steps in sequence:

  * fetch the state of the bandit environment and ask the learner for actions;
  * send the actions to the bandit environment for execution;
  * update the learner with the feedback of the bandit environment.

  The game runs until one of the following two stopping conditions is satisfied:

  * no actions are returned by the learner;
  * total number of actions achieve `horizon`.

  :param Bandit bandit: bandit environment
  :param List[SinglePlayerLearner] learners: learners to be compared with

  .. note::
    During a round, a learner may want to perform multiple actions, which is
    so-called batched learner.
  """
  def __init__(self, bandit: Bandit, learners: List[SinglePlayerLearner]):
    super().__init__(bandit=bandit, learners=cast(List[Learner], learners))

  @property
  def name(self) -> str:
    return 'single_player_protocol'

  def _one_trial(self, random_seed: int) -> bytes:
    if self._debug:
      logging.set_verbosity(logging.DEBUG)
    np.random.seed(random_seed)

    # Reset the bandit environment and the learner
    self._bandit.reset()
    current_learner = cast(SinglePlayerLearner, self._current_learner)
    current_learner.reset()

    trial = Trial()
    trial.bandit = self._bandit.name
    trial.learner = current_learner.name
    rounds = 0
    # Number of actions the learner has made
    total_actions = 0

    def add_result():
      result = trial.results.add()
      result.rounds = rounds
      result.total_actions = total_actions
      result.regret = self._bandit.regret(current_learner.goal)

    while total_actions < self._horizon:
      actions = current_learner.actions(self._bandit.context)

      # Stop the game if no actions are returned by the learner
      if not actions.arm_pulls:
        break

      # Record intermediate regrets
      if rounds in self._intermediate_horizons:
        add_result()

      feedback = self._bandit.feed(actions)
      current_learner.update(feedback)

      for arm_pull in actions.arm_pulls:
        total_actions += arm_pull.times
      rounds += 1

    # Record final regret
    add_result()
    return trial.SerializeToString()

from typing import Dict

import numpy as np

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This protocol is used to simulate the ordinary single-player game.
  """
  def __init__(self,
               bandit: Bandit,
               learner: Learner,
               intermediate_regrets=None):
    """
    Args:
      bandit: bandit environment
      learner: learner
      intermediate_regrets: whether to record intermediate regrets
    """
    super().__init__(bandit, learner)
    self.__intermediate_regrets = \
        intermediate_regrets if intermediate_regrets else []

  @property
  def name(self):
    return 'single_player_protocol'

  def _one_trial(self, random_seed: int) -> Dict:
    np.random.seed(random_seed)

    # reset the bandit environment and the learner
    self.bandit.reset()
    self.learner.reset()

    one_trial_data = []
    # number of rounds to communicate with the bandit environment
    adaptive_rounds = 0
    # total actions executed by the bandit environment
    total_actions = 0
    while True:
      context = self.bandit.context()
      actions = self.learner.actions(context)

      # stop the game if actions returned by the learner are None
      if not actions:
        break

      # record intermediate regrets
      if adaptive_rounds in self.__intermediate_regrets:
        one_trial_data.append(
            dict({
                'bandit': self.bandit.name,
                'learner': self.learner.name,
                'rounds': adaptive_rounds,
                'total_actions': total_actions,
                'regret': self.learner.regret(self.bandit)
            }))

      feedback = self.bandit.feed(actions)
      self.learner.update(feedback)

      # information update
      for (_, times) in actions:
        total_actions += int(times)
      adaptive_rounds += 1

    # record final regret
    one_trial_data.append(
        dict({
            'bandit': self.bandit.name,
            'learner': self.learner.name,
            'rounds': adaptive_rounds,
            'total_actions': total_actions,
            'regret': self.learner.regret(self.bandit)
        }))
    return one_trial_data

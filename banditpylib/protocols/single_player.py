import copy
from typing import Dict

import numpy as np

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol

  This protocol is used to simulate the game when the learner only has one
  player and the learner only interacts with one bandit environment.
  """
  def __init__(self, bandit, learner, intermediate_regrets=None):
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

  def _one_trial(self, bandit: Bandit, learner: Learner,
                 random_seed: int) -> Dict:
    np.random.seed(random_seed)

    # make sure not changing the behavior of outside bandit environment
    # the learner
    bandit = copy.deepcopy(bandit)
    learner = copy.deepcopy(learner)

    # reset the bandit environment and the learner
    bandit.reset()
    learner.reset()

    one_trial_data = []
    # number of rounds to communicate with the bandit environment
    adaptive_rounds = 0
    # total actions executed by the bandit environment
    total_actions = 0
    while True:
      context = bandit.context()
      actions = learner.actions(context)

      # stop the game if actions returned by the learner are None
      if not actions:
        break

      # record intermediate regrets
      if adaptive_rounds in self.__intermediate_regrets:
        one_trial_data.append(
            dict({
                'bandit': bandit.name,
                'learner': learner.name,
                'rounds': adaptive_rounds,
                'total_actions': total_actions,
                'regret': learner.regret(bandit)
            }))

      feedback = bandit.feed(actions)
      learner.update(feedback)

      # information update
      for (_, times) in actions:
        total_actions += times
      adaptive_rounds += 1

    # record final regret
    one_trial_data.append(
        dict({
            'bandit': bandit.name,
            'learner': learner.name,
            'rounds': adaptive_rounds,
            'total_actions': total_actions,
            'regret': learner.regret(bandit)
        }))
    return one_trial_data

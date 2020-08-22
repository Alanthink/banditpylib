import copy
from typing import Dict

import numpy as np

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol
  """
  def __init__(self, bandit, learner, intermediate_regrets=None):
    """
    Args:
      bandit: bandit environment
      learner: learner
      intermediate_regrets: intermediate regrets to record
    """
    super().__init__(bandit, learner)
    self.__intermediate_regrets = intermediate_regrets

  @property
  def name(self):
    return 'single_player_protocol'

  def _one_trial(self, bandit: Bandit, learner: Learner,
                 random_seed: int) -> Dict:
    np.random.seed(random_seed)
    bandit = copy.deepcopy(bandit)
    learner = copy.deepcopy(learner)

    # reset
    bandit.reset()
    learner.reset()

    data = []

    rounds = 0
    total_actions = 0
    while True:
      if rounds in self.__intermediate_regrets:
        data.append(
            dict({
                'bandit': bandit.name,
                'learner': learner.name,
                'rounds': rounds,
                'total_actions': total_actions,
                'regret': learner.regret(bandit)
            }))
      context = bandit.context()
      actions = learner.actions(context)
      # When actions suggested by the learner is None, it means the learner
      # wants to stop.
      if not actions:
        break
      for (_, times) in actions:
        total_actions += times
      feedback = bandit.feed(actions)
      learner.update(feedback)
      rounds += 1

    data.append(
        dict({
            'bandit': bandit.name,
            'learner': learner.name,
            'rounds': rounds,
            'total_actions': total_actions,
            'regret': learner.regret(bandit)
        }))
    return data

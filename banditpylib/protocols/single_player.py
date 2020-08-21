import copy
from typing import Dict

import numpy as np

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner
from .utils import Protocol


class SinglePlayerProtocol(Protocol):
  """Single player protocol
  """
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

    rounds = 0
    total_actions = 0
    while True:
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
    return dict({
        'bandit': bandit.name,
        'learner': learner.name,
        'rounds': rounds,
        'total_actions': total_actions,
        'regret': learner.regret(bandit)
    })

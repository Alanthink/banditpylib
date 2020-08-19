from typing import Dict

from absl import flags

import numpy as np

from bandits import Bandit
from learners import Learner
from .utils import Protocol

FLAGS = flags.FLAGS


class SinglePlayerProtocol(Protocol):
  """Single player protocol
  """

  @property
  def name(self):
    return 'single_player_protocol'

  def _one_trial(
      self,
      bandit: Bandit,
      learner: Learner,
      random_seed: int) -> Dict:
    np.random.seed(random_seed)

    # reset
    bandit.reset()
    learner.reset()

    rounds = 0
    while True:
      context = bandit.context()
      actions = learner.actions(context)
      if not actions:
        break
      feedback = bandit.feed(actions)
      learner.update(feedback)
      rounds += 1
    return dict({'learner_name': learner.name,
                 'rounds': rounds,
                 'regret': learner.regret(bandit)})

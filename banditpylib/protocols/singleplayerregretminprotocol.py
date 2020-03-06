from absl import flags

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['SinglePlayerRegretMinProtocol']


class SinglePlayerRegretMinProtocol(Protocol):
  """single player regret minimization protocol
  """

  def __init__(self, pars=None):
    pass

  @property
  def type(self):
    return 'SinglePlayerRegretMinProtocol'

  @property
  def __horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    self._bandit.init()
    self._player.init(self._bandit)
    ############################################################################

    regrets = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._player.learner_choice(context)
        feedback = self._bandit.feed(action)
        self._player.update(context, action, feedback)
      if t % self.__frequency == 0:
        # call a private method
        regrets[t] = getattr(
            self._bandit,
            '_'+type(self._bandit).__name__+'__'+self._regret_def)(
                self._player.rewards_def())
    return dict({self._player.name: regrets})

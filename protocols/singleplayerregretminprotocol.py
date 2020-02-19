from absl import flags

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['SinglePlayerRegretMinProtocol']


class SinglePlayerRegretMinProtocol(Protocol):
  """Single Player Regret Minimization Protocol
  """

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
    self._player.init(self._bandit, self.__horizon)
    ############################################################################

    agg_regret = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._player.learner_choice(context)
        feedback = self._bandit.feed(action)
        self._player.update(context, action, feedback)
      if t % self.__frequency == 0:
        agg_regret[t] = self._bandit.regret(self._player.rewards)
    return dict({self._player.name: agg_regret})

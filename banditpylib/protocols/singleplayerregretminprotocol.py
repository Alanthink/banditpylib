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

  def __independent_run(self):
    regrets = dict()
    for t in range(self.__horizon + 1):
      if t % self.__frequency == 0:
        ########################################################################
        # initialization
        self._bandit.reset()
        self._player.reset(self._bandit)
        ########################################################################
        for _ in range(t):
          context = self._bandit.context
          action = self._player.learner_step(context)
          feedback = self._bandit.feed(action)
          self._player.update(context, action, feedback)
        regrets[t] = getattr(self._bandit, self._regret_funcname)(
            getattr(self._player, self._rewards_funcname)())
    return dict({self._player.name: regrets})

  def __dependent_run(self):
    ############################################################################
    # initialization
    self._bandit.reset()
    self._player.reset(self._bandit)
    ############################################################################

    regrets = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._player.learner_step(context)
        feedback = self._bandit.feed(action)
        self._player.update(context, action, feedback)
      if t % self.__frequency == 0:
        regrets[t] = getattr(self._bandit, self._regret_funcname)(
            getattr(self._player, self._rewards_funcname)())
    return dict({self._player.name: regrets})

  def _one_trial(self, seed):
    np.random.seed(seed)
    if self._ind_run:
      return self.__independent_run()
    return self.__dependent_run()

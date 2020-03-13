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


  def _one_trial(self, seed, stop_cond):
    # stop_cond is horizon
    np.random.seed(seed)

    ############################################################################
    # initialization
    self._bandit.reset()
    self._player.reset(self._bandit, stop_cond)
    ############################################################################

    regrets = dict()
    regrets[0] = 0.0

    if len(self._pars['horizons']) > 1:
      # when there are multiple horizons, do not record intermedaite regrets
      freq = stop_cond
    else:
      freq = self._pars['freq']

    # simulation starts from t = 1
    for t in range(1, stop_cond + 1):
      context = self._bandit.context
      action = self._player.learner_step(context)
      feedback = self._bandit.feed(action)
      self._player.update(context, action, feedback)
      if t % freq == 0:
        # record the intermediate regret
        regrets[t] = getattr(self._bandit, self._regret_funcname)(
            getattr(self._player, self._rewards_funcname)())
    return dict({self._player.name: regrets})

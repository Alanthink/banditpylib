from random import randint

from absl import flags

import numpy as np

from banditpylib.bandits.arms import EmArm
from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedRegretMinProtocol']


class DecentralizedRegretMinProtocol(Protocol):
  """decentralized regret minimization protocol :cite:`feraud2018decentralized`.
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      raise Exception('%s: number of players is not specified!' % self.type)
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedRegretMinProtocol'

  def _one_round(self):
    # sample player
    k = randint(0, self.__num_players-1)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast message
    context = bandit.context
    action = player.learner_step(context, self.__messages)
    feedback = bandit.feed(action)
    player.update(context, action, feedback)
    message = player.broadcast_message(context, action, feedback)
    if message:
      self.__messages[message[0]].update(message[1])

  def _one_trial(self, seed, stop_cond):
    np.random.seed(seed)
    self.__messages = [EmArm() for ind in range(self._bandits[0].arm_num)]

    ############################################################################
    # initialization
    for k in range(self.__num_players):
      bandit = self._bandits[k]
      player = self._players[k]
      bandit.reset()
      player.reset(bandit, stop_cond)
    ############################################################################

    if len(self._pars['horizons']) > 1:
      # when there are multiple horizons, do not record intermedaite regrets
      freq = stop_cond
    else:
      freq = self._pars['freq']

    regrets = dict()
    regrets[0] = 0.0
    for t in range(1, stop_cond + 1):
      self._one_round()
      if t % freq == 0:
        regrets[t] = self.__regret()
    return dict({self._players[0].name: regrets})

  def __regret(self):
    return sum([getattr(self._bandits[k], self._regret_funcname)(
        getattr(self._players[k], self._rewards_funcname)())
                for k in range(self.__num_players)])

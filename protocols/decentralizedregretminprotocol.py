from random import randint

from absl import flags
from absl import logging

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedRegretMinProtocol']


class DecentralizedRegretMinProtocol(Protocol):
  """decentralized regret minimization protocol
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      logging.fatal('%s: number of players is not specified!' % self.type)
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedRegretMinProtocol'

  @property
  def __horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  def _one_round(self):
    # sample player
    k = randint(0, self.__num_players-1)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast message
    context = bandit.context
    action = player.learner_choice(context, self.__messages)
    feedback = bandit.feed(action)
    player.update(context, action, feedback)
    message = player.broadcast_message(context, action, feedback)
    self.__messages.append({player.name: message})

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    for k in range(self.__num_players):
      bandit = self._bandits[k]
      player = self._players[k]
      bandit.init()
      player.init(bandit)
    ############################################################################

    agg_regret = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        self._one_round()
      if t > 0 and t % self.__frequency == 0:
        agg_regret[t] = self.__regret()
    return dict({self._players[0].name: agg_regret})

  def __regret(self):
    return sum([self._bandits[k].regret(self._players[k].rewards)
                for k in range(self.__num_players)])

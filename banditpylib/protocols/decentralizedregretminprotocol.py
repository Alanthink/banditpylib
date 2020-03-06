from random import randint

from absl import flags

import numpy as np

from banditpylib.bandits.arms import EmArm
from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedRegretMinProtocol']


class DecentralizedRegretMinProtocol(Protocol):
  """decentralized regret minimization protocol
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      raise Exception('%s: number of players is not specified!' % self.type)
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
    if message:
      self.__messages[message[0]].update(message[1])

  def _one_trial(self, seed):
    np.random.seed(seed)
    self.__messages = [EmArm() for ind in range(self._bandits[0].arm_num)]

    ############################################################################
    # initialization
    for k in range(self.__num_players):
      bandit = self._bandits[k]
      player = self._players[k]
      bandit.init()
      player.init(bandit)
    ############################################################################

    regrets = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        self._one_round()
      if t > 0 and t % self.__frequency == 0:
        regrets[t] = self.__regret()
    return dict({self._players[0].name: regrets})

  def __regret(self):
    return sum([getattr(
        self._bandits[k],
        '_'+type(self._bandits[k]).__name__+'__'+self._regret_def)(
            self._players[k].rewards_def())
                for k in range(self.__num_players)])

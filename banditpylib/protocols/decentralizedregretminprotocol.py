from random import randint

from absl import flags

import numpy as np

from banditpylib.bandits.arms import EmArm
from .utils import Protocol

FLAGS = flags.FLAGS


class DecentralizedRegretMinProtocol(Protocol):
  """Decentralized regret minimization protocol :cite:`feraud2018decentralized`
  """

  def __init__(self, pars):
    """
    Args:
      pars:
        ``"num_players"`` (int): number of players.
        ``"horizons"`` ([int,]): horizons. When the length is 1,
        ``"freq"`` will be used to record the intermedaite regrets and plot the
        final figure. Otherwise, run each horizon in ``"horizons"``
        independently and use the regrets to plot the final figure.
        ``"freq"`` (int, optional): frequency to record intermediate regrets.
        ``"trials"`` (int): number of repetitions of the game.
        ``"processors"`` (int): maximum number of processors can be used. -1
        means trying to make use all available cpus.

    .. warning::
      To ensure the independence between different horizons in the final figure,
      make sure ``"horizons"`` is a list of two or more elements.
    """
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

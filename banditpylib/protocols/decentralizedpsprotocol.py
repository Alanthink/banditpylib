from random import choice

from absl import flags

import numpy as np

from banditpylib.bandits.arms import EmArm
from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedPEProtocol']


class DecentralizedPEProtocol(Protocol):
  """decentralized pure exploration protocol :cite:`feraud2018decentralized`.
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      raise Exception('%s: number of players is not specified!' % self.type)
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedPEProtocol'

  def __one_round(self):
    # sample player
    k = choice(self.__playable_players)
    player = self._players[k]

    # play player and broadcast message
    state = player.learner_round(self.__messages)
    if state == -1:
      return k
    message = player.broadcast_message()
    if message:
      self.__messages[message[0]].update(message[1])
    return -1

  def __one_trial_fixbudget(self, stop_cond):
    budget = stop_cond
    self.__playable_players = list(range(self.__num_players))

    ############################################################################
    # initialization
    for k in range(self.__num_players):
      player = self._players[k]
      bandit = self._bandits[k]
      bandit.reset()
      player.reset(bandit, budget)
    ############################################################################

    round_ind = 0
    while True:
      round_ind += 1
      res = self.__one_round()
      if res > -1:
        # the sampled player stops
        self.__playable_players.remove(res)
      if len(self.__playable_players) == 0:
        regret = self.__regret()
        break
    return dict({self._players[0].name: [budget, round_ind, regret]})

  def __one_trial_fixconf(self, stop_cond):
    fail_prob = stop_cond
    self.__playable_players = list(range(self.__num_players))

    ############################################################################
    # initialization
    for k in range(self.__num_players):
      player = self._players[k]
      bandit = self._bandits[k]
      bandit.reset()
      player.reset(bandit, fail_prob)
    ############################################################################

    round_ind = 0
    while True:
      round_ind += 1
      res = self.__one_round()
      if res > -1:
        # the sampled player stops
        self.__playable_players.remove(res)
      if len(self.__playable_players) == 0:
        regret = self.__regret()
        break
    return dict({self._players[0].name: [fail_prob, round_ind, regret]})

  def _one_trial(self, seed, stop_cond):
    np.random.seed(seed)
    self.__messages = [EmArm() for ind in range(self._bandits[0].arm_num)]
    if 'Fix Budget' in self._players[0].goal:
      return self.__one_trial_fixbudget(stop_cond)
    return self.__one_trial_fixconf(stop_cond)

  def __regret(self):
    return int(sum([getattr(self._bandits[k], self._regret_funcname)(
        getattr(self._players[k], self._rewards_funcname)())
                    for k in range(self.__num_players)]) > 0)

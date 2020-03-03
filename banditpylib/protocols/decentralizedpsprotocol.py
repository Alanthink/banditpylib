from random import choice

from absl import flags
from absl import logging

import numpy as np

from ..bandits.arms import EmArm
from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedPEProtocol']


class DecentralizedPEProtocol(Protocol):
  """decentralized pure exploration protocol
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      logging.fatal('%s: number of players is not specified!' % self.type)
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedPEProtocol'

  def __one_round(self):
    # sample player
    k = choice(self.__playable_players)
    player = self._players[k]

    # play player and broadcast message
    state = player.learner_run(self.__messages)
    if state == -1:
      return k
    message = player.broadcast_message()
    if message:
      self.__messages[message[0]].update(message[1])
    return -1

  def __one_trial_fixbudget(self):
    results = []
    for budget in self._pars['budgets']:
      self.__playable_players = list(range(self.__num_players))

      ##########################################################################
      # initialization
      for k in range(self.__num_players):
        player = self._players[k]
        bandit = self._bandits[k]
        bandit.init()
        player.init(bandit, budget)
      ##########################################################################

      round_ind = 0
      while True:
        round_ind += 1
        res = self.__one_round()
        if res > -1:
          # the sampled player stops
          self.__playable_players.remove(res)
        if len(self.__playable_players) == 0:
          regret = self.__regret()
          results.append(
              dict({self._players[0].name: [budget, round_ind, regret]}))
          break
    return results

  def __one_trial_fixconf(self):
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__playable_players = list(range(self.__num_players))

      ##########################################################################
      # initialization
      for k in range(self.__num_players):
        player = self._players[k]
        bandit = self._bandits[k]
        bandit.init()
        player.init(bandit, fail_prob)
      ##########################################################################

      round_ind = 0
      while True:
        round_ind += 1
        res = self.__one_round()
        if res > -1:
          # the sampled player stops
          self.__playable_players.remove(res)
        if len(self.__playable_players) == 0:
          regret = self.__regret()
          results.append(
              dict({self._players[0].name: [fail_prob, round_ind, regret]}))
          break
    return results

  def _one_trial(self, seed):
    np.random.seed(seed)
    self.__messages = [EmArm() for ind in range(self._bandits[0].arm_num)]
    if 'FixBudget' in self._players[0].goal:
      return self.__one_trial_fixbudget()
    return self.__one_trial_fixconf()

  def __regret(self):
    return int(sum([getattr(
        self._bandits[k],
        '_'+type(self._bandits[k]).__name__+'__'+self._regret_def)(
            self._players[k].reward())
                    for k in range(self.__num_players)]) > 0)

from random import choice

from absl import flags
from absl import logging

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedBAIProtocol']


class DecentralizedBAIProtocol(Protocol):
  """decentralized best arm identification protocol
  """

  def __init__(self, pars):
    self.__messages = []
    if 'num_players' not in pars:
      logging.fatal('%s: number of players is not specified!' % self.type)
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedBAIProtocol'

  def __one_round(self):
    # sample player
    k = choice(self.__playable_players)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast message
    action = player.learner_choice(self.__messages)
    if action == -1:
      return k
    feedback = bandit.feed(action)
    player.update(action, feedback)
    message = player.broadcast_message(action, feedback)
    self.__messages.append({k: message})

    return -1

  def _one_trial(self, seed):
    np.random.seed(seed)

    if self._players[0].goal == 'FixedBudgetBAI':
      results = []
      for budget in self._pars['budgets']:
        self.__playable_players = list(range(self.__num_players))

        ########################################################################
        # initialization
        for k in range(self.__num_players):
          player = self._players[k]; bandit = self._bandits[k]
          bandit.init()
          player.init(bandit, budget)
        ########################################################################

        # round index
        r = 0
        while True:
          if r > budget:
            logging.fatal('%s uses more than the given budget!' %
                           self._players[0].name)
          # stop the recursion if there is no playable players available
          if len(self.__playable_players) == 0:
            regret = self.__regret()
            results.append(
                dict({self._players[0].name: [budget, r, regret]}))
            break
          res = self.__one_round()
          if res > -1:
            self.__playable_players.remove(res)
          r += 1
      return results

    # FixedConfidenceBAI
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__playable_players = list(range(self.__num_players))

      ##########################################################################
      # initialization
      for k in range(self.__num_players):
        player = self._players[k]; bandit = self._bandits[k]
        bandit.init()
        player.init(bandit, fail_prob)
      ##########################################################################

      # round index
      r = 0
      while True:
        if len(self.__playable_players) == 0:
          regret = self.__regret()
          results.append(
              dict({self._players[0].name: [fail_prob, r, regret]}))
          break

        res = self.__one_round()
        if res > -1:
          # no action is returned we should not add r by 1
          self.__playable_players.remove(res)
        else:
          r += 1

    return results

  def __regret(self):
    return sum([self._bandits[k].best_arm_regret(self._players[k].best_arm())
                for k in range(self.__num_players)])

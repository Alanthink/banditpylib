from random import choice


from absl import flags
from absl import logging

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedBAIProtocol']


class DecentralizedBAIProtocol(Protocol):
  """Decentralized Best Arm Identification Protocol
  """

  def __init__(self, pars):
    self.__messages = []
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedBAIProtocol'

  @property
  def _num_players(self):
    return self.__num_players

  @property
  def _budget(self):
    return self.__budget

  @property
  def _fail_prob(self):
    return self.__fail_prob

  def _one_round(self):
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

    message = player._broadcast_message(action, feedback)
    self.__messages.append({player.name: message})

    return -1

  def _one_trial(self, seed):
    np.random.seed(seed)
    if self._players[0].goal == 'FixedBudgetBAI':
      results = []
      for budget in self._pars['budgets']:
        self.__playable_players = list(range(self.__num_players))
        self.__budget = budget

        ############################################################################
        # initialization
        for k in range(self._num_players):
          player = self._players[k]
          bandit = self._bandits[k]
          bandit.init()
          player.init(bandit, self.__budget)
        ############################################################################
        while True:
          if bandit.tot_samples > budget:
            logging.fatal('%s uses more than the given budget!'
                % player.name)
          if len(self.__playable_players) == 0:
            break

          res = self._one_round()

          if res > -1:
            self.__playable_players.remove(res)

          regret = self.regret()
          results.append(
            dict({player.name:
                  [budget, bandit.tot_samples, regret]}))
      return results

    #  FixedConfidenceBAI
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__playable_players = list(range(self.__num_players))
      self.__fail_prob = fail_prob

      ############################################################################
      # initialization
      for k in range(self._num_players):
        player = self._players[k]
        bandit = self._bandits[k]
        bandit.init()
        player.init(bandit, self.__fail_prob)
      ############################################################################
      while True:
        if len(self.__playable_players) == 0:
          break

        res = self._one_round()
        if res > -1:
          self.__playable_players.remove(res)

        regret = self.regret()
        results.append(
            dict({player.name:
                  [fail_prob, bandit.tot_samples, regret]}))
    return results

  def regret(self):
    return sum([self._bandits[k].best_arm_regret(self._players[k].best_arm())
                for k in range(self._num_players)])

from absl import logging
import numpy as np
from random import randint

from bandits import ordinarybandit
from .utils import Protocol

__all__ = ['SinglePlayerProtocol']


class SinglePlayerProtocol(Protocol):
  """Decentralized protocol
  """

  def __init__(self, pars):
    self.__messages = []
    print(pars)
    if pars['num_players'] != 1:
      logging.fatal("SinglePlayerProtocol",
                     " only supports one player!")

    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'singleplayerprotocol: ' + self._player_type

  @property
  def _num_players(self):
    return self.__num_players

  @property
  def _messages(self):
    return self.__messages

  def _broadcast_message(self, message, player):
    self.__messages.append(message)

  def __init(self):
    # time starts from 1
    for k in range(self._num_players):
      player = self._players[k]
      bandit = self._bandits[k]
      player.init(bandit)

  def _one_trial(self, seed):
    np.random.seed(seed)
    ############################################################################
    # initialization
    self.__init()
    ############################################################################
    agg_regret = dict()
  
    for t in range(self._horizon + 1):
      if t > 0:
        self._play_round(t)
      if t > 0 and t % self._frequency == 0:
        agg_regret[t] = self.regret()
    return dict({self.type: agg_regret})

  def _play_round(self, t):
    # sample player
    k = randint(0, self._num_players-1)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast  message
    message = player._one_decentralized_iteration(self._messages)
    self._broadcast_message(message, player)

  def regret(self):
    return sum([self._players[k]._agg_decentralized_regret()
               for k in range(self._num_players)])

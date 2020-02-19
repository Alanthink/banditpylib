from random import randint

from absl import flags
import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['DecentralizedRegretMinProtocol']


class DecentralizedRegretMinProtocol(Protocol):
  """Decentralized Regret Minimization Protocol
  """

  def __init__(self, pars):
    self.__messages = []
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedRegretMinProtocol'

  @property
  def _num_players(self):
    return self.__num_players

  @property
  def __horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  def _one_round(self):
    # sample player
    k = randint(0, self._num_players-1)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast message
    context = bandit.context
    action = player.learner_choice(context, self.__messages)
    feedback = bandit.feed(action)
    player.update(context, action, feedback)
    message = player._broadcast_message(context, action, feedback)
    self.__messages.append({player.name: message})

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    for k in range(self._num_players):
      player = self._players[k]
      bandit = self._bandits[k]
      bandit.init()
      player.init(bandit, self.__horizon)
    ############################################################################

    agg_regret = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        self._one_round()
      if t > 0 and t % self.__frequency == 0:
        agg_regret[t] = self.regret()
    return dict({self._players[0].name: agg_regret})

  def regret(self):
    return sum([self._bandits[k].regret(self._players[k].rewards)
                for k in range(self._num_players)])

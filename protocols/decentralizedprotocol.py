from absl import logging
from random import choice

from bandits import ordinarybandit
from .utils import Protocol

__all__ = ['DecentralizedProtocol']

class DecentralizedProtocol(Protocol):
  """Decentralized protocol
  """

  def __init__(self, bandits, players):
    logging.info('Decentralized communication protocol')
    if not isinstance(players, list):
      logging.fatal('Players should be given in a list!')
	if not isinstance(bandits, list):
		logging.fatal('bandits shoould be given in a list!')
	for player in players:
      if not isinstance(arm, DELIM):
        logging.fatal('Not a DELIM arm!')
	if not ((len(bandits) == 1) or len(bandits == len(players)))
		logging.fatal('number of bandit instances should be one or equal \
					   to the number of players!')

    self.__bandits = bandits
    self.__players = players

  @property
  def type(self):
    return 'decentralizedprotocol'

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    self.__init()
    ############################################################################

    agg_regret = dict()
    for t in range(self._horizon + 1):
      if t > 0:
      	_play_round()
      #if t % self.__frequency == 0:
      #  agg_regret[t] = self._bandit.regret(self.__rewards)
    return dict({self.name: agg_regret})


  def _play_round(self):
  	# sample player
  	player = choice(self.__players)

  	# play player
  	player.
  	# broadcast messages
  	pass

  def init(self):
    pass

  def regret(self, rewards):
    return self.__max_rewards - rewards
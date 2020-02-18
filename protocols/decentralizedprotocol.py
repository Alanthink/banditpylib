import json
import time
from random import randint
from multiprocessing import Pool

from absl import flags
from absl import logging
import numpy as np

from bandits import Bandit
from .utils import Protocol, current_time

FLAGS = flags.FLAGS

__all__ = ['DecentralizedProtocol']


class DecentralizedProtocol(Protocol):
  """Decentralized protocol
  """

  def __init__(self, pars):
    self.__messages = []
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'decentralizedprotocol: ' + self._player_type

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

  @property
  def __trials(self):
    # # of repetitions of the play
    return self._pars['trials']

  @property
  def __processors(self):
    # maximum number of processors can be used
    return self._pars['processors']

  @property
  def _horizon(self):
    return self._pars['horizon']

  @property
  def _frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  @property
  def _bandits(self):
    return self.__bandits

  @property
  def _players(self):
    # frequency to record intermediate regret results
    return self.__players

  @property
  def _player_type(self):
    # frequency to record intermediate regret results
    return self.__player_type

  def __write_to_file(self, data):
    with open(self.__output_file, 'a') as f:
      if isinstance(data, list):
        for item in data:
          json.dump(item, f)
          f.write('\n')
      else:
        json.dump(data, f)
        f.write('\n')
      f.flush()

  def __multi_proc(self):
    pool = Pool(processes=self.__processors)

    for _ in range(self.__trials):
      result = pool.apply_async(self._one_trial, args=(current_time(), ),
          callback=self.__write_to_file)
      if FLAGS.debug:
        # for debugging purposes
        # to make sure error info of subprocesses will be reported
        # this flag could heavily increase the running time
        result.get()

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandits, players, output_file, pars):
    for b in bandits:
      if not isinstance(b, Bandit):
        logging.fatal('Not a legimate bandit!')

    self.__bandits = bandits
    self.__players = players
    self.__player_type = players[0].name
    self.__output_file = output_file
    self._pars = pars

    logging.info('run protocol %s' % (self.type))
    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time() - start_time))

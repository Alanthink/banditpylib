import json
import time

from abc import ABC, abstractmethod
from multiprocessing import Pool

from absl import flags
from absl import logging

from bandits import Bandit

__all__ = ['Protocol']
FLAGS = flags.FLAGS


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  @property
  @abstractmethod
  def _num_players(self):
    pass

  def __init(self):
    # time starts from 1
    pass

  @abstractmethod
  def _one_trial(self, seed):
    pass

  @abstractmethod
  def _play_round(self):
    pass

  @abstractmethod
  def regret(self, rewards):
    pass

  @property
  @abstractmethod
  def _messages(self):
    pass

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
    pool = Pool(processes = self.__processors)

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
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))
import json
import time
from multiprocessing import Pool

from abc import ABC, abstractmethod

from absl import flags
from absl import logging

from bandits import Bandit

FLAGS = flags.FLAGS

__all__ = ['Protocol', 'current_time']


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """Abstract protocol class"""

  @property
  @abstractmethod
  def type(self):
    pass

  @property
  def __trials(self):
    # # of repetitions of the play
    return self._pars['trials']

  @property
  def __processors(self):
    # maximum number of processors can be used
    return self._pars['processors']

  @abstractmethod
  def _one_trial(self, seed):
    pass

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

  def play(self, bandit, player, output_file, pars):
    if isinstance(bandit, list):
      for b in bandit:
        if not isinstance(b, Bandit):
          logging.fatal('There exists a non-bandit!')
      self._bandits = bandit
      self._players = player
      logging.info('run learner %s with goal %s under protocol %s' %
        (player[0].name, player[0].goal, self.type))
    else:
      if not isinstance(bandit, Bandit):
        logging.fatal('Not a legimate bandit!')
      self._bandit = bandit
      self._player = player
      logging.info('run learner %s with goal %s under protocol %s' %
        (player.name, player.goal, self.type))

    self.__output_file = output_file
    self._pars = pars

    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

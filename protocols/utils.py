import json
import time
from multiprocessing import Pool

from abc import ABC, abstractmethod

from absl import flags
from absl import logging

FLAGS = flags.FLAGS

__all__ = ['Protocol', 'current_time']


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """abstract protocol class"""

  @property
  @abstractmethod
  def type(self):
    pass

  @property
  def __trials(self):
    # # of trials of the simulation
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

  def play(self, bandit, learner, running_pars, output_file):
    if isinstance(bandit, list):
      # multiple players
      self._bandits = bandit
      self._players = learner
      logging.info(
        'run learner %s under protocol %s' % (learner[0].name, self.type))
    else:
      # single player
      self._bandit = bandit
      self._player = learner
      logging.info(
        'run learner %s with protocol %s' % (learner.name, self.type))

    self.__output_file = output_file
    self._pars = running_pars

    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

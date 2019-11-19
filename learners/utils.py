"""
Abstract learner
"""
import json
import time

from abc import ABC, abstractmethod
from multiprocessing import Pool

from absl import logging

from bandits import Bandit

__all__ = ['Learner']


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Learner(ABC):
  """Abstract class for learners"""

  # learner goal
  @property
  @abstractmethod
  def goal(self):
    pass

  # learner name
  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _goal_init(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def _goal_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  # action suggested by the learner
  @abstractmethod
  def _choice(self, context):
    pass

  @abstractmethod
  def _one_trial(self, seed):
    pass

  def __init__(self):
    pass

  def _init(self, bandit):
    # time starts from 1
    bandit.init()
    self._bandit = bandit
    self._t = 1
    self._goal_init()
    self._model_init()
    self._learner_init()

  def _update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1

  def __write_to_file(self, data):
    with open(self._pars['output'], 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def __multi_proc(self):
    pool = Pool(processes = self._pars['processors'])

    for _ in range(self._pars['trials']):
      result = pool.apply_async(self._one_trial, args=(current_time(), ),
          callback=self.__write_to_file)
      del result
      # for debug purposes
      # result.get()

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandit, pars):
    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')

    self._bandit = bandit
    self._pars = pars

    logging.info('run learner %s with goal %s' % (self.name, self.goal))
    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

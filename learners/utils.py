"""
Abstract learner
"""
import json
import time

from abc import ABC, abstractmethod
from multiprocessing import Pool

from absl import logging

from bandits import Bandit

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
  def choice(self, context):
    pass

  @abstractmethod
  def one_trial(self, seed):
    pass

  def __init__(self):
    pass

  def init(self, bandit):
    # time starts from 1
    bandit.init()
    self._bandit = bandit
    self._t = 1
    self._goal_init()
    self._model_init()
    self._learner_init()

  def update(self, context, action, feedback):
    self._goal_update(context, action, feedback)
    self._model_update(context, action, feedback)
    self._learner_update(context, action, feedback)
    self._t += 1

  def write_to_file(self, data):
    with open(self._pars['output'], 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def multi_proc(self):
    pool = Pool(processes = self._pars['processors'])

    for _ in range(self._pars['trials']):
      pool.apply_async(self.one_trial, args=(current_time(), ),
          callback=self.write_to_file)

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandit, pars):
    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')

    self._bandit = bandit
    self._pars = pars

    logging.info('run learner %s' % self.name)
    start_time = time.time()
    self.multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

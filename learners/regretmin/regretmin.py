import json
import time

from abc import abstractmethod
from multiprocessing import Pool

import numpy as np
from absl import logging

from bandits.bandit import Bandit
from learners.learner import Learner
from utils import current_time


class RegretMinimizationLearner(Learner):
  """Base class for regret minimization learners"""

  @property
  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def _model_init(self):
    pass

  @abstractmethod
  def _learner_init(self):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def _model_update(self, context, action, feedback):
    pass

  @abstractmethod
  def _learner_update(self, context, action, feedback):
    pass

  @property
  def goal(self):
    return self.__goal

  @property
  def rewards(self):
    return self.__rewards

  @property
  def horizon(self):
    return self.__horizon

  def __init__(self):
    super().__init__()
    self.__goal = 'Regret minimization'

  def _goal_init(self):
    self.__rewards = 0

  def _goal_update(self, context, action, feedback):
    self.__rewards += feedback[0]

  # methods for simulation

  def one_trial(self, bandit, horizon, breakpoints, seed):
    np.random.seed(seed)
    self.__horizon = horizon

    ############################################################################
    # learner initialization
    self.init(bandit)
    ############################################################################

    agg_regret = dict()
    for t in range(horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self.choice(context)
        feedback = self._bandit.feed(action)
        self.update(context, action, feedback)
      if t in breakpoints:
        agg_regret[t] = self._bandit.regret(self.rewards)
    return dict({self.name: agg_regret})

  def write_to_file(self, data):
    with open(self.__output_file, 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def multi_proc(self, bandit, horizon, breakpoints, trials, processors):
    pool = Pool(processes = processors)

    for _ in range(trials):
      pool.apply_async(self.one_trial, args=(
          bandit, horizon, breakpoints, current_time(), ),
          callback=self.write_to_file)

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandit, output_file, horizon=20,
           mod=10, trials=2, processors=2):
    """Simulation method

    Input:
      mod: record regret after every `mod` actions
      trials: total number of independent runs
      processors: maximum number of processors allowed to be used
    """

    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')

    self.__output_file = output_file

    # 0 is included
    breakpoints = []
    for i in range(horizon + 1):
      if i % mod == 0:
        breakpoints.append(i)

    logging.info('run learner %s' % self.name)
    start_time = time.time()
    self.multi_proc(bandit, horizon, breakpoints, trials, processors)
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

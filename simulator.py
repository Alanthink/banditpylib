"""
Simulators for doing experiments
"""

import json
import time

from multiprocessing import Pool

import numpy as np
from absl import logging

from bandits.bandit import Bandit
from learners.learner import Learner
from utils import current_time


class Simulator:
  """Base class for simulator"""

  def __init__(self, bandit, learners, seed=0):
    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')
    if not isinstance(learners, list):
      logging.fatal('Learners should be given in a list!')
    if not learners:
      logging.fatal('There should be at least one learner!')

    goal = learners[0].goal
    for learner in learners:
      if learner.goal != goal:
        logging.fatal('Some learner has different goal!')
      if not isinstance(learner, Learner):
        logging.fatal('Some learner is not legimate!')

    logging.info(goal)

    del seed
    # Set random seed only when there is one process
    # if not isinstance(seed, int):
    #   logging.fatal('Random seed should be an integer!')
    #   np.random.seed(seed)

    self._bandit = bandit
    self._learners = learners


class RegretMinimizationSimulator(Simulator):
  """Simulator for regret minimization"""

  def __init__(self, bandit, learners):
    super().__init__(bandit, learners)

  def one_trial(self, learner, horizon, breakpoints, seed):
    ############################################################################
    # bandit initialization
    self._bandit.init()
    # learner initialization
    learner.init(self._bandit, horizon)
    ############################################################################
    np.random.seed(seed)

    agg_regret = dict()
    for t in range(horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = learner.choice(context)
        feedback = self._bandit.feed(action)
        learner.update(context, action, feedback)
      if t in breakpoints:
        agg_regret[t] = self._bandit.regret(learner.rewards)
    return dict({learner.name: agg_regret})

  def write_to_file(self, data):
    with open(self.__output_file, 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def multi_proc(self, learner, horizon, breakpoints, trials, processors):
    pool = Pool(processes = processors)

    for _ in range(trials):
      pool.apply_async(self.one_trial, args=(
          learner, horizon, breakpoints, current_time(), ),
          callback=self.write_to_file)

    # can not apply for processes any more
    pool.close()
    pool.join()

  def sim(self, output_file, horizon=20, mod=10, trials=2, processors=2):
    """Simulation method

    Input:
      mod: record regret after every `mod` actions
      trials: total number of independent runs
      processors: maximum number of processors allowed to be used
    """

    # clean file
    open(output_file, 'w').close()
    self.__output_file = output_file

    # 0 is included
    breakpoints = []
    for i in range(horizon + 1):
      if i % mod == 0:
        breakpoints.append(i)

    for learner in self._learners:
      logging.info('run learner %s' % learner.name)
      start_time = time.time()
      self.multi_proc(learner, horizon, breakpoints, trials, processors)
      logging.info('%.2f seconds elapsed' % (time.time()-start_time))

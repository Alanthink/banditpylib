"""
Simulators for doing experiments
"""

import json
import time

from multiprocessing import Process

import numpy as np
from absl import logging

from bandits.bandit import BanditEnvironment
from learners.learner import Learner
from utils import current_time


class Simulator:
  """Base class for simulator"""

  def __init__(self, bandit, learners, seed=0):
    if not isinstance(bandit, BanditEnvironment):
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

  def one_run(self, learner, horizon, breakpoints, output_file, seed):
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
        context = self._bandit.context()
        action = learner.choice(context)
        reward = self._bandit.pull(context, action)
        learner.update(context, action, reward)
      if t in breakpoints:
        agg_regret[t] = self._bandit.regret(learner.rewards)
    json.dump(dict({learner.name: agg_regret}), output_file)
    output_file.write('\n')
    output_file.flush()

  def multi_proc(self, learner, horizon, breakpoints, output_file, processors):
    procs = [Process(target=self.one_run, args=(learner, horizon, breakpoints, output_file, current_time())) for _ in range(processors)]
    for proc in procs:
      proc.start()
    for proc in procs:
      proc.join()

  def multi_proc_helper(self, learner, horizon, breakpoints, output_file, trials, processors):
    for _ in range(trials//processors):
      self.multi_proc(learner, horizon, breakpoints, output_file, processors)

  def sim(self, output_path, horizon=100, mod=10, trials=1, processors=1):
    """Simulation method

    Input:
      mod: record regret after every `mod` actions
      trials: total number of independent runs
      processors: maximum number of processors allowed to be used
    """

    # 0 is included
    breakpoints = []
    for i in range(horizon + 1):
      if i % mod == 0:
        breakpoints.append(i)

    with open(output_path, 'w') as output_file:
      for learner in self._learners:
        logging.info('run learner %s' % learner.name)
        start_time = time.time()
        self.multi_proc_helper(learner, horizon, breakpoints, output_file, trials, processors)
        logging.info('%.2f seconds elapsed' % (time.time()-start_time))

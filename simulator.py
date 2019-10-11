"""
Simulators for doing experiments
"""

import json
import time

from multiprocessing import Process

import numpy as np
from absl import logging

from bandit import BanditEnvironment
from learner import Learner
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

    goal = learners[0].get_goal()
    for learner in learners:
      if learner.get_goal() != goal:
        logging.fatal('Some learner has different goal!')
      if not isinstance(learner, Learner):
        logging.fatal('Some learner is not legimate!')

    logging.info(goal)

    del seed
    # Set random seed only when there is one process
    # if not isinstance(seed, int):
    #   logging.fatal('Random seed should be an integer!')
    #   np.random.seed(seed)

    self.bandit = bandit
    self.learners = learners


class RegretMinimizationSimulator(Simulator):
  """Simulator for regret minimization"""

  def __init__(self, bandit, learners):
    super().__init__(bandit, learners)

  def one_run(self, learner, horizon, breakpoints, output_file, seed):
    ############################################################################
    # learner initialization
    learner.init(self.bandit, horizon)
    ############################################################################
    np.random.seed(seed)

    total_regret = dict()
    for step in range(horizon + 1):
      if step > 0:
        choice = learner.choice(step)
        reward = self.bandit.pull(choice)
        learner.update(choice, reward)
      if step in breakpoints:
        total_regret[step] = total_regret.get(step, 0) + \
          self.bandit.regret(step, learner.get_rewards())
    json.dump(dict({learner.get_name(): total_regret}), output_file)
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

  def sim(self, horizon, output_path, interval=20, trials=200, processors=40):
    """Simulation method

    Input:
      interval: record regret every `interval` actions
      trials: total number of independent runs
      processors: maximum number of processors allowed to be used
    """

    # 0 is included
    breakpoints = []
    for i in range(horizon + 1):
      if i % interval == 0:
        breakpoints.append(i)

    with open(output_path, 'w') as output_file:
      for learner in self.learners:
        logging.info('run learner %s' % learner.get_name())
        start_time = time.time()
        self.multi_proc_helper(learner, horizon, breakpoints, output_file, trials, processors)
        logging.info('%.2f seconds elapsed' % (time.time()-start_time))

"""
Simulators for doing experiments
"""

import json
import time

from multiprocessing import Process

import numpy as np
from absl import logging

from bandit import Bandit
from learner import Learner


# for generating random seeds
def current_time():
  import time
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


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

    if not isinstance(seed, int):
      logging.fatal('Random seed should be an integer!')
      np.random.seed(seed)

    # initialize all learners
    for learner in learners:
      learner.init(bandit.get_num_of_arm())

    self.bandit = bandit
    self.learners = learners


class RegretMinimizationSimulator(Simulator):
  """Simulator for regret minimization"""

  def __init_(self, bandit, learners):
    Simulator.__init__(self, bandit, learners)

  def one_run(self, learner, horizon, breakpoints, output_file, seed):
    np.random.seed(seed)
    total_regret = dict()
    learner.reset()
    for time in range(horizon + 1):
      if time > 0:
        choice = learner.choice(time)
        reward = self.bandit.pull(choice)
        learner.update(choice, reward)
      if time in breakpoints:
        total_regret[time] = total_regret.get(time, 0) + \
          self.bandit.regret(time, learner.rewards)
    json.dump(dict({learner.name: total_regret}), output_file)
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
      interval: record regret every `interval` samples
      trials: total number of independent runs
      processors: maximum number of processors allowed to be used
    """

    for learner in self.learners:
      learner.pass_horizon(horizon)

    # 0 is included
    breakpoints = []
    for i in range(horizon + 1):
      if i % interval == 0:
        breakpoints.append(i)

    with open(output_path, 'w') as output_file:
      for learner in self.learners:
        logging.info('Run learner %s' % learner.name)
        start_time = time.time()
        self.multi_proc_helper(learner, horizon, breakpoints, output_file, trials, processors)
        logging.info('%.2f seconds used' % (time.time()-start_time))

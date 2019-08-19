"""
Simulators for doing experiments
"""

from absl import logging

from bandit import Bandit
from learner import Learner


class Simulator:
  """Base class for simulator"""

  def __init__(self, bandit, learners):
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

    # initialize all learners
    for learner in learners:
      learner.init(bandit.get_num_of_arm())

    self.bandit = bandit
    self.learners = learners


class RegretMinimizationSimulator(Simulator):
  """Simulator for regret minimization """

  def __init_(self, bandit, learners):
    Simulator.__init__(self, bandit, learners)

  def sim(self, horizon, interval, trials):
    """Simulation method"""

    for learner in self.learners:
      learner.pass_horizon(horizon)

    breakpoints = []
    for i in range(horizon + 1):
      if i % interval == 0:
        breakpoints.append(i)

    results = dict()
    results['breakpoints'] = breakpoints

    for learner in self.learners:
      logging.info('Simulate learner %s' % learner.name)
      total_regret = dict()
      for _ in range(trials):
        learner.reset()
        for time in range(horizon + 1):
          choice = learner.choice(time)
          reward = self.bandit.pull(choice)
          learner.update(choice, reward)
          if time in breakpoints:
            total_regret[time] = total_regret.get(time, 0) + \
              self.bandit.regret(time, learner.rewards)
      regret = []
      for breakpoint in breakpoints:
        regret.append(total_regret[breakpoint] / trials)
      results[learner.name] = regret

    return results

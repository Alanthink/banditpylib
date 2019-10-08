"""
Different strategies
"""

import numpy as np

from absl import logging


class EmArm:
  """Data Structure for storing empirical information of each arm"""

  def __init__(self):
    self.pulls = 0
    self.rewards = 0

  def get_em_mean(self):
    """get empirical mean"""
    if self.pulls == 0:
      logging.fatal('No empirical mean yet!')
    return self.rewards / self.pulls

  def reset(self):
    """clear historical records"""
    self.pulls = 0
    self.rewards = 0


class Learner:
  """Base class for learners"""

  def __init__(self):
    pass

  def init(self, K):
    """initialize K arms"""
    self.arm_num = K
    self.em_arms = [EmArm() for ind in range(self.arm_num)]
    self.rewards = 0

  def update(self, ind, reward):
    """update historical record for a specific arm"""
    self.em_arms[ind].pulls += 1
    self.em_arms[ind].rewards += reward
    self.rewards += reward

  def pass_horizon(self, horizon):
    """pass the horizon to the learner"""
    self.horizon = horizon

  def reset(self):
    """clear historical records for all arms"""
    for arm in self.em_arms:
      arm.reset()
    self.rewards = 0


class RegretMinimizationLearner(Learner):
  """Base class for regret minimization learners"""

  def __init__(self):
    Learner.__init__(self)
    self.goal = 'Regret minimization'


class Uniform(RegretMinimizationLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  def __init__(self):
    RegretMinimizationLearner.__init__(self)
    self.name = 'Uniform'

  def choice(self, time):
    """return an arm to pull"""
    return (time-1) % self.arm_num


class UCB(RegretMinimizationLearner):
  """UCB"""

  def __init__(self, alpha=2):
    RegretMinimizationLearner.__init__(self)
    self.alpha = alpha
    self.name = 'UCB'

  def upper_confidence_bound(self, arm, time):
    """upper confidence bound"""
    return arm.get_em_mean() + \
        np.sqrt(self.alpha / arm.pulls * np.log(time))

  def choice(self, time):
    """return an arm to pull"""
    if time <= self.arm_num:
      return (time-1) % self.arm_num

    upper_bound = np.zeros(self.arm_num)
    for ind in range(self.arm_num):
      upper_bound[ind] = self.upper_confidence_bound(self.em_arms[ind], time)
    return np.argmax(upper_bound)


class MOSS(RegretMinimizationLearner):
  """MOSS"""

  def __init__(self):
    RegretMinimizationLearner.__init__(self)
    self.name = 'MOSS'

  def upper_confidence_bound(self, arm, time):
    """upper confidence bound"""
    del time
    return arm.get_em_mean() + np.sqrt(
        max(0, np.log(self.horizon / (self.arm_num * arm.pulls))) / arm.pulls)

  def choice(self, time):
    """return an arm to pull"""
    if time <= self.arm_num:
      return (time-1) % self.arm_num

    upper_bound = np.zeros(self.arm_num)
    for ind in range(self.arm_num):
      upper_bound[ind] = self.upper_confidence_bound(self.em_arms[ind], time)
    return np.argmax(upper_bound)


class TS(RegretMinimizationLearner):
  """Thompson Sampling"""

  def __init__(self):
    RegretMinimizationLearner.__init__(self)
    self.name = 'Thompson Sampling'

  def choice(self, time):
    """return an arm to pull"""
    if time <= self.arm_num:
      return (time-1) % self.arm_num

    # each arm has a uniform prior B(1, 1)
    vir_means = np.zeros(self.arm_num)
    for arm in range(self.arm_num):
      a = 1 + self.em_arms[arm].rewards
      b = 1 + self.em_arms[arm].pulls - self.em_arms[arm].rewards
      vir_means[arm] = np.random.beta(a, b)

    return np.argmax(vir_means)

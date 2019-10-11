"""
Learners under the classic bandit model.
"""

from abc import abstractmethod

import numpy as np

from absl import logging

from learner import Learner


class EmArm:
  """Class for storing empirical information of an arm"""

  def __init__(self):
    self.reset()

  @property
  def pulls(self):
    return self._pulls

  @property
  def rewards(self):
    return self._rewards

  @property
  def em_mean(self):
    """get empirical mean"""
    if self._pulls == 0:
      logging.fatal('No empirical mean yet!')
    return self._rewards / self._pulls

  def reset(self):
    """clear historical records"""
    self._pulls = 0
    self._rewards = 0

  def update(self, pulls, rewards):
    self._pulls += pulls
    self._rewards += rewards


class BanditLearner(Learner):
  """Base class for learners in the classic bandit model"""

  def model_init(self):
    """local initialization"""
    if self._bandit.type != 'classicbandit':
      logging.fatal('(classiclearner) I don\'t understand the bandit environment!')

  @abstractmethod
  def goal_init(self):
    pass

  @abstractmethod
  def learner_init(self):
    pass

  @abstractmethod
  def update(self, action, feedback):
    pass

  @abstractmethod
  def choice(self, time):
    pass

  @abstractmethod
  def goal(self):
    pass

  @abstractmethod
  def name(self):
    pass

  @abstractmethod
  def local_update(self, action, feedback):
    pass


class RegretMinimizationLearner(BanditLearner):
  """Base class for regret minimization learners"""

  def __init__(self):
    self._goal = 'Regret minimization'

  @property
  def goal(self):
    return self._goal

  @property
  def rewards(self):
    return self.__rewards

  def goal_init(self):
    self._arm_num = self._bandit.arm_num
    self.__rewards = 0

  def learner_init(self):
    """clear historical records for all arms"""
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  def update(self, action, feedback):
    """update historical record for a specific arm"""
    self._em_arms[action].update(1, feedback)
    self.__rewards += feedback
    self.local_update(action, feedback)

  @abstractmethod
  def choice(self, time):
    pass

  @abstractmethod
  def name(self):
    pass

  def local_update(self, action, feedback):
    pass


class Uniform(RegretMinimizationLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  def __init__(self):
    super().__init__()
    self._name = 'Uniform'

  @property
  def name(self):
    return self._name

  def choice(self, time):
    """return an arm to pull"""
    return (time-1) % self._arm_num


class UCB(RegretMinimizationLearner):
  """UCB"""

  def __init__(self, alpha=2):
    super().__init__()
    self._alpha = alpha
    self._name = 'UCB'

  @property
  def name(self):
    return self._name

  def upper_confidence_bound(self, arm, time):
    """upper confidence bound"""
    return arm.em_mean + \
        np.sqrt(self._alpha / arm.pulls * np.log(time))

  def choice(self, time):
    """return an arm to pull"""
    if time <= self._arm_num:
      return (time-1) % self._arm_num

    upper_bound = np.zeros(self._arm_num)
    for ind in range(self._arm_num):
      upper_bound[ind] = self.upper_confidence_bound(self._em_arms[ind], time)
    return np.argmax(upper_bound)


class MOSS(RegretMinimizationLearner):
  """MOSS"""

  def __init__(self):
    super().__init__()
    self._name = 'MOSS'
    logging.info('(MOSS) I will use horizon.')

  @property
  def name(self):
    return self._name

  def upper_confidence_bound(self, arm, time):
    """upper confidence bound"""
    del time
    return arm.em_mean + np.sqrt(
        max(0, np.log(self._horizon / (self._arm_num * arm.pulls))) / arm.pulls)

  def choice(self, time):
    """return an arm to pull"""
    if time <= self._arm_num:
      return (time-1) % self._arm_num

    upper_bound = np.zeros(self._arm_num)
    for ind in range(self._arm_num):
      upper_bound[ind] = self.upper_confidence_bound(self._em_arms[ind], time)
    return np.argmax(upper_bound)


class TS(RegretMinimizationLearner):
  """Thompson Sampling"""

  def __init__(self):
    super().__init__()
    self._name = 'Thompson Sampling'

  @property
  def name(self):
    return self._name

  def choice(self, time):
    """return an arm to pull"""
    if time <= self._arm_num:
      return (time-1) % self._arm_num

    # each arm has a uniform prior B(1, 1)
    vir_means = np.zeros(self._arm_num)
    for arm in range(self._arm_num):
      a = 1 + self._em_arms[arm].rewards
      b = 1 + self._em_arms[arm].pulls - self._em_arms[arm].rewards
      vir_means[arm] = np.random.beta(a, b)

    return np.argmax(vir_means)

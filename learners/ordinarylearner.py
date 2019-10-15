"""
Learners under the classic bandit model.
"""

from abc import abstractmethod

import numpy as np

from absl import logging

from learners.learner import Learner


class EmArm:
  """Class for storing empirical information of an arm"""

  def __init__(self):
    self.reset()

  @property
  def pulls(self):
    return self.__pulls

  @property
  def rewards(self):
    return self.__rewards

  @property
  def em_mean(self):
    """get empirical mean"""
    if self.__pulls == 0:
      logging.fatal('No empirical mean yet!')
    return self.__rewards / self.__pulls

  @property
  def em_var(self):
    """get empirical variance"""
    if self.__pulls == 0:
      logging.fatal('No empirical std yet!')
    return (self.__sq_rewards-self.__rewards**2/self.__pulls)/self.__pulls

  def reset(self):
    """clear historical records"""
    self.__pulls = 0
    self.__rewards = 0
    self.__sq_rewards = 0

  def update(self, pulls, rewards):
    self.__pulls += pulls
    self.__rewards += rewards
    self.__sq_rewards += rewards**2


class BanditLearner(Learner):
  """Base class for learners in the classic bandit model"""

  def model_init(self):
    """local initialization"""
    if self._bandit.type != 'classicbandit':
      logging.fatal('(classiclearner) I don\'t understand the bandit environment!')
    self._arm_num = self._bandit.arm_num
    # record empirical information for every arm
    self._em_arms = [EmArm() for ind in range(self._arm_num)]

  @abstractmethod
  def goal_init(self):
    pass

  @abstractmethod
  def learner_init(self):
    pass

  def model_update(self, context, action, feedback):
    self._em_arms[action].update(1, feedback)

  @abstractmethod
  def goal_update(self, context, action, feedback):
    pass

  @abstractmethod
  def learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def goal(self):
    pass

  @abstractmethod
  def name(self):
    pass


class RegretMinimizationLearner(BanditLearner):
  """Base class for regret minimization learners"""

  def __init__(self):
    self.__goal = 'Regret minimization'

  @property
  def goal(self):
    return self.__goal

  @property
  def rewards(self):
    return self.__rewards

  def goal_init(self):
    self.__rewards = 0

  @abstractmethod
  def learner_init(self):
    pass

  def goal_update(self, context, action, feedback):
    self.__rewards += feedback

  @abstractmethod
  def learner_update(self, context, action, feedback):
    pass

  @abstractmethod
  def choice(self, context):
    pass

  @abstractmethod
  def name(self):
    pass


class Uniform(RegretMinimizationLearner):
  """Naive uniform algorithm: sample each arm the same number of times"""

  def __init__(self):
    super().__init__()
    self.__name = 'Uniform'

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    return (self._t) % self._arm_num

  def learner_update(self, context, action, feedback):
    pass


class EpsGreedy(RegretMinimizationLearner):
  """Epsilon-Greedy Algorithm

  With probability eps/t do uniform sampling and with the left probability,
  pull arm with the maximum empirical mean.
  """

  def __init__(self, eps=1):
    super().__init__()
    self.__eps = eps
    self.__name = 'EpsilonGreedy'

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    rand = np.random.random_sample()
    if rand <= self.__eps/self._t:
      return np.random.randint(self._arm_num)
    return np.argmax(np.array([arm.em_mean for arm in self._em_arms]))

  def learner_update(self, context, action, feedback):
    pass


class UCB(RegretMinimizationLearner):
  """UCB"""

  def __init__(self, alpha=2):
    super().__init__()
    self.__alpha = alpha
    self.__name = 'UCB'

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t))
        for arm in self._em_arms]

    return np.argmax(ucb)

  def learner_update(self, context, action, feedback):
    pass


class MOSS(RegretMinimizationLearner):
  """MOSS"""

  def __init__(self):
    super().__init__()
    self.__name = 'MOSS'
    logging.info('(MOSS) I will use horizon.')

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+
           np.sqrt(max(0, np.log(self._horizon/(self._arm_num*arm.pulls)))/arm.pulls)
           for arm in self._em_arms]

    return np.argmax(ucb)

  def learner_update(self, context, action, feedback):
    pass


class TS(RegretMinimizationLearner):
  """Thompson Sampling"""

  def __init__(self):
    super().__init__()
    self.__name = 'Thompson Sampling'

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    # each arm has a uniform prior B(1, 1)
    vir_means = np.zeros(self._arm_num)
    for arm in range(self._arm_num):
      a = 1 + self._em_arms[arm].rewards
      b = 1 + self._em_arms[arm].pulls - self._em_arms[arm].rewards
      vir_means[arm] = np.random.beta(a, b)

    return np.argmax(vir_means)

  def learner_update(self, context, action, feedback):
    pass


class UCBV(RegretMinimizationLearner):
  """UCB-V algorithm"""

  def __init__(self, eta=1.2):
    """eta should be greater than 1"""
    super().__init__()
    self.__name = 'UCBV'
    self.__eta = eta

  @property
  def name(self):
    return self.__name

  def learner_init(self):
    pass

  def choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    ucb = [arm.em_mean+
           np.sqrt(2*self.__eta*arm.em_var/arm.pulls*np.log(self._t))+
           3*self.__eta*np.log(self._t)/arm.pulls for arm in self._em_arms]

    return np.argmax(ucb)

  def learner_update(self, context, action, feedback):
    pass

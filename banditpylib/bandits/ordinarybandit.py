from importlib import import_module

from abc import abstractmethod
from absl import logging

from .utils import Bandit

ARM_PKG = 'banditpylib.bandits.arms'


class OrdinaryBanditItf(Bandit):
  """Ordinary bandit interface"""

  @property
  @abstractmethod
  def type(self):
    pass

  @abstractmethod
  def reset(self):
    pass

  @property
  @abstractmethod
  def context(self):
    pass

  @property
  @abstractmethod
  def arm_num(self):
    pass

  @property
  @abstractmethod
  def arm_type(self):
    pass

  @property
  @abstractmethod
  def tot_samples(self):
    pass

  @abstractmethod
  def _take_action(self, action):
    pass

  @abstractmethod
  def _update_context(self):
    pass


class OrdinaryBandit(OrdinaryBanditItf):
  """Ordinary bandit

  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, pars):
    if pars['arm'] not in ['BernoulliArm', 'GaussianArm']:
      raise Exception('Can not setup %s!' % pars['arm'])
    self.__arm_type = pars['arm']
    means = pars['means']
    if not isinstance(means, list):
      raise Exception('Means should be given in a list!')
    if self.__arm_type == 'GaussianArm':
      if 'var' not in pars['means']:
        logging.warn('Variance upper bound is set 1!')
        self.__var = 1
      else:
        self.__var = pars['var']
    Arm = getattr(import_module(ARM_PKG), pars['arm'])
    arms = [Arm(mean) for mean in means]
    self.__arms = arms
    self.__arm_num = len(arms)
    if self.__arm_num < 2:
      raise Exception('The number of arms should be at least two!')

    self.__best_arm_ind = max(
        [(tup[0], tup[1].mean) for tup in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_ind]

  @property
  def arm_num(self):
    """return number of arms"""
    return self.__arm_num

  @property
  def arm_type(self):
    return self.__arm_type

  @property
  def type(self):
    return 'ordinarybandit'

  @property
  def tot_samples(self):
    return self.__tot_samples

  def reset(self):
    self.__tot_samples = 0
    self.__max_rewards = 0

  @property
  def context(self):
    return None

  def _take_action(self, action):
    is_list = True
    if not isinstance(action, list):
      is_list = False
      action = [(action, 1)]

    rewards = []
    for tup in action:
      ind = tup[0]
      if ind not in range(self.arm_num):
        raise Exception('Wrong arm index!')

      rewards.append(self.__arms[ind].pull(tup[1]))
      self.__tot_samples += tup[1]
      self.__max_rewards += (self.__best_arm.mean * tup[1])

    if not is_list:
      # rewards[0] is a numpy array with size 1
      return (rewards[0][0],)
    return (rewards,)

  def _update_context(self):
    pass

  def regret(self, rewards):
    return self.__max_rewards - rewards

  def best_arm_regret(self, ind):
    return 1 - (self.__best_arm_ind == ind)

from importlib import import_module

from absl import logging

from .utils import OrdinaryBanditItf

__all__ = ['OrdinaryBanditItf', 'OrdinaryBandit']

ARM_PKG = 'banditpylib.bandits.arms'


class OrdinaryBandit(OrdinaryBanditItf):
  """ordinary bandit
  Arms are numbered from 0 to len(arms)-1 by default.
  """

  def __init__(self, pars):
    if pars['arm'] not in ['BernoulliArm', 'GaussianArm']:
      logging.fatal('Can not setup %s!' % pars['arm'])
    self.__arm_type = pars['arm']
    means = pars['means']
    if not isinstance(means, list):
      logging.fatal('Means should be given in a list!')
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
      logging.fatal('The number of arms should be at least two!')

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

  def init(self):
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
        logging.fatal('Wrong arm index!')

      rewards.append(self.__arms[ind].pull(tup[1]))
      self.__tot_samples += tup[1]
      self.__max_rewards += (self.__best_arm.mean * tup[1])

    if not is_list:
      # rewards[0] is a numpy array with size 1
      return (rewards[0][0],)
    return (rewards,)

  def _update_context(self):
    pass

  def __regret(self, rewards):
    return self.__max_rewards - rewards

  def __best_arm_regret(self, ind):
    return 1 - (self.__best_arm_ind == ind)

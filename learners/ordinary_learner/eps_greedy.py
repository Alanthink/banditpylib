import random

import numpy as np

from arms import PseudoArm
from .utils import OrdinaryLearner


class EpsGreedy(OrdinaryLearner):
  r"""Epsilon-Greedy policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability play the arm with the maximum empirical mean.
  """

  def __init__(self, arm_num: int, horizon: int, eps=1.0, name=None):
    if arm_num <= 2:
      raise Exception('Number of arms %d is less then 3!' % arm_num)
    self.__arm_num = arm_num
    if horizon < arm_num:
      raise Exception('Horizon %d is less than number of arms %d!' % \
          (horizon, arm_num))
    self.__horizon = horizon
    if eps <= 0:
      raise Exception('Epsilon %.2f in %s is no greater than 0!' % \
          (eps, name))
    self.__eps = eps
    self.__name = name if name else 'epsilon_greedy'

  @property
  def name(self):
    return self.__name

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.__arm_num)]
    self.__total_rewards = 0.0
    self.__time = 1

  def actions(self, context=None):
    if self.__time > self.__horizon:
      self.__last_actions = None
      return self.__last_actions

    if self.__time <= self.__arm_num:
      self.__last_actions = [((self.__time-1) % self.__arm_num, 1)]
      return self.__last_actions

    # with probability eps/t, randomly select an arm to pull
    if random.random() <= self.__eps/self.__time:
      self.__last_actions = [(random.randint(0, self.__arm_num-1), 1)]
      return self.__last_actions

    self.__last_actions = \
        [(np.argmax(
            np.array([arm.em_mean() for arm in self.__pseudo_arms])), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__total_rewards += sum(feedback[0][0])
    self.__time += 1

  def _total_rewards(self):
    return self.__total_rewards

from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class EpsGreedy(OrdinaryLearner):
  r"""Epsilon-Greedy policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability play the arm with the maximum empirical mean.
  """
  def __init__(self, arm_num: int, horizon: int, eps=1.0, name=None):
    self.__name = name if name else 'epsilon_greedy'
    super().__init__(arm_num, horizon)
    if eps <= 0:
      raise Exception('Epsilon %.2f in %s is no greater than 0!' % \
          (eps, self.__name))
    self.__eps = eps

  @property
  def name(self):
    return self.__name

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Return:
      arms to pull
    """
    del context
    # pylint: disable=E1101
    if self.__time > self.horizon():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    # with probability eps/t, randomly select an arm to pull
    elif np.random.random() <= self.__eps / self.__time:
      self.__last_actions = [(np.random.randint(0, self.arm_num()), 1)]
    else:
      self.__last_actions = [
          (np.argmax(np.array([arm.em_mean()
                               for arm in self.__pseudo_arms])), 1)
      ]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

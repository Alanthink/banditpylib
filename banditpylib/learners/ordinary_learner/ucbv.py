from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class UCBV(OrdinaryLearner):
  r"""UCBV policy :cite:`audibert2009exploration`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{ \frac{ 2
    \hat{V}_i(t) \ln(t) }{T_i(t)} }+ \frac{ b \ln(t) }{T_i(t)} \right\}

  .. note::
    Reward has to be bounded within :math:`[0, b]`.
  """
  def __init__(self, arm_num: int, horizon: int, b=1.0, name=None):
    """
    Args:
      b: upper bound of reward
    """
    self.__name = name if name else 'ucbv'
    super().__init__(arm_num, horizon)
    if b <= 0:
      raise Exception('%s: b is set to %.2f which is no greater than 0!' %
                      (self.name, b))
    self.__b = b

  @property
  def name(self):
    return self.__name

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def UCBV(self) -> np.ndarray:
    """
    Return:
      optimistic estimate of arms' real means using empirical variance
    """
    ucbv = [
        arm.em_mean() +
        np.sqrt(2 * arm.em_var() * np.log(self.__time) / arm.total_pulls()) +
        self.__b * np.log(self.__time) / arm.total_pulls()
        for arm in self.__pseudo_arms
    ]
    return ucbv

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Return:
      arms to pull
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    else:
      self.__last_actions = [(np.argmax(self.UCBV()), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

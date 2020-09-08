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
  def __init__(self, arm_num: int, horizon: int,
               name: str = None, b: float = 1.0):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
      b: upper bound of reward
    """
    super().__init__(arm_num=arm_num, horizon=horizon, name=name)
    if b <= 0:
      raise Exception('%s: b is set to %.2f which is no greater than 0!' %
                      (self.name, b))
    self.__b = b

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'ucbv'

  def reset(self):
    """Learner reset

    Initialization. This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def UCBV(self) -> np.ndarray:
    """
    Returns:
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
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
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

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the ordinary bandit by executing
        `self.__last_actions`.
    """
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

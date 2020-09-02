from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class MOSS(OrdinaryLearner):
  r"""MOSS policy :cite:`audibert2009minimax`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{
    \frac{\mathrm{max}(\ln( \frac{T}{N T_i(t)} ), 0 ) }{T_i(t)} } \right\}

  .. note::
    MOSS uses time horizon in its confidence interval. Reward has to be bounded
    in [0, 1].
  """
  def __init__(self, arm_num: int, horizon: int, name=None):
    super().__init__(arm_num, horizon, name)

  def _name(self):
    return 'moss'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def MOSS(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means using time horizon
    """
    moss = [
        arm.em_mean() + np.sqrt(
            np.maximum(
                0, np.log(self.horizon() /
                          (self.arm_num() * arm.total_pulls()))) /
            arm.total_pulls()) for arm in self.__pseudo_arms
    ]
    return moss

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Returns:
      arms to pull
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    else:
      self.__last_actions = [(np.argmax(self.MOSS()), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

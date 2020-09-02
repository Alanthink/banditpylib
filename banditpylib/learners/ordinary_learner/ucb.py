from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class UCB(OrdinaryLearner):
  r"""Upper confidence bound policy :cite:`auer2002finite`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in [0, N-1]} \left\{ \hat{\mu}_i(t) + \sqrt{ \frac{
    \alpha  \ln(t) }{T_i(t)} } \right\}
  """
  def __init__(self, arm_num: int, horizon: int, name=None, alpha=2.0):
    super().__init__(arm_num, horizon, name)
    if alpha <= 0:
      raise Exception('Alpha %.2f in %s is no greater than 0!' %
                      (alpha, self.__name))
    self.__alpha = alpha

  def _name(self):
    return 'ucb'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def UCB(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means
    """
    ucb = [
        arm.em_mean() +
        np.sqrt(self.__alpha * np.log(self.__time) / arm.total_pulls())
        for arm in self.__pseudo_arms
    ]
    return ucb

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
      self.__last_actions = [(np.argmax(self.UCB()), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

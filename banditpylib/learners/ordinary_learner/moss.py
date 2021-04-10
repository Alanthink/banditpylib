from typing import List, Tuple, Optional

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
  def __init__(self, arm_num: int, horizon: int, name: str = None):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
    """
    super().__init__(arm_num=arm_num, name=name)
    if horizon < arm_num:
      raise Exception('Expected horizon >= %d, got %d' % (arm_num, horizon))
    self.__horizon = horizon

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'moss'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def MOSS(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means
    """
    moss = np.array([
        arm.em_mean + np.sqrt(
            np.maximum(
                0, np.log(self.__horizon /
                          (self.arm_num() * arm.total_pulls()))) /
            arm.total_pulls()) for arm in self.__pseudo_arms
    ])
    return moss

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    else:
      self.__last_actions = [(int(np.argmax(self.MOSS())), 1)]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

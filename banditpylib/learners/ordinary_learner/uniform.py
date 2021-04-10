from typing import List, Tuple, Optional

import numpy as np

from .utils import OrdinaryLearner


class Uniform(OrdinaryLearner):
  r"""Uniform policy

  Play each arm the same number of times.
  """
  def __init__(self, arm_num: int, name=None):
    """
    Args:
      arm_num: number of arms
      name: alias name
    """
    super().__init__(arm_num=arm_num, name=name)

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'uniform'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    # current time step
    self.__time = 1

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    self.__time += 1

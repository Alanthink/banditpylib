from typing import List, Tuple

from .utils import OrdinaryLearner


class Uniform(OrdinaryLearner):
  r"""Uniform policy

  Play each arm the same number of times.
  """
  def __init__(self, arm_num: int, horizon: int, name=None):
    self.__name = name if name else 'uniform'
    super().__init__(arm_num, horizon)

  @property
  def name(self):
    return self.__name

  def reset(self):
    # current time step
    self.__time = 1

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Return:
      arms to pull
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    else:
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__time += 1

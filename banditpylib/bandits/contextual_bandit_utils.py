from abc import ABC, abstractmethod

from typing import Tuple

import numpy as np


class ContextGenerator(ABC):
  """Abstract context generator class

  This class is used to generate the context of contextual bandit.

  :param int arm_num: number of actions
  :param int dimension: dimension of the context
  """
  def __init__(self, arm_num: int, dimension: int):
    self.__arm_num = arm_num
    self.__dimension = dimension

  @property
  @abstractmethod
  def name(self) -> str:
    """Context generator name"""

  @property
  def dimension(self) -> int:
    """Dimension of the context"""
    return self.__dimension

  @property
  def arm_num(self) -> int:
    """Number of actions"""
    return self.__arm_num

  @abstractmethod
  def reset(self):
    """Reset the context generator"""

  @abstractmethod
  def context(self) -> Tuple[np.ndarray, np.ndarray]:
    """Returns:
      the context and the rewards corresponding to different actions
    """


class RandomContextGenerator(ContextGenerator):
  """Random context generator

  Fill contexts and rewards information with random numbers in [0, 1].

  :param int arm_num: number of actions
  :param int dimension: dimension of the context
  """
  def __init__(self, arm_num: int, dimension: int):
    super().__init__(arm_num, dimension)

  @property
  def name(self) -> str:
    return 'random_context_generator'

  def reset(self):
    pass

  def context(self) -> Tuple[np.ndarray, np.ndarray]:
    return (np.random.random(self.dimension), np.random.random(self.arm_num))

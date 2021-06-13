from abc import ABC, abstractmethod

from typing import Optional, Union

import numpy as np


class Arm(ABC):
  """Arm

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """Arm name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default arm name
    """


class StochasticArm(Arm):
  """Stochastic arm

  :param Optional[str] name: alias name
  """
  def __init__(self, name: Optional[str]):
    super().__init__(name)

  @property
  @abstractmethod
  def mean(self) -> float:
    """Mean of rewards"""

  @abstractmethod
  def pull(self, pulls: int = None) -> Union[float, np.ndarray]:
    """Pull the arm

    When `pulls` is `None`, a float number will be returned. Otherwise, a numpy
    array will be returned.

    Args:
      pulls: number of times to pull

    Returns:
      stochastic rewards
    """

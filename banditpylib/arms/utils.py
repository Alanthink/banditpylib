from abc import ABC, abstractmethod

from typing import Optional

import numpy as np


class Arm(ABC):
  """Arm"""
  def __init__(self, name: Optional[str]):
    """
    Args:
      name: alias name for the arm
    """
    self.__name = self._name() if name is None else name

  @property
  def name(self) -> str:
    """arm name"""
    return self.__name

  @abstractmethod
  def _name(self) -> str:
    """
    Returns:
      default arm name
    """

  @property
  @abstractmethod
  def mean(self) -> float:
    """mean of rewards"""

  @abstractmethod
  def pull(self, pulls: int = 1) -> np.ndarray:
    """Pull the arm

    Args:
      pulls: number of times to pull

    Returns:
      rewards
    """

from typing import List
import numpy as np

from .utils import Arm


class CategoricalArm(Arm):
  """Categorical arm

  Arm with rewards generated from a Categorical distribution.
  """

  def __init__(self, p: List[float], name: str = None):
    """
    Args:
      p[i]: probability of coutcome i
      n: number of categories
      name: alias name
    """
    super().__init__(name)
    if sum(p) != 1:
      raise Exception('Sum of probability %.2f is not 1!' % sum(p))
    if any(x < 0 or x > 1 for x in p):
      raise Exception('Elements in %.2f must be between 0 and 1!' % p)
    self.__p = p
    self.__n = len(p)

  # Should be removed. But has to be added here because it is inherated from Arm. 
  def mean(self) -> float:
    """mean of rewards"""
    print('You should not call mean() for the Categorical arm.')
    return sum([x*i for i, x in enumerate(self.__p)])

  def _name(self) -> str:
    """
    Returns:
      default arm name
    """
    return 'categorical_arm'

  @property
  def p(self) -> List[float]:
    """the probability list"""
    return self.__p

  def top_categ(self) -> int:
    """
    Return the index of the category with the highest probability.
    If there are more than one, return the one with the smallest index.
    """
    return self.__p.index(max(self.__p))

  def pull(self, pulls: int = 1) -> np.ndarray:
    """Pull the arm

    Args:
      pulls: number of times to pull

    Returns:
      stochastic category from range(self.__n)
    """
    return np.random.choice(self.__n, pulls, p=self.__p)

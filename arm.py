"""
Arms
"""

from abc import ABC, abstractmethod

import numpy as np


class Arm(ABC):
  """Base class for an arm in the classic bandit model"""

  @abstractmethod
  def pull(self):
    pass


class BernoulliArm(Arm):
  """Bernoulli arm"""

  def __init__(self, mean):
    super().__init__()
    self.mean = mean

  def pull(self):
    """return a stochastic reward"""
    return np.random.binomial(1, self.mean)


class GaussianArm(Arm):
  """Gaussian arm"""

  def __init__(self, mu, sigma):
    super().__init__()
    self.mu = mu
    self.sigma = sigma

  def pull(self):
    """return a stochastic reward"""
    return np.random.normal(self.mu, self.sigma, 1)[0]

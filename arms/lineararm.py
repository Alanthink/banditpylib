import numpy as np
import cvxpy as cp
from .utils import Arm

__all__ = ['LinearArm']


class LinearArm(Arm):
  """Linear arm"""

  def __init__(self, a, theta=1):
    self.__a = np.array(a)
    self.__cp_a = cp.Constant(self.__a)
    self.__mu = np.dot(self.__a, theta)
    self.__phi = 0.0

  @property
  def action(self):
    return self.__a

  def mu_theta(self, em_theta):
    return np.dot(self.__a, em_theta)

  def cp_mu_theta(self, em_cp_theta):
    return self.__cp_a * em_cp_theta

  @property
  def mean(self):
    return self.__mu

  def pull(self, theta, pulls=1):
    """return a numpy array of stochastic rewards using true theta"""
    return np.dot(self.__a, theta) + np.random.normal(0, 1, pulls)

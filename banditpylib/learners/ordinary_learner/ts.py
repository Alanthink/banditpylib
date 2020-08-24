from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class ThompsonSampling(OrdinaryLearner):
  r"""Thompson Sampling policy

  Assume a prior distribution for every arm. At time :math:`t`, sample a
  virtual mean from the posterior distribution for every arm. Play the arm with
  the maximum sampled virtual mean.
  """
  def __init__(self, arm_num: int, horizon: int, prior_dist='beta', name=None):
    """
    Args:
      prior_dist: prior distribution of thompson sampling
    """
    self.__name = name if name else 'thompson_sampling'
    super().__init__(arm_num, horizon)
    if prior_dist not in ['gaussian', 'beta']:
      raise Exception('Prior distribution %s is not supported!' % prior_dist)
    self.__prior_dist = prior_dist

  @property
  def name(self):
    return self.__name

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def actions_from_beta_prior(self) -> int:
    """
    Return:
      arm id of the arm to pull
    """
    # the mean of each arm has a uniform prior Beta(1, 1)
    virtual_means = np.zeros(self.arm_num())
    for arm_id in range(self.arm_num()):
      a = 1 + self.__pseudo_arms[arm_id].total_rewards()
      b = 1 + self.__pseudo_arms[arm_id].total_pulls(
      ) - self.__pseudo_arms[arm_id].total_rewards()
      virtual_means[arm_id] = np.random.beta(a, b)
    return np.argmax(virtual_means)

  def actions_from_gaussian_prior(self) -> int:
    """
    Return:
      arm id of the arm to pull
    """
    # the mean of each arm has a Gaussian prior Normal(0, 1)
    virtual_means = np.zeros(self.arm_num())
    for arm_id in range(self.arm_num()):
      mu = self.__pseudo_arms[arm_id].total_rewards() / (
          self.__pseudo_arms[arm_id].total_pulls() + 1)
      sigma = 1.0 / (self.__pseudo_arms[arm_id].total_pulls() + 1)
      virtual_means[arm_id] = np.random.normal(mu, sigma)
    return np.argmax(virtual_means)

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
    Return:
      [(assortment, 1)]: assortment to serve
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
      return self.__last_actions

    self.__last_actions = [(self.actions_from_beta_prior(),
                            1)] if self.__prior_dist == 'beta' else [
                                (self.actions_from_gaussian_prior(), 1)
                            ]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

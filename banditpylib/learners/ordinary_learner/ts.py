from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class ThompsonSampling(OrdinaryLearner):
  r"""Thompson Sampling policy :cite:`agrawal2017near`

  Assume a prior distribution for every arm. At time :math:`t`, sample a
  virtual mean from the posterior distribution for every arm. Play the arm with
  the maximum sampled virtual mean.

  .. warning::
    Reward should be Bernoulli when Beta prior is chosen.
  """
  def __init__(self, arm_num: int, horizon: int,
               name: str = None, prior_dist: str = 'beta'):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
      prior_dist: prior distribution of thompson sampling. Only two priors are
        supported i.e., `beta` and `gaussian`
    """
    super().__init__(arm_num=arm_num, horizon=horizon, name=name)
    if prior_dist not in ['gaussian', 'beta']:
      raise Exception('Prior distribution %s is not supported!' % prior_dist)
    self.__prior_dist = prior_dist

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'thompson_sampling'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def actions_from_beta_prior(self) -> int:
    """
    Returns:
      arm to pull using beta prior
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
    Returns:
      arm to pull using gaussian prior
    """
    # the mean of each arm has a Gaussian prior Normal(0, 1)
    virtual_means = np.zeros(self.arm_num())
    for arm_id in range(self.arm_num()):
      mu = self.__pseudo_arms[arm_id].total_rewards() / (
          self.__pseudo_arms[arm_id].total_pulls() + 1)
      sigma = 1.0 / (self.__pseudo_arms[arm_id].total_pulls() + 1)
      virtual_means[arm_id] = np.random.normal(mu, sigma)
    return np.argmax(virtual_means)

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
    else:
      self.__last_actions = [(self.actions_from_beta_prior(),
                              1)] if self.__prior_dist == 'beta' else [
                                  (self.actions_from_gaussian_prior(), 1)
                              ]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

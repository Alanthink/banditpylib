from typing import Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import OrdinaryLearner


class ThompsonSampling(OrdinaryLearner):
  r"""Thompson Sampling policy :cite:`agrawal2017near`

  Assume a prior distribution for every arm. At time :math:`t`, sample a
  virtual mean reward from the posterior distribution for every arm. Play the
  arm with the maximum sampled virtual mean reward.

  :param int arm_num: number of arms
  :param str prior_dist: prior distribution of thompson sampling. Only two
    priors are supported i.e., `beta` and `gaussian`
  :param Optional[str] name: alias name

  .. warning::
    Reward should be Bernoulli when Beta prior is chosen.
  """
  def __init__(self,
               arm_num: int,
               prior_dist: str = 'beta',
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    if prior_dist not in ['gaussian', 'beta']:
      raise ValueError('Prior distribution %s is not supported.' % prior_dist)
    self.__prior_dist = prior_dist

  def _name(self) -> str:
    return 'thompson_sampling'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Current time step
    self.__time = 1

  def __sample_from_beta_prior(self) -> int:
    """
    Returns:
      arm to pull using beta prior
    """
    # The average reward of each arm has a uniform prior Beta(1, 1)
    virtual_means = np.zeros(self.arm_num)
    for arm_id in range(self.arm_num):
      a = 1 + self.__pseudo_arms[arm_id].total_rewards
      b = 1 + self.__pseudo_arms[arm_id].total_pulls - self.__pseudo_arms[
          arm_id].total_rewards
      virtual_means[arm_id] = np.random.beta(a, b)
    return int(np.argmax(virtual_means))

  def __sample_from_gaussian_prior(self) -> int:
    """
    Returns:
      arm to pull using gaussian prior
    """
    # The average reward of each arm has a Gaussian prior Normal(0, 1)
    virtual_means = np.zeros(self.arm_num)
    for arm_id in range(self.arm_num):
      mu = self.__pseudo_arms[arm_id].total_rewards / (
          self.__pseudo_arms[arm_id].total_pulls + 1)
      sigma = 1.0 / (self.__pseudo_arms[arm_id].total_pulls + 1)
      virtual_means[arm_id] = np.random.normal(mu, sigma)
    return int(np.argmax(virtual_means))

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()
    arm_pulls_pair.arm.id = self.__sample_from_beta_prior(
    ) if self.__prior_dist == 'beta' else self.__sample_from_gaussian_prior()
    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__pseudo_arms[arm_rewards_pair.arm.id].update(
        np.array(arm_rewards_pair.rewards))
    self.__time += 1

from typing import List, Tuple

import numpy as np

from banditpylib.arms import GaussianArm
from .ordinary_bandit_itf import OrdinaryBanditItf
from .linear_bandit_itf import LinearBanditItf


class LinearBandit(OrdinaryBanditItf, LinearBanditItf):
  """Class for finite-armed linear bandit

  Arms are indexed from 0 by default.
  """

  def __init__(self,
               features: np.ndarray, theta: np.ndarray, var=1.0, name=None):
    """
    Args:
      features: features of the arms
      theta: unknown parameter theta
      var: variance of noise
      name: alias name
    """
    super().__init__(name)
    if len(features) < 2:
      raise Exception('Number of arms %d is less than 2!' % len(features))
    for (i, feature) in enumerate(features):
      if feature.shape != theta.shape:
        raise Exception('Dimension of arm %d\'s feature %d does not equal to '
                        'theta\'s %d!' % (i, len(feature), len(theta)))
    self.__features = features
    self.__theta = theta
    self.__arm_num = len(features)

    if var < 0:
      raise Exception('Variance of noise %d is less than 0!' % var)
    self.__var = var
    # each arm in linear bandit can be seen as a gaussian arm
    self.__arms = [GaussianArm(np.dot(feature, self.__theta), self.__var) \
                   for feature in self.__features]
    self.__best_arm_id = max(
        [(arm_id, arm.mean) for (arm_id, arm) in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  def _name(self):
    return 'linear_bandit'

  def _take_action(self, arm_id, pulls=1) -> Tuple[np.ndarray, None]:
    """Pull one arm

    Args:
      arm_id: arm id
      pulls: number of pulls to apply

    Returns:
      feedback where the first dimension denotes the stochastic rewards
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    self.__total_pulls += pulls
    em_rewards = self.__arms[arm_id].pull(pulls)
    self.__regret += (self.__best_arm.mean * pulls - em_rewards)
    return (em_rewards, None)

  def feed(self,
           actions: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: for each tuple, the first dimension denotes the arm id and the \
      second dimension is the number of times this arm is going to be pulled.

    Returns:
      feedback where for each tuple, the first dimension is the stochatic \
      rewards
    """
    feedback = []
    for (arm_id, pulls) in actions:
      feedback.append(self._take_action(arm_id, pulls))
    return feedback

  def reset(self):
    self.__total_pulls = 0
    self.__regret = 0.0

  # implement methods of ordinary bandit
  def arm_num(self) -> int:
    return self.__arm_num

  def total_pulls(self) -> int:
    return self.__total_pulls

  # implement methods of linear bandit
  def features(self) -> np.ndarray:
    return self.__features

  def regret(self) -> float:
    """
    Returns:
      regret compared with the optimal policy
    """
    return self.__regret

  def best_arm_regret(self, arm_id) -> int:
    """
    Args:
      arm_id: best arm identified by the learner

    Returns:
      regret compared with the best arm
    """
    return self.__best_arm_id != arm_id

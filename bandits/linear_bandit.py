from typing import List, Tuple

import numpy as np

from arms import GaussianArm
from .ordinary_bandit_itf import OrdinaryBanditItf
from .linear_bandit_itf import LinearBanditItf


class LinearBandit(OrdinaryBanditItf, LinearBanditItf):
  """Class for finite-armed linear bandit

  Arms are indexed from 0 by default.
  """

  def __init__(self, features: np.ndarray, theta: np.ndarray, var=1.0):
    """
    Args:
      features: features of the arms. First dimension: number of arms. Second
      dimension: dimension of the features.
      theta: unknown parameter theta
      var: variance of noise
    """
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

  @property
  def name(self):
    return 'linear_bandit'

  def _take_action(self, arm_id, pulls=1) -> Tuple[np.ndarray, None]:
    """Pull one arm

    Args:
      arm_id: arm id
      pulls: number of pulls to apply

    Return:
      feedback: first dimension denotes the stochastic rewards
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    self.__total_pulls += pulls
    return (self.__arms[arm_id].pull(pulls), None)

  def feed(self,
           actions: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: For each tuple, the first dimension denotes the arm id and the
      second dimension is the number of times this arm is going to be pulled.

    Return:
      feedback: For each tuple, the first dimension is the stochatic rewards.
    """
    feedback = []
    for (arm_id, pulls) in actions:
      feedback.append(self._take_action(arm_id, pulls))
    return feedback

  def reset(self):
    self.__total_pulls = 0

  # implement methods for ordinary bandit
  def arm_num(self) -> int:
    return self.__arm_num

  def total_pulls(self) -> int:
    return self.__total_pulls

  # implement methods for linear bandit
  def features(self) -> np.ndarray:
    return self.__features

  def regret(self, rewards):
    """
    Args:
      rewards: empirical rewards obtained by the learner

    Return:
      regret compared with the optimal policy
    """
    return self.__best_arm.mean * self.__total_pulls - rewards

  def best_arm_regret(self, arm_id):
    """
    Args:
      arm_id: best arm identified by the learner

    Return:
      regret compared with the best arm
    """
    return self.__best_arm_id != arm_id

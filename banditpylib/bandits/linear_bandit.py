from typing import List, Tuple

import numpy as np

from banditpylib.arms import GaussianArm
from banditpylib.learners import Goal, BestArmId, MaxReward
from .ordinary_bandit_itf import OrdinaryBanditItf
from .linear_bandit_itf import LinearBanditItf


class LinearBandit(OrdinaryBanditItf, LinearBanditItf):
  r"""Finite-armed linear bandit

  Arms are indexed from 0 by default. Each pull of arm :math:`i` will generate
  an `i.i.d.` reward from distribution :math:`\langle \theta, v_i \rangle
  + \epsilon`, where :math:`v_i` is the feature of arm :math:`i`, :math:`\theta`
  is an unknown parameter and :math:`\epsilon` is a zero-mean noise.
  """

  def __init__(self,
               features: List[np.ndarray],
               theta: np.ndarray,
               var: float = 1.0,
               name: str = None):
    """
    Args:
      features: features of the arms
      theta: parameter theta
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
    # each arm in linear bandit can be seen as a Gaussian arm
    self.__arms = [GaussianArm(np.dot(feature, self.__theta), self.__var) \
                   for feature in self.__features]
    self.__best_arm_id = max(
        [(arm_id, arm.mean) for (arm_id, arm) in enumerate(self.__arms)],
        key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'linear_bandit'

  def _take_action(self, arm_id, pulls=1) -> Tuple[np.ndarray, None]:
    """Pull one arm

    Args:
      arm_id: arm id
      pulls: number of times to pull

    Returns:
      feedback where the first dimension denotes the stochastic rewards
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    em_rewards = self.__arms[arm_id].pull(pulls)
    if em_rewards is not None:
      self.__regret += (self.__best_arm.mean * pulls - em_rewards)
      self.__total_pulls += pulls
    return (em_rewards, None)

  def feed(self,
           actions: List[Tuple[int, int]]) -> List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: for each tuple, the first dimension denotes the arm id and the
        second dimension is the number of times this arm will be pulled

    Returns:
      feedback. For each tuple, the first dimension is the stochatic rewards.
    """
    feedback = []
    for (arm_id, pulls) in actions:
      feedback.append(self._take_action(arm_id, pulls))
    return feedback

  def reset(self):
    """Reset the bandit environment

    .. warning::
      This function should be called before the start of the game.
    """
    self.__total_pulls = 0
    self.__regret = 0.0

  def arm_num(self) -> int:
    """
    Returns:
      total number of arms
    """
    return self.__arm_num

  def total_pulls(self) -> int:
    """
    Returns:
      total number of pulls so far
    """
    return self.__total_pulls

  def features(self) -> List[np.ndarray]:
    """
    Returns:
      feature vectors
    """
    return self.__features

  def __best_arm_regret(self, arm_id) -> int:
    """
    Args:
      arm_id: best arm identified by the learner

    Returns:
      regret compared with the best arm
    """
    return int(self.__best_arm_id != arm_id)

  def regret(self, goal: Goal) -> float:
    """
    Args:
      goal: goal of the learner

    Returns:
      regret of the learner
    """
    if isinstance(goal, BestArmId):
      return self.__best_arm_regret(goal.value)
    elif isinstance(goal, MaxReward):
      return self.__regret
    raise Exception('Goal %s is not supported!' % goal.name)

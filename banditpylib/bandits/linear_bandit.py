from typing import List

import numpy as np

from banditpylib.arms import GaussianArm
from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, BestArmId, MaxReward
from .utils import Bandit


class LinearBandit(Bandit):
  r"""Finite-armed linear bandit

  Arms are indexed from 0 by default. Each pull of arm :math:`i` will generate
  an `i.i.d.` reward from distribution :math:`\langle \theta, v_i \rangle
  + \epsilon`, where :math:`v_i` is the feature of arm :math:`i`, :math:`\theta`
  is an unknown parameter and :math:`\epsilon` is a zero-mean noise.
  """
  def __init__(self,
               features: List[np.ndarray],
               theta: np.ndarray,
               var: float = 1.0):
    """
    Args:
      features: features of the arms
      theta: parameter theta
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
    # each arm in linear bandit can be seen as a Gaussian arm
    self.__arms = [GaussianArm(np.dot(feature, self.__theta), self.__var) \
                   for feature in self.__features]
    self.__best_arm_id = max([(arm_id, arm.mean)
                              for (arm_id, arm) in enumerate(self.__arms)],
                             key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'linear_bandit'

  def context(self):
    """
    Returns:
      current state of the bandit environment
    """
    return None

  def _take_action(self, arm_pulls_pair: ArmPullsPair) -> ArmRewardsPair:
    """Pull one arm

    Args:
      arm_pulls_pair: arm and its pulls

    Returns:
      arm_rewards_pair: arm and its rewards
    """
    arm_id = arm_pulls_pair.arm.id
    pulls = arm_pulls_pair.pulls

    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))

    arm_rewards_pair = ArmRewardsPair()
    if pulls < 1:
      return arm_rewards_pair

    # Empirical rewards when `arm_id` is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls)

    self.__regret += (self.__best_arm.mean * pulls - em_rewards)
    self.__total_pulls += pulls

    arm_rewards_pair.arm.id = arm_id
    arm_rewards_pair.rewards.extend(list(em_rewards)) # type: ignore

    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    """Pull multiple arms

    Args:
      actions: actions to perform

    Returns:
      feedback after actions are performed
    """
    feedback = Feedback()
    for arm_pulls_pair in actions.arm_pulls_pairs:
      arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
      if arm_rewards_pair.rewards:
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
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

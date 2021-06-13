from typing import List

import numpy as np

from banditpylib.arms import GaussianArm
from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, IdentifyBestArm, MaximizeTotalRewards
from .utils import Bandit


class LinearBandit(Bandit):
  r"""Finite-armed linear bandit

  Arms are indexed from 0 by default. Each pull of arm :math:`i` will generate
  an `i.i.d.` reward from distribution :math:`\langle \theta, v_i \rangle
  + \epsilon`, where :math:`v_i` is the feature vector of arm :math:`i`,
  :math:`\theta` is the unknown parameter and :math:`\epsilon` is a zero-mean
  noise.

  :param List[np.ndarray] features: feature vectors of the arms
  :param np.ndarray theta: unknown parameter theta
  :param float std: standard variance of noise
  """
  def __init__(self,
               features: List[np.ndarray],
               theta: np.ndarray,
               std: float = 1.0):
    if len(features) < 2:
      raise ValueError('The number of arms is expected at least 2. Got %d.' %
                       len(features))
    for (i, feature) in enumerate(features):
      if feature.shape != theta.shape:
        raise ValueError('Dimension of arm %d\'s feature vector is expected '
                         'the same as theta\'s. Got %d.' % (i, len(feature)))
    self.__features = features
    self.__theta = theta
    self.__arm_num = len(features)

    if std < 0:
      raise ValueError(
          'Standard deviation of noise is expected greater than 0. Got %.2f' %
          std)
    self.__std = std
    # Each arm in linear bandit can be seen as a Gaussian arm
    self.__arms = [GaussianArm(np.dot(feature, self.__theta), self.__std) \
                   for feature in self.__features]
    self.__best_arm_id = max([(arm_id, arm.mean)
                              for (arm_id, arm) in enumerate(self.__arms)],
                             key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  @property
  def name(self) -> str:
    return 'linear_bandit'

  def context(self):
    return None

  def _take_action(self, arm_pulls_pair: ArmPullsPair) -> ArmRewardsPair:
    """Pull one arm

    Args:
      arm_pulls_pair: arm id and its pulls

    Returns:
      arm_rewards_pair: arm id and its rewards
    """
    arm_id = arm_pulls_pair.arm.id
    pulls = arm_pulls_pair.pulls

    if arm_id not in range(self.__arm_num):
      raise ValueError('Arm id is expected in the range [0, %d). Got %d.' %
                       (self.__arm_num, arm_id))

    arm_rewards_pair = ArmRewardsPair()

    # Empirical rewards when `arm_id` is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls)

    self.__regret += (self.__best_arm.mean * pulls - em_rewards)
    self.__total_pulls += pulls

    arm_rewards_pair.arm.id = arm_id
    arm_rewards_pair.rewards.extend(list(em_rewards))  # type: ignore

    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()
    for arm_pulls_pair in actions.arm_pulls_pairs:
      if arm_pulls_pair.pulls > 0:
        arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
    return feedback

  def reset(self):
    self.__total_pulls = 0
    self.__regret = 0.0

  @property
  def arm_num(self) -> int:
    """
    Returns:
      total number of arms
    """
    return self.__arm_num

  @property
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
      0 if `arm_id` is the best arm else 1
    """
    return int(self.__best_arm_id != arm_id)

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, IdentifyBestArm):
      return self.__best_arm_regret(goal.best_arm)
    elif isinstance(goal, MaximizeTotalRewards):
      return self.__regret
    raise Exception('Goal %s is not supported.' % goal.name)

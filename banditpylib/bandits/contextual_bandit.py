import numpy as np

from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, MaximizeTotalRewards
from .utils import Bandit, ContextGenerator


class ContextualBandit(Bandit):
  r"""Finite-armed contextual bandit

  Arms are indexed from 0 by default. At time :math:`t`, it will generate a
  context and a list of rewards incurred by different arms denoted by
  :math:`(X_t, \{r_i^t\}_i)` where :math:`X_t` is the context and
  :math:`r_i^t` is the reward when arm :math:`i` is pulled. After receiving
  learner's action :math:`a_t`, the reward :math:`r_{a_t}^t` will be revealed to
  the learner. The batched version can be defined in a similar way.

  :param ContextGenerator context_generator: context generator
  """
  def __init__(self, context_generator: ContextGenerator):
    self.__context_generator = context_generator
    self.__arm_num = self.__context_generator.arm_num
    # Maximum rewards the learner can obtain
    self.__regret = 0.0

  @property
  def name(self) -> str:
    return 'contextual_bandit'

  def context(self) -> np.ndarray:
    self.__context_and_rewards = self.__context_generator.context()
    return self.__context_and_rewards[0]

  def __update_regret(self, arm_id: int):
    """Update the regret

    Args:
      arm_id: arm pulled
    """
    rewards = self.__context_and_rewards[1]
    self.__regret += max(rewards) - rewards[arm_id]

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

    self.__update_regret(arm_id)
    for _ in range(pulls - 1):
      self.__context_and_rewards = self.__context_generator.context()
      self.__update_regret(arm_id)
      arm_rewards_pair.rewards.append(self.__context_and_rewards[1][arm_id])

    self.__total_pulls += pulls
    arm_rewards_pair.arm.id = arm_id
    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()

    for arm_pulls_pair in actions.arm_pulls_pairs:
      arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
      if arm_rewards_pair.rewards:
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
    return feedback

  def reset(self):
    self.__context_generator.reset()
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

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, MaximizeTotalRewards):
      return self.__regret
    raise ValueError('Goal %s is not supported.' % goal.name)

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import argmax_or_min_tuple
from .utils import OrdinaryLearner


class ExploreThenCommit(OrdinaryLearner):
  r"""Explore-Then-Commit policy

  During the first :math:`T' \leq T` time steps (exploration period), play each
  arm in a round-robin way. Then for the remaining time steps, play the arm
  with the maximum empirical mean reward within exploration period consistently.
  """
  def __init__(self, arm_num: int, T_prime: int, name: str = None):
    """
    Args:
      arm_num: number of arms
      T_prime: time steps to explore
      name: alias name
    """
    super().__init__(arm_num=arm_num, name=name)
    if T_prime < arm_num:
      raise Exception('T\' is expected to be at least %d, got %d.' %
                      (arm_num, T_prime))
    self.__T_prime = T_prime
    self.__best_arm: int = -1

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'explore_then_commit'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # Current time step
    self.__time = 1

  def actions(self, context=None) -> Actions:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()

    if self.__time <= self.__T_prime:
      arm_pulls_pair.arm.id = (self.__time - 1) % self.arm_num()
    else:
      arm_pulls_pair.arm.id = self.__best_arm

    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__pseudo_arms[arm_rewards_pair.arm.id].update(
        np.array(arm_rewards_pair.rewards))
    self.__time += 1
    if self.__best_arm < 0 and self.__time > self.__T_prime:
      self.__best_arm = argmax_or_min_tuple([
          (self.__pseudo_arms[arm_id].em_mean, arm_id)
          for arm_id in range(self.arm_num())
      ])

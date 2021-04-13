import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import argmax
from .utils import OrdinaryFBBAILearner


class Uniform(OrdinaryFBBAILearner):
  """Uniform sampling policy

  Play each arm the same number of times and then output the arm with the
  highest empirical mean.
  """
  def __init__(self, arm_num: int, budget: int, name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'uniform'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    self.__best_arm = None
    self.__stop = False

  def actions(self, context=None) -> Actions:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context

    actions = Actions()

    if not self.__stop:
      # make sure each arm is sampled at least once
      pulls = np.random.multinomial(self.budget() - self.arm_num(),
                                    np.ones(self.arm_num()) / self.arm_num(),
                                    size=1)[0]
      for arm_id in range(self.arm_num()):
        arm_pulls_pair = actions.arm_pulls_pairs.add()
        arm_pulls_pair.arm.id = arm_id
        arm_pulls_pair.pulls = pulls[arm_id] + 1

      self.__stop = True

    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    for arm_rewards_pair in feedback.arm_rewards_pairs:
      self.__pseudo_arms[arm_rewards_pair.arm.id].update(
          np.array(arm_rewards_pair.rewards))
    if self.__stop:
      self.__best_arm = argmax([arm.em_mean for arm in self.__pseudo_arms])

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner
    """
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm

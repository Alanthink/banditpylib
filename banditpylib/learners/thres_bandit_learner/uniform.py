import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import Goal, AllCorrect
from .utils import ThresBanditLearner


class Uniform(ThresBanditLearner):
  """Uniform Sampling

  Sample each arm in a round-robin way.
  """
  def __init__(self, arm_num: int, theta: float, eps: float, name: str = None):
    """
    Args:
      arm_num: number of arms
      theta: threshold
      eps: radius of indifferent zone
      name: alias name
    """
    super().__init__(arm_num=arm_num, name=name)
    self.__theta = theta
    self.__eps = eps

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'uniform_sampling'

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
      context: context of the thresholding bandit which should be `None`

    Returns:
      arms to pull
    """
    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()
    arm_pulls_pair.arm.id = (self.__time - 1) % self.arm_num()
    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        `actions`
    """
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__pseudo_arms[arm_rewards_pair.arm.id].update(
        np.array(arm_rewards_pair.rewards))
    self.__time += 1

  @property
  def goal(self) -> Goal:
    answers = [
        1 if arm.em_mean >= self.__theta else 0 for arm in self.__pseudo_arms
    ]
    return AllCorrect(answers=answers)

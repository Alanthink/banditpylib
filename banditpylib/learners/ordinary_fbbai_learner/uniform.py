from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryFBBAILearner


class Uniform(OrdinaryFBBAILearner):
  """Uniform policy

  Play each arm the same number of times and then output the arm with the best
  empirical mean.
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
    self.__last_round = False

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__last_round:
      self.__last_actions = None
    else:
      pulls = np.random.multinomial(self.budget(),
                                    np.ones(self.arm_num()) / self.arm_num(),
                                    size=1)[0]
      self.__last_actions = [(arm_id, pulls[arm_id])
                             for arm_id in range(self.arm_num())]
      self.__last_round = True
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    for (ind, (rewards, _)) in enumerate(feedback):
      self.__pseudo_arms[self.__last_actions[ind][0]].update(rewards)
    if self.__last_round:
      self.__best_arm = np.argmax([arm.em_mean for arm in self.__pseudo_arms])

  def best_arm(self) -> int:
    """
    Returns:
      best arm identified by the learner

    .. todo::
      Randomize the output when there are multiple arms with the same empirical
      mean.
    """
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm

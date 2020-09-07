from typing import List, Tuple

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryFBBAILearner


class Uniform(OrdinaryFBBAILearner):
  """Uniform policy

  Play each arm the same number of times and then output the arm with the best
  empirical mean.
  """
  def __init__(self, arm_num: int, budget: int, name=None):
    super().__init__(arm_num=arm_num, budget=budget, name=name)

  def _name(self) -> str:
    return 'uniform'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    self.__best_arm = None
    self.__last_round = False

  def actions(self, context=None) -> List[Tuple[int, int]]:
    """
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

  def update(self, feedback):
    """
    .. todo::
      Randomize the output when there are multiple arms with the same empirical
      mean.
    """
    for (ind, (rewards, _)) in enumerate(feedback):
      self.__pseudo_arms[self.__last_actions[ind][0]].update(rewards)
    if self.__last_round:
      self.__best_arm = np.argmax([arm.em_mean() for arm in self.__pseudo_arms])

  def best_arm(self):
    if self.__best_arm is None:
      raise Exception('%s: I don\'t have an answer yet!' % self.name)
    return self.__best_arm

from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoCategArm
from banditpylib.learners import Goal, AllCorrect
from .utils import ThresBanditLearner


class OPT_KG(ThresBanditLearner):
  """
  Bayesian Thresholding (two class) algorithm for Bernoulli arms. threshold is 0.5.
  :cite:
  """
  def __init__(self,
               arm_num: int,
               budget: int,
               name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      theta: the threshold is fixed at 0.5
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    self.__theta = 0.5

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'opt-kg'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    """
      The default prior for PseudoCategArm is a two class arm with prior Beta[1,1]. So
      the initial value of I and beta function are 0.5 and 1. The incre is defined to
      be 0.5^(a+b)/beta(a,b). So initial value is 0.5^2/1.
    """
    self.__pseudo_categ_arms = [PseudoCategArm() for arm_id in range(self.arm_num())]
    self.__I = [0.5]*self.arm_num()
    self.__incre = [0.25]*self.arm_num()
    # store the metrics and update it only for the arm just pulled at each time step.
    self.__metrics = [float('inf')]*self.arm_num()
    # current time step
    self.__time = 1


  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the thresholding bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    if self.__time > self.budget():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    else:
      self.__last_actions = [(np.argmin(self.__metrics), 1)]
    return self.__last_actions


  def _R1(self, a, I, incre):
    I_new = I + incre/a
    return max([I_new, 1-I_new]) - max([I, 1-I])

  def _R2(self, b, I, incre):
    I_new = I - incre/b
    return max([I_new, 1-I_new]) - max([I, 1-I])


  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        `self.__last_actions`
    """
    last_action = self.__last_actions[0][0]
    a, b = self.__pseudo_categ_arms[last_action].freq
    # update I and incre based on the feedback
    if feedback[0][0][0] == 0:
      self.__I[last_action] += self.__incre[last_action]/a
      self.__incre[last_action] *= 0.5*(a+b)/a
      a += 1
    elif feedback[0][0][0] == 1:
      self.__I[last_action] -= self.__incre[last_action]/b
      self.__incre[last_action] *= 0.5*(a+b)/b
      b += 1

    self.__pseudo_categ_arms[last_action].update(feedback[0][0])
    self.__time += 1
    # update metrics
    self.__metrics[last_action] = -max(
        self._R1(a, self.__I[last_action], self.__incre[last_action]),
        self._R2(b, self.__I[last_action], self.__incre[last_action]))

  @property
  def goal(self) -> Goal:
    answers = [
        1 if arm.freq[1]/(sum(arm.freq)) >= self.__theta else 0 for arm in self.__pseudo_categ_arms
    ]
    print(answers)
    return AllCorrect(answers=answers)

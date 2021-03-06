from typing import List, Tuple, Optional

import numpy as np
from scipy.stats import gamma
import scipy.integrate as integrate

from banditpylib.arms import PseudoCategArm
from banditpylib.learners import Goal, AllCorrect
from .utils import MultiThresBanditLearner


class M_OPT_KG(MultiThresBanditLearner):
  """
  Bayesian Thresholding (two class) algorithm for Bernoulli arms. threshold is 0.5.
  :cite:
  """
  def __init__(self,
               arm_num: int,
               budget: int,
               categ_num: int,
               name: str = None):
    """
    Args:
      arm_num: number of arms
      budget: total number of pulls
      categ_num: The number of categories to each question
      name: alias name
    """
    super().__init__(arm_num=arm_num, budget=budget, name=name)
    self.__categ_num = categ_num


  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'm-opt-kg'

  def __integrand(self, x: float, alpha: List, c: int) -> float:
    f_x = gamma.pdf(x, alpha[c])
    for i, a in enumerate(alpha):
      if i != c:
        f_x *= gamma.cdf(x,a)
    return f_x


  def __I_integral(self, alpha: List) -> List:
    return [integrate.quad(self.__integrand, 0, np.inf, args=(alpha, c))[0] for c in range(len(alpha))]

  # We can also compute I using Monte Carlo.
  #def __I_MC(alpha: List) -> List:



  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_categ_arms = [PseudoCategArm([1]*self.__categ_num) for arm_id in range(self.arm_num())]
    self.__I = [1/self.__categ_num]*self.arm_num()
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


  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        `self.__last_actions`
    """
    last_action = self.__last_actions[0][0]

    self.__pseudo_categ_arms[last_action].update(feedback[0][0])
    self.__time += 1

    # update I based on the feedback I = max(I_c)
    alpha = self.__pseudo_categ_arms[last_action].freq
    self.__I[last_action] = max(self.__I_integral(alpha))

    # update metrics R as in Xi Chen paper page 43 last line

    self.__metrics[last_action] = (
      self.__I[last_action]
      - sum([(alpha[i]/sum(alpha)) * max(self.__I_integral(alpha[:i]+[alpha[i]+1]+alpha[i+1:]))
      for i in range(self.__categ_num)])
    )

  @property
  def goal(self) -> Goal:
    answers = [
        arm.top_categ for arm in self.__pseudo_categ_arms
    ]
    return AllCorrect(answers=answers)

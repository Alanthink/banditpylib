from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class EpsGreedy(OrdinaryLearner):
  r"""Epsilon-Greedy policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability play the arm with the maximum empirical mean.
  """
  def __init__(self,
               arm_num: int,
               horizon: int,
               name: str = None,
               eps: float = 1.0):
    """
    Args:
      arm_num: number of arms
      horizon: total number of time steps
      name: alias name
      eps: epsilon
    """
    super().__init__(arm_num=arm_num, horizon=horizon, name=name)
    if eps <= 0:
      raise Exception('Epsilon %.2f in %s is no greater than 0!' % \
          (eps, self.__name))
    self.__eps = eps

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'epsilon_greedy'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def actions(self, context=None) -> Optional[List[Tuple[int, int]]]:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context
    # pylint: disable=no-member
    if self.__time > self.horizon():
      self.__last_actions = None
    elif self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
    # with probability eps/t, randomly select an arm to pull
    elif np.random.random() <= self.__eps / self.__time:
      self.__last_actions = [(np.random.randint(0, self.arm_num()), 1)]
    else:
      self.__last_actions = [
          (np.argmax(np.array([arm.em_mean for arm in self.__pseudo_arms])), 1)
      ]
    return self.__last_actions

  def update(self, feedback: List[Tuple[np.ndarray, None]]):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

import numpy as np

from banditpylib.arms import PseudoArm
from .utils import OrdinaryLearner


class UCB(OrdinaryLearner):
  r"""Upper confidence bound policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability play the arm with the maximum empirical mean.
  """
  def __init__(self, arm_num: int, horizon: int, alpha=2.0, name=None):
    self.__name = name if name else 'ucb'
    super().__init__(arm_num, horizon)
    if alpha <= 0:
      raise Exception('Alpha %.2f in %s is no greater than 0!' %
                      (alpha, self.__name))
    self.__alpha = alpha

  @property
  def name(self):
    return self.__name

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def UCB(self) -> np.ndarray:
    """
    Return:
      optimistic estimate of arms' real means
    """
    ucb = [
        arm.em_mean() +
        np.sqrt(self.__alpha * np.log(self.__time) / arm.total_pulls())
        for arm in self.__pseudo_arms
    ]
    return ucb

  def actions(self, context=None):
    del context
    if self.__time > self.horizon():
      self.__last_actions = None
      return self.__last_actions

    if self.__time <= self.arm_num():
      self.__last_actions = [((self.__time - 1) % self.arm_num(), 1)]
      return self.__last_actions

    self.__last_actions = [(np.argmax(self.UCB()), 1)]
    return self.__last_actions

  def update(self, feedback):
    self.__pseudo_arms[self.__last_actions[0][0]].update(feedback[0][0])
    self.__time += 1

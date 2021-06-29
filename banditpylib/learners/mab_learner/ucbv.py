from typing import Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class UCBV(MABLearner):
  r"""UCBV policy :cite:`audibert2009exploration`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in \{0, \dots, N-1\}} \left\{ \bar{\mu}_i(t) +
    \sqrt{ \frac{ 2 \bar{V}_i(t) \ln(t) }{T_i(t)} }+
    \frac{ b \ln(t) }{T_i(t)} \right\}

  :param int arm_num: number of arms
  :param float b: upper bound of rewards
  :param Optional[str] name: alias name

  .. note::
    Reward has to be bounded within :math:`[0, b]`.
  """
  def __init__(self, arm_num: int, b: float = 1.0, name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    if b <= 0:
      raise ValueError('B is expected greater than 0. Got %.2f.' % b)
    self.__b = b

  def _name(self) -> str:
    return 'ucbv'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Current time step
    self.__time = 1

  def __UCBV(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means using empirical variance
    """
    ucbv = np.array([
        arm.em_mean +
        np.sqrt(2 * arm.em_var * np.log(self.__time) / arm.total_pulls) +
        self.__b * np.log(self.__time) / arm.total_pulls
        for arm in self.__pseudo_arms
    ])
    return ucbv

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()

    if self.__time <= self.arm_num:
      arm_pull.arm.id = self.__time - 1
    else:
      arm_pull.arm.id = int(np.argmax(self.__UCBV()))

    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    self.__pseudo_arms[arm_feedback.arm.id].update(
        np.array(arm_feedback.rewards))
    self.__time += 1

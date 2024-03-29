from typing import Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class UCB(MABLearner):
  r"""Upper Confidence Bound policy :cite:`auer2002finite`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in \{0, \dots, N-1\}} \left\{ \bar{\mu}_i(t) +
    \sqrt{ \frac{\alpha  \ln(t) }{T_i(t)} } \right\}

  :param int arm_num: number of arms
  :param float alpha: alpha
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               alpha: float = 2.0,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    if alpha <= 0:
      raise ValueError('Alpha is expected greater than 0. Got %.2f.' % alpha)
    self.__alpha = alpha

  def _name(self) -> str:
    return 'ucb'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Current time step
    self.__time = 1

  def __UCB(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means
    """
    ucb = np.array([
        arm.em_mean +
        np.sqrt(self.__alpha * np.log(self.__time) / arm.total_pulls)
        for arm in self.__pseudo_arms
    ])
    return ucb

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()

    if self.__time <= self.arm_num:
      arm_pull.arm.id = self.__time - 1
    else:
      arm_pull.arm.id = int(np.argmax(self.__UCB()))

    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    self.__pseudo_arms[arm_feedback.arm.id].update(
        np.array(arm_feedback.rewards))
    self.__time += 1

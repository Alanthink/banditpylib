from typing import Optional

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from banditpylib.learners import Goal, MakeAllAnswersCorrect
from .utils import ThresholdingBanditLearner


class APT(ThresholdingBanditLearner):
  """Anytime Parameter-free Thresholding algorithm
  :cite:`DBLP:conf/icml/LocatelliGC16`

  :param int arm_num: number of arms
  :param float theta: threshold
  :param float eps: radius of indifferent zone
  :param Optional[str] name: alias name
  """
  def __init__(self,
               arm_num: int,
               theta: float,
               eps: float,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    self.__theta = theta
    self.__eps = eps

  def _name(self) -> str:
    return 'apt'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Current time step
    self.__time = 1

  def __metrics(self) -> np.ndarray:
    """
    Returns:
      metrics of apt for each arm
    """
    metrics = np.array([
        np.sqrt(arm.total_pulls) *
        (np.abs(arm.em_mean - self.__theta) + self.__eps)
        for arm in self.__pseudo_arms
    ])
    return metrics

  def actions(self, context: Context) -> Actions:
    actions = Actions()
    arm_pull = actions.arm_pulls.add()

    if self.__time <= self.arm_num:
      arm_pull.arm.id = self.__time - 1
    else:
      arm_pull.arm.id = int(np.argmin(self.__metrics()))

    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    self.__pseudo_arms[arm_feedback.arm.id].update(
        np.array(arm_feedback.rewards))
    self.__time += 1

  @property
  def goal(self) -> Goal:
    answers = [
        1 if arm.em_mean >= self.__theta else 0 for arm in self.__pseudo_arms
    ]
    return MakeAllAnswersCorrect(answers=answers)

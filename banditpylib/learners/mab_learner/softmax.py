from typing import Optional

import math

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class Softmax(MABLearner):
  r"""Softmax policy

  At time :math:`t`, sample arm :math:`i` to play with sampling weight

  .. math::
    \exp\left( \bar{\mu}_i(t) / \gamma \right)

  where :math:`\gamma` is a parameter to control how much exploration we want.

  :param int arm_num: number of arms
  :param float gamma: gamma
  :param Optional[str] name: alias name

  .. note::
    When :math:`\gamma` approaches 0, the learner will have an increasing
    probability to select the arm with the maximum empirical mean rewards. When
    :math:`\gamma` approaches to infinity, the policy of the learner tends to
    become uniform sampling.
  """
  def __init__(self,
               arm_num: int,
               gamma: float = 1.0,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    if gamma <= 0:
      raise ValueError('Gamma is expected greater than 0. Got %.2f.' % gamma)
    self.__gamma = gamma

  def _name(self) -> str:
    return 'softmax'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num)]
    # Current time step
    self.__time = 1

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()

    if self.__time <= self.arm_num:
      arm_pull.arm.id = self.__time - 1
    else:
      weights = np.array([
          math.exp(self.__pseudo_arms[arm_id].em_mean / self.__gamma)
          for arm_id in range(self.arm_num)
      ])
      arm_pull.arm.id = np.random.choice(
          self.arm_num, 1, p=[weight / sum(weights) for weight in weights])[0]

    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    self.__pseudo_arms[arm_feedback.arm.id].update(
        np.array(arm_feedback.rewards))
    self.__time += 1

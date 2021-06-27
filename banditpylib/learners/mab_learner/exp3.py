from typing import Optional

import numpy as np

from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class EXP3(MABLearner):
  r"""EXP3 policy :cite:`DBLP:journals/siamcomp/AuerCFS02`

  At time :math:`t`, with probability :math:`\gamma`, uniformly randomly sample
  an arm to play. With the remaining probability i.e., :math:`(1 - \gamma)`,
  sample arm :math:`i` to play with sampling weight

  .. math::
    \left\{
    \begin{array}
    ~w_i^{t-1} & \text{if}~i_{t-1} \neq i \\
    w_i^{t-1} \exp\left( \frac{\gamma}{N}
    \frac{X_i^{t-1}}{p_i^{t-1}} \right) & \text{if}~i_{t-1} = i \\
    \end{array}
    \right.

  where :math:`w_i^{t-1}` and :math:`p_i^{t-1}` denote the weight of arm
  :math:`i` and the probability to pull arm :math:`i` at time :math:`(t-1)`
  respectively and initially we set :math:`w_i^0 = 1` for every arm
  :math:`i \in \{0, \dots, N-1\}`.

  :param int arm_num: number of arms
  :param float gamma: probability to do uniform sampling
  :param str name: alias name
  """
  def __init__(self,
               arm_num: int,
               gamma: float = 0.01,
               name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)
    if gamma < 0 or gamma > 1:
      raise ValueError('Gamma is expected in [0, 1]. Got %.2f.' % gamma)
    self.__gamma = gamma

  def _name(self) -> str:
    return 'exp3'

  def reset(self):
    self.__weights = np.array([1] * self.arm_num)
    # Current time step
    self.__time = 1

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()
    self.__probabilities = (1 - self.__gamma) * self.__weights / sum(
        self.__weights) + self.__gamma / self.arm_num
    arm_pull.arm.id = np.random.choice(self.arm_num, 1,
                                       p=self.__probabilities)[0]
    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]
    arm_id = arm_feedback.arm.id
    reward = arm_feedback.rewards[0]
    estimated_mean = reward / self.__probabilities[arm_id]
    self.__weights[arm_id] *= np.exp(self.__gamma / self.arm_num *
                                     estimated_mean)
    self.__time += 1

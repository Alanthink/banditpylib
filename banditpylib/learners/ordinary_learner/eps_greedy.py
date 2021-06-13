import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryLearner


class EpsGreedy(OrdinaryLearner):
  r"""Epsilon-Greedy policy

  With probability :math:`\frac{\epsilon}{t}` do uniform sampling and with the
  remaining probability play the arm with the maximum empirical mean.

  :param int arm_num: number of arms
  :param str name: alias name
  :param float eps: epsilon
  """
  def __init__(self, arm_num: int, name: str = None, eps: float = 1.0):
    super().__init__(arm_num=arm_num, name=name)
    if eps <= 0:
      raise ValueError('Epsilon is expected greater than 0. Got %.2f.' % eps)
    self.__eps = eps

  def _name(self) -> str:
    return 'epsilon_greedy'

  def reset(self):
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # Current time step
    self.__time = 1

  def actions(self, context=None) -> Actions:
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()

    if self.__time <= self.arm_num():
      arm_pulls_pair.arm.id = self.__time - 1
    # With probability eps/t, randomly select an arm to pull
    elif np.random.random() <= self.__eps / self.__time:
      arm_pulls_pair.arm.id = np.random.randint(0, self.arm_num())
    else:
      arm_pulls_pair.arm.id = int(
          np.argmax(np.array([arm.em_mean for arm in self.__pseudo_arms])))

    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__pseudo_arms[arm_rewards_pair.arm.id].update(
        np.array(arm_rewards_pair.rewards))
    self.__time += 1

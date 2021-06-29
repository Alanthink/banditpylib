from typing import Optional

from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import MABLearner


class Uniform(MABLearner):
  r"""Uniform policy

  Play each arm in a round-robin way.

  :param int arm_num: number of arms
  :param Optional[str] name: alias name
  """
  def __init__(self, arm_num: int, name: Optional[str] = None):
    super().__init__(arm_num=arm_num, name=name)

  def _name(self) -> str:
    return 'uniform'

  def reset(self):
    # Current time step
    self.__time = 1

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()
    arm_pull.arm.id = (self.__time - 1) % self.arm_num
    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    self.__time += 1

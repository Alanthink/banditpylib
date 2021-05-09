from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryLearner


class Uniform(OrdinaryLearner):
  r"""Uniform policy

  Play each arm in a round-robin way.
  """
  def __init__(self, arm_num: int, name=None):
    """
    Args:
      arm_num: number of arms
      name: alias name
    """
    super().__init__(arm_num=arm_num, name=name)

  def _name(self) -> str:
    return 'uniform'

  def reset(self):
    # Current time step
    self.__time = 1

  def actions(self, context=None) -> Actions:
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()
    arm_pulls_pair.arm.id = (self.__time - 1) % self.arm_num()
    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    self.__time += 1

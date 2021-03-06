import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryLearner


class UCBV(OrdinaryLearner):
  r"""UCBV policy :cite:`audibert2009exploration`

  At time :math:`t`, play arm

  .. math::
    \mathrm{argmax}_{i \in \{0, \dots, N-1\}} \left\{ \bar{\mu}_i(t) +
    \sqrt{ \frac{ 2 \bar{V}_i(t) \ln(t) }{T_i(t)} }+
    \frac{ b \ln(t) }{T_i(t)} \right\}

  .. note::
    Reward has to be bounded within :math:`[0, b]`.
  """
  def __init__(self, arm_num: int, name: str = None, b: float = 1.0):
    """
    Args:
      arm_num: number of arms
      name: alias name
      b: upper bound of reward
    """
    super().__init__(arm_num=arm_num, name=name)
    if b <= 0:
      raise Exception('%s: b is set to %.2f which is no greater than 0!' %
                      (self.name, b))
    self.__b = b

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'ucbv'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__pseudo_arms = [PseudoArm() for arm_id in range(self.arm_num())]
    # current time step
    self.__time = 1

  def __UCBV(self) -> np.ndarray:
    """
    Returns:
      optimistic estimate of arms' real means using empirical variance
    """
    ucbv = np.array([
        arm.em_mean +
        np.sqrt(2 * arm.em_var * np.log(self.__time) / arm.total_pulls()) +
        self.__b * np.log(self.__time) / arm.total_pulls()
        for arm in self.__pseudo_arms
    ])
    return ucbv

  def actions(self, context=None) -> Actions:
    """
    Args:
      context: context of the ordinary bandit which should be `None`

    Returns:
      arms to pull
    """
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()

    if self.__time <= self.arm_num():
      arm_pulls_pair.arm.id = self.__time - 1
    else:
      arm_pulls_pair.arm.id = int(np.argmax(self.__UCBV()))

    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    self.__pseudo_arms[arm_rewards_pair.arm.id].update(
        np.array(arm_rewards_pair.rewards))
    self.__time += 1

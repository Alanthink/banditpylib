import numpy as np

from banditpylib.data_pb2 import Actions, Feedback
from .utils import OrdinaryLearner


class EXP3(OrdinaryLearner):
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
  """
  def __init__(self, arm_num: int, name: str = None, gamma: float = 0.01):
    """
    Args:
      arm_num: number of arms
      name: alias name
      gamma: probability to do uniform sampling
    """
    super().__init__(arm_num=arm_num, name=name)
    if gamma < 0 or gamma > 1:
      raise Exception('Gamma %.2f is out of range [0, 1].' % gamma)
    self.__gamma = gamma

  def _name(self) -> str:
    """
    Returns:
      default learner name
    """
    return 'exp3'

  def reset(self):
    """Reset the learner

    .. warning::
      This function should be called before the start of the game.
    """
    self.__weights = np.array([1] * self.arm_num())
    # Current time step
    self.__time = 1

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
    self.__probabilities = (1 - self.__gamma) * self.__weights / sum(
        self.__weights) + self.__gamma / self.arm_num()
    arm_pulls_pair.arm.id = np.random.choice(self.arm_num(),
                                             1,
                                             p=self.__probabilities)[0]
    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    """Learner update

    Args:
      feedback: feedback returned by the bandit environment by executing
        :func:`actions`
    """
    arm_rewards_pair = feedback.arm_rewards_pairs[0]
    arm_id = arm_rewards_pair.arm.id
    reward = arm_rewards_pair.rewards[0]
    estimated_mean = reward / self.__probabilities[arm_id]
    self.__weights[arm_id] *= np.exp(self.__gamma / self.arm_num() *
                                     estimated_mean)
    self.__time += 1

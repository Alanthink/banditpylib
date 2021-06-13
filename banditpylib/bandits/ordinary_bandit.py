from typing import List

from banditpylib.arms import StochasticArm
from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, IdentifyBestArm, MaximizeTotalRewards
from .utils import Bandit


class OrdinaryBandit(Bandit):
  r"""Ordinary bandit

  Arms are indexed from 0 by default. Each pull of arm :math:`i` will generate
  an `i.i.d.` reward from distribution :math:`\mathcal{D}_i`, which is unknown
  beforehand.

  :param List[StochasticArm] arms: available arms
  """
  def __init__(self, arms: List[StochasticArm]):
    if len(arms) < 2:
      raise ValueError('The number of arms is expected at least 2. Got %d.' %
                       len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # Find the best arm
    self.__best_arm_id = max([(arm_id, arm.mean)
                              for (arm_id, arm) in enumerate(self.__arms)],
                             key=lambda x: x[1])[0]
    self.__best_arm = self.__arms[self.__best_arm_id]

  @property
  def name(self) -> str:
    return 'ordinary_bandit'

  def context(self):
    return None

  def _take_action(self, arm_pulls_pair: ArmPullsPair) -> ArmRewardsPair:
    """Pull one arm

    Args:
      arm_pulls_pair: arm id and its pulls

    Returns:
      arm_rewards_pair: arm id and its empirical rewards
    """
    arm_id = arm_pulls_pair.arm.id
    pulls = arm_pulls_pair.pulls

    if arm_id not in range(self.__arm_num):
      raise ValueError('Arm id is expected in the range [0, %d). Got %d.' %
                       (self.__arm_num, arm_id))

    arm_rewards_pair = ArmRewardsPair()

    # Empirical rewards when `arm_id is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls=pulls)

    self.__regret += (
        self.__best_arm.mean * pulls - sum(em_rewards)  # type: ignore
    )
    self.__total_pulls += pulls

    arm_rewards_pair.arm.id = arm_id
    arm_rewards_pair.rewards.extend(list(em_rewards))  # type: ignore

    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()
    for arm_pulls_pair in actions.arm_pulls_pairs:
      if arm_pulls_pair.pulls > 0:
        arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
    return feedback

  def reset(self):
    self.__total_pulls = 0
    self.__regret = 0.0

  @property
  def arm_num(self) -> int:
    """
    Returns:
      total number of arms
    """
    return self.__arm_num

  def __best_arm_regret(self, arm_id: int) -> int:
    """
    Args:
      arm_id: best arm identified by the learner

    Returns:
      0 if `arm_id` is the best arm else 1
    """
    return int(self.__best_arm_id != arm_id)

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, IdentifyBestArm):
      return self.__best_arm_regret(goal.best_arm.id)
    elif isinstance(goal, MaximizeTotalRewards):
      return self.__regret
    raise Exception('Goal %s is not supported.' % goal.name)

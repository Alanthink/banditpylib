from typing import List

from banditpylib.arms import Arm
from banditpylib.data_pb2 import Actions, Feedback, ArmPullsPair, ArmRewardsPair
from banditpylib.learners import Goal, MaxCorrectAnswers, AllCorrect
from .utils import Bandit


class ThresholdingBandit(Bandit):
  r"""Thresholding bandit environment

  Arms are indexed from 0 by default. Each time the learner pulls arm :math:`i`,
  she will obtain an `i.i.d.` reward generated from an `unknown` distribution
  :math:`\mathcal{D}_i`. Different from the ordinary MAB, there is a threshold
  parameter :math:`\theta`. The learner should try to infer whether an arm's
  expected reward is above the threshold or not. Besides, the environment also
  accepts a parameter :math:`\epsilon >= 0` which is the radius of indifference
  zone meaning that the answers about the arms with expected rewards within
  :math:`[\theta - \epsilon, \theta + \epsilon]` do not matter.
  """
  def __init__(self, arms: List[Arm], theta: float, eps: float):
    """
    Args:
      arms: arms in thresholding bandit
      theta: threshold
      eps: radius of indifferent zone
      name: alias name
    """
    if len(arms) < 2:
      raise Exception('The number of arms %d is less than 2.' % len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # Correct answers of all the arms whether its expected rewards is above the
    # threshold or not
    self.__correct_answers = [
        1 if self.__arms[arm_id].mean >= theta else 0
        for arm_id in range(self.__arm_num)
    ]
    if eps < 0:
      raise Exception('Radius of indifference zone is less than 0!')
    # The answer of the learner does not matter if the expected rewards of an
    # arm is within the range [theta-eps, theta+eps]. Hence weight assigned to
    # such an arm is 0.
    self.__weights = [
        0 if theta - eps <= self.__arms[arm_id].mean <= theta + eps else 1
        for arm_id in range(self.__arm_num)
    ]

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'thresholding_bandit'

  def reset(self):
    """Reset the bandit environment

    .. warning::
      This function should be called before the start of the game.
    """
    self.__total_pulls = 0

  def arm_num(self) -> int:
    """
    Returns:
      total number of arms
    """
    return self.__arm_num

  def _take_action(self, arm_pulls_pair: ArmPullsPair) -> ArmRewardsPair:
    """Pull one arm

    Args:
      arm_pulls_pair: arm and its pulls

    Returns:
      arm_rewards_pair: arm and its rewards
    """
    arm_id = arm_pulls_pair.arm.id
    pulls = arm_pulls_pair.pulls

    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))

    arm_rewards_pair = ArmRewardsPair()
    if pulls < 1:
      return arm_rewards_pair

    # Empirical rewards when `arm_id` is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls=pulls)
    self.__total_pulls += pulls

    arm_rewards_pair.arm.id = arm_id
    arm_rewards_pair.rewards.extend(list(em_rewards))  # type: ignore

    return arm_rewards_pair

  def feed(self, actions: Actions) -> Feedback:
    """Pull multiple arms

    Args:
      actions: actions to perform

    Returns:
      feedback after actions are performed
    """
    feedback = Feedback()
    for arm_pulls_pair in actions.arm_pulls_pairs:
      arm_rewards_pair = self._take_action(arm_pulls_pair=arm_pulls_pair)
      if arm_rewards_pair.rewards:
        feedback.arm_rewards_pairs.append(arm_rewards_pair)
    return feedback

  def context(self) -> None:
    """
    Returns:
      current context of the bandit environment
    """
    return None

  def regret(self, goal: Goal) -> float:
    """
    Args:
      goal: goal of the learner

    Returns:
      regret of the learner
    """
    if isinstance(goal, MaxCorrectAnswers):
      # Aggregate regret which is equal to the number of wrong answers
      agg_regret = 0
      for arm_id in range(self.__arm_num):
        agg_regret += (goal.value[arm_id] !=
                       self.__correct_answers[arm_id]) * self.__weights[arm_id]
      return agg_regret
    elif isinstance(goal, AllCorrect):
      # Simple regret which is 1 when there is at least one wrong answer and 0
      # otherwise
      for arm_id in range(self.__arm_num):
        if (goal.value[arm_id] !=
            self.__correct_answers[arm_id]) and self.__weights[arm_id] == 1:
          return 1
      return 0
    raise Exception('Goal %s is not supported.' % goal.name)

from typing import List

from banditpylib.arms import StochasticArm
from banditpylib.data_pb2 import Context, Actions, Feedback, ArmPull, \
    ArmFeedback
from banditpylib.learners import Goal, MaximizeCorrectAnswers, \
    MakeAllAnswersCorrect
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


  :param List[StochasticArm] arms: arms in thresholding bandit
  :param float theta: threshold
  :param float eps: radius of indifferent zone
  """
  def __init__(self, arms: List[StochasticArm], theta: float, eps: float):
    if len(arms) < 2:
      raise ValueError('Number of arms is expected at least 2. Got %d.' %
                       len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # Correct answers of all the arms whether its expected rewards is above the
    # threshold or not
    self.__correct_answers = [
        1 if self.__arms[arm_id].mean >= theta else 0
        for arm_id in range(self.__arm_num)
    ]
    if eps < 0:
      raise ValueError(
          'Radius of indifference zone is expected at least 0. Got %.2f.' %
          eps)
    # The answer of the learner does not matter if the expected rewards of an
    # arm is within the range [theta-eps, theta+eps]. Hence weight assigned to
    # such an arm is 0.
    self.__weights = [
        0 if theta - eps <= self.__arms[arm_id].mean <= theta + eps else 1
        for arm_id in range(self.__arm_num)
    ]

  @property
  def name(self) -> str:
    return 'thresholding_bandit'

  def reset(self):
    self.__total_pulls = 0

  @property
  def arm_num(self) -> int:
    """Total number of arms"""
    return self.__arm_num

  def _take_action(self, arm_pull: ArmPull) -> ArmFeedback:
    """Pull one arm

    Args:
      arm_pull: arm and its pulls

    Returns:
      arm_feedback: arm and its feedback
    """
    arm_id = arm_pull.arm.id
    pulls = arm_pull.times

    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))

    arm_feedback = ArmFeedback()
    if pulls < 1:
      return arm_feedback

    # Empirical rewards when `arm_id` is pulled for `pulls` times
    em_rewards = self.__arms[arm_id].pull(pulls=pulls)
    self.__total_pulls += pulls

    arm_feedback.arm.id = arm_id
    arm_feedback.rewards.extend(list(em_rewards))  # type: ignore

    return arm_feedback

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()
    for arm_pull in actions.arm_pulls:
      arm_feedback = self._take_action(arm_pull=arm_pull)
      if arm_feedback.rewards:
        feedback.arm_feedbacks.append(arm_feedback)
    return feedback

  @property
  def context(self) -> Context:
    return Context()

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, MaximizeCorrectAnswers):
      # Aggregate regret which is equal to the number of wrong answers
      agg_regret = 0
      for arm_id in range(self.__arm_num):
        agg_regret += (goal.answers[arm_id] !=
                       self.__correct_answers[arm_id]) * self.__weights[arm_id]
      return agg_regret
    elif isinstance(goal, MakeAllAnswersCorrect):
      # Simple regret which is 1 when there is at least one wrong answer and 0
      # otherwise
      for arm_id in range(self.__arm_num):
        if (goal.answers[arm_id] !=
            self.__correct_answers[arm_id]) and self.__weights[arm_id] == 1:
          return 1
      return 0
    raise Exception('Goal %s is not supported.' % goal.name)

from banditpylib.data_pb2 import Context, Actions, Feedback, ArmPull, \
    ArmFeedback
from banditpylib.learners import Goal, MaximizeTotalRewards
from .contextual_bandit_utils import ContextGenerator
from .utils import Bandit


class ContextualBandit(Bandit):
  r"""Finite-armed contextual bandit

  Arms are indexed from 0 by default. At time :math:`t`, it will generate a
  context and a list of rewards incurred by different arms denoted by
  :math:`(X_t, \{r_i^t\}_i)` where :math:`X_t` is the context and
  :math:`r_i^t` is the reward when arm :math:`i` is pulled. After receiving
  learner's action :math:`a_t`, the reward :math:`r_{a_t}^t` will be revealed to
  the learner. The batched version can be defined in a similar way.

  :param ContextGenerator context_generator: context generator
  """
  def __init__(self, context_generator: ContextGenerator):
    self.__context_generator = context_generator
    self.__arm_num = self.__context_generator.arm_num
    # Maximum rewards the learner can obtain
    self.__regret = 0.0

  @property
  def name(self) -> str:
    return 'contextual_bandit'

  @property
  def context(self) -> Context:
    self.__context_and_rewards = self.__context_generator.context()
    context = Context()
    context.sequential_context.value.extend(self.__context_and_rewards[0])
    return context

  def __update_regret(self, arm_id: int):
    """Update the regret

    Args:
      arm_id: arm pulled
    """
    rewards = self.__context_and_rewards[1]
    self.__regret += max(rewards) - rewards[arm_id]

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

    self.__update_regret(arm_id)
    for _ in range(pulls - 1):
      self.__context_and_rewards = self.__context_generator.context()
      self.__update_regret(arm_id)
      arm_feedback.rewards.append(self.__context_and_rewards[1][arm_id])

    arm_feedback.arm.id = arm_id
    return arm_feedback

  def feed(self, actions: Actions) -> Feedback:
    feedback = Feedback()

    for arm_pull in actions.arm_pulls:
      arm_feedback = self._take_action(arm_pull=arm_pull)
      if arm_feedback.rewards:
        feedback.arm_feedbacks.append(arm_feedback)
    return feedback

  def reset(self):
    self.__context_generator.reset()
    self.__regret = 0.0

  @property
  def arm_num(self) -> int:
    """Total number of arms"""
    return self.__arm_num

  def regret(self, goal: Goal) -> float:
    if isinstance(goal, MaximizeTotalRewards):
      return self.__regret
    raise ValueError('Goal %s is not supported.' % goal.name)

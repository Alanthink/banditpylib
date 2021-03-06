from typing import List, Tuple, Optional

import numpy as np

from banditpylib.arms import CategoricalArm
from banditpylib.learners import Goal, MaxCorrectAnswers, AllCorrect
from .utils import Bandit


class MultiThresholdingBandit(Bandit):
  r"""Thresholding Bandit Environment

  Arms are indexed from 0 by default. Each time the learner pulls arm :math:`i`,
  she will obtain an `i.i.d.` label generated from a categorical distribution.
  Different from the ordinary MAB, the learner should try to infer the true
  category of an arm.
  """
  def __init__(self,
               arms: List[CategoricalArm],
               name: str = None):
    """
    Args:
      arms: arms in multi-thresholding bandit
      name: alias name
    """
    super().__init__(name)
    if len(arms) < 2:
      raise Exception('The number of arms %d is less than 2.' % len(arms))
    self.__arms = arms
    self.__arm_num = len(arms)
    # correct answers of all the arms, which is the category with the highest probability
    self.__correct_answers = [
        self.__arms[arm_id].top_categ() for arm_id in range(self.__arm_num)
    ]

  def _name(self) -> str:
    """
    Returns:
      default bandit name
    """
    return 'multi_thresholding_bandit'

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

  def __take_action(self, arm_id: int, pulls: int) -> \
      Optional[Tuple[np.ndarray, None]]:
    """Pull one arm

    Args:
      arm_id: arm to pull
      pulls: number of times to pull

    Returns:
      stochastic rewards after `arm_id` is pulled. The first element is the
        stochstic rewards. `None` is returned if `pulls` is less than 1.
    """
    if arm_id not in range(self.__arm_num):
      raise Exception('Arm id %d is out of range [0, %d)!' % \
          (arm_id, self.__arm_num))
    if pulls < 1:
      return None
    # empirical labels when `arm_id` is pulled for `pulls` times
    em_labels = self.__arms[arm_id].pull(pulls=pulls)
    self.__total_pulls += pulls
    return (em_labels, None)

  def feed(self, actions: List[Tuple[int, int]]) -> \
        List[Tuple[np.ndarray, None]]:
    """Pull multiple arms

    Args:
      actions: for each tuple, the first element is the arm id and the
        second element is the pull times

    Returns:
      feedback after arms are pulled. For each tuple, the first element is
        the stochstic labels.
    """
    feedback = []
    for (arm_id, pulls) in actions:
      stochastic_labels = self.__take_action(arm_id=arm_id, pulls=pulls)
      if stochastic_labels is not None:
        feedback.append(stochastic_labels)
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
      # aggregate regret which is equal to the number of wrong answers
      agg_regret = 0
      for arm_id in range(self.__arm_num):
        agg_regret += (goal.value[arm_id] !=
                       self.__correct_answers[arm_id])
      return agg_regret
    elif isinstance(goal, AllCorrect):
      # simple regret which is 1 when there is at least one wrong answer and 0
      # otherwise
      for arm_id in range(self.__arm_num):
        if (goal.value[arm_id] !=
            self.__correct_answers[arm_id]) == 1:
          return 1
      return 0
    raise Exception('Goal %s is not supported.' % goal.name)

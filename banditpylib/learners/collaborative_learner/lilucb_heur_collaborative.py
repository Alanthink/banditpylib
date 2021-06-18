import numpy as np
import math
import random
from copy import deepcopy as dcopy
from typing import Tuple

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Feedback, CollaborativeActions
from banditpylib import argmax_or_min_tuple

from banditpylib.learners.ordinary_fcbai_learner import OrdinaryFCBAILearner

class LilUCBHeuristicCollaborative(OrdinaryFCBAILearner):
  """LilUCB heuristic policy :cite:`jamieson2014lil`
  Modified implementation to supplement CollaborativeAgent
  along with additional functionality to work on only a subset of arms

  :param int arm_num: number of arms of the bandit
  :param float confidence: confidence level. It should be within (0, 1). The
    algorithm should output the best arm with probability at least this value.
  :param np.ndarray assigned_arms: arm indices the learner has to work with
  :param str name: alias name
  """
  def __init__(self, arm_num: int, confidence: float, assigned_arms: np.ndarray = None, name: str = None):
    assert np.max(assigned_arms)<arm_num and len(assigned_arms)<=arm_num, (
      "assigned_arms should be a subset of [arm_num], with unique arm indices\nReceived "
        + str(assigned_arms))
    super().__init__(arm_num=arm_num, confidence=confidence, name=name)
    if assigned_arms is not None:
      self.__assigned_arms = assigned_arms
    else:
      self.__assigned_arms = np.arange(arm_num)


  def _name(self) -> str:
    return 'lilUCB_heur_collaborative'

  def reset(self):
    # create only as many local arms as num_assigned_arms
    # entire algo behaves as if there are just num_assigned_arms in the bandit
    self.__pseudo_arms = [PseudoArm() for arm_id in self.__assigned_arms]
    # Parameters suggested by the paper
    self.__beta = 0.5
    self.__a = 1 + 10 / len(self.__assigned_arms)
    self.__eps = 0
    self.__delta = (1 - self.confidence) / 5
    # Total number of pulls used
    self.__total_pulls = 0
    self.__stage = 'initialization'

  def __confidence_radius(self, pulls: int) -> float:
    """
    Args:
      pulls: number of pulls

    Returns:
      confidence radius
    """
    if (1 + self.__eps) * pulls == 1:
      return math.inf
    return (1 + self.__beta) * (1 + math.sqrt(self.__eps)) * math.sqrt(
        2 * (1 + self.__eps) *
        math.log(math.log((1 + self.__eps) * pulls) / self.__delta) / pulls)

  def __ucb(self) -> np.ndarray:
    """
    Returns:
      upper confidence bound
    """
    return np.array([
        pseudo_arm.em_mean + self.__confidence_radius(pseudo_arm.total_pulls)
        for pseudo_arm in self.__pseudo_arms
    ])

  def actions(self, context=None) -> CollaborativeActions:
    if self.__stage == 'initialization':
      actions = CollaborativeActions() # default state is normal

      # 1 pull each for every assigned arm
      for arm_id in self.__assigned_arms:
        arm_pulls_pair = actions.arm_pulls_pairs.add()
        arm_pulls_pair.arm.id = arm_id
        arm_pulls_pair.pulls = 1
      return actions

    # self.__stage == 'main'
    actions = CollaborativeActions()

    for pseudo_arm in self.__pseudo_arms:
      if pseudo_arm.total_pulls >= (
          1 + self.__a * (self.__total_pulls - pseudo_arm.total_pulls)):
        return actions

    arm_pulls_pair = actions.arm_pulls_pairs.add()

    # map local arm index to the bandits arm index
    arm_pulls_pair.arm.id = self.__assigned_arms[int(np.argmax(self.__ucb()))]
    arm_pulls_pair.pulls = 1

    return actions

  def update(self, feedback: Feedback):
    for arm_rewards_pair in feedback.arm_rewards_pairs:
      # reverse map from bandit index to local index
      pseudo_arm_index = np.where(self.__assigned_arms==arm_rewards_pair.arm.id)[0][0]
      self.__pseudo_arms[pseudo_arm_index].update(
          np.array(arm_rewards_pair.rewards))
      self.__total_pulls += len(arm_rewards_pair.rewards)

    if self.__stage == 'initialization':
      self.__stage = 'main'

  def best_arm(self) -> int:
    # map best arm local index to actual bandit index
    return self.__assigned_arms[
      argmax_or_min_tuple([
        (pseudo_arm.total_pulls, arm_id)
        for (arm_id, pseudo_arm) in enumerate(self.__pseudo_arms)
      ])
    ]

  def get_total_pulls(self) -> int:
    return self.__total_pulls

from typing import Optional, List

import numpy as np

from banditpylib.data_pb2 import Context, Actions, Feedback
from .utils import LinearBanditLearner


class LinUCB(LinearBanditLearner):
  r"""Linear Upper Confidence Bound policy

  .. todo::
    Add algorithm description.

  :param List[np.ndarray] features: feature vector of each arm in a list
  :param float delta: delta
  :param float lambda_reg: lambda for regularization
  :param Optional[str] name: alias name
  """
  def __init__(self,
               features: List[np.ndarray],
               delta: float,
               lambda_reg: float,
               name: Optional[str] = None):
    super().__init__(arm_num=len(features), name=name)
    if delta <= 0 or delta >= 1:
      raise ValueError('Delta is expected within (0, 1). Got %.2f.' % delta)

    if lambda_reg <= 0:
      raise ValueError('lambda_reg is expected greater than 0. Got %.2f.' %
                       lambda_reg)

    self.__delta = delta
    self.__lambda_reg = lambda_reg
    self.__features = features
    self.__d = len(features[0])  # d: length of each feature
    self.__k = len(features)  # arm_nums

    # feature_matrix: d x k matrix of features stacked
    self.__feature_matrix = np.zeros((self.__d, self.__k))
    for i, feature in enumerate(features):
      self.__feature_matrix[:, i] = feature.reshape(-1)

  def _name(self) -> str:
    return 'linucb'

  def reset(self):
    self.__summation_AtXt = np.zeros(
        (self.__d, 1))  # summation_AtXt: accumulated sum of At * Xt, d x 1
    self.__Vt = self.__lambda_reg * np.eye(
        self.__d)  #Vt: V matrix at time t, d x d
    self.__theta_hat_t = np.random.normal(
        0, size=(self.__d,
                 1))  # theta_hat_t: the learners estimate of theta, d x 1

    # Current time step
    self.__time = 1

  def __LinUCB(self) -> np.ndarray:
    """Optimistic estimate of arms' real means

    Returns:
      optimistic estimate of arms' real means
    """
    root_beta_t = np.sqrt(
        self.__lambda_reg) + np.sqrt(2 * np.log(1 / self.__delta) + self.__d *
                                     np.log(1 + (self.__time - 1) /
                                            (self.__lambda_reg * self.__d)))
    ucb = self.__feature_matrix.T @ self.__theta_hat_t + root_beta_t *\
        np.sqrt((self.__feature_matrix.T @ np.linalg.pinv(self.__Vt) @
                 self.__feature_matrix).diagonal()).reshape(-1, 1)
    return ucb

  def actions(self, context: Context) -> Actions:
    del context

    actions = Actions()
    arm_pull = actions.arm_pulls.add()

    ucb = self.__LinUCB()
    arm_pull.arm.id = int(np.argmax(ucb, axis=0))

    arm_pull.times = 1
    return actions

  def update(self, feedback: Feedback):
    arm_feedback = feedback.arm_feedbacks[0]

    pulled_arm_index = arm_feedback.arm.id

    # Xt: reward observed at t
    Xt = np.array(arm_feedback.rewards)
    # At: feature of arm played at t
    At = self.__feature_matrix[:, pulled_arm_index].reshape(-1, 1)

    self.__Vt += (At @ At.T)
    self.__summation_AtXt += At * Xt
    self.__theta_hat_t = np.linalg.pinv(self.__Vt) @ self.__summation_AtXt

    self.__time += 1

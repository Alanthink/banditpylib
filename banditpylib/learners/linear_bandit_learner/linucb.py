from typing import List

import numpy as np

from banditpylib.arms import PseudoArm
from banditpylib.data_pb2 import Actions, Feedback
from banditpylib.learners import MaxReward, Goal
from .utils import LinearBanditLearner


class LinUCB(LinearBanditLearner):
  # incorrect description, to be filled later
  r"""Upper Confidence Bound policy :cite:`auer2002finite`

                At time :math:`t`, play arm

                .. math::
                                        \mathrm{argmax}_{i \in \{0, \dots, N-1\}} \left\{ \bar{\mu}_i(t) +
                                        \sqrt{ \frac{\alpha  \ln(t) }{T_i(t)} } \right\}
                """
  def __init__(self,
               features: List[np.ndarray],
               delta: float,
               lambda_reg: float,
               name: str = None):
    """Args:
                    name: alias name
                    features: feature vector of each arm in a list
                    delta: delta
                    lambda_reg: lambda for regularization
            """
    super().__init__(arm_num=len(features), name=name)
    if delta <= 0 or delta >= 1:
      raise ValueError('Delta is expected within (0, 1). Got %.2f.' % delta)

    if lambda_reg <= 0:
      raise ValueError('lambda_reg is expected greater than 0. Got %.2f.' %
                       lambda_reg)

    self.__delta = delta
    self.__lambda_reg = lambda_reg
    self.__features = features
    self.__d = len(features[0])
    self.__k = len(features)  # arm_nums

    self.__feature_matrix = np.zeros((self.__d, self.__k))
    for i in range(len(features)):
      self.__feature_matrix[:, i] = features[i].reshape(-1)

  def _name(self) -> str:
    return 'linucb'

  def reset(self):
    # variables used in the algorithm
    """
                                d: length of each feature
                                k: number of features/arms
                                feature_matrix: d x k matrix of features stacked
                                At: feature of arm played at t
                                Xt: reward observed at t
                                summation_AtXt: acuumulated sum of At * Xt, d x 1
                                Vt: V matrix at time t, d x d
                                theta_hat_t: the learners estimate of theta, d x 1
                                """
    self.__summation_AtXt = np.zeros((self.__d, 1))
    self.__Vt = self.__lambda_reg * np.eye(self.__d)
    self.__theta_hat_t = np.random.normal(0, size=(self.__d, 1))

    # Current time step
    self.__time = 1

  def __LinUCB(self) -> np.ndarray:
    """
                                Returns:
                                        optimistic estimate of arms' real means
                                """
    root_beta_t = np.sqrt(
        self.__lambda_reg) + np.sqrt(2 * np.log(1 / self.__delta) + self.__d *
                                     np.log(1 + (self.__time - 1) /
                                            (self.__lambda_reg * self.__d)))

    # unvectorized:
    # for i in range(len(self.__features)):
    #     a = self.__features[i]
    #     a = a.reshape(-1, 1)
    #     ucb[i] = np.dot(a.T, self.__theta_hat_t) +\
    #         root_beta_t * np.sqrt(a.T @ np.linalg.pinv(self.__Vt) @ a)

    # vectorized:
    ucb = self.__feature_matrix.T @ self.__theta_hat_t + root_beta_t *\
        np.sqrt((self.__feature_matrix.T @ np.linalg.pinv(self.__Vt) @
                 self.__feature_matrix).diagonal()).reshape(-1, 1)
    return ucb

  def actions(self, context=None) -> Actions:
    del context

    actions = Actions()
    arm_pulls_pair = actions.arm_pulls_pairs.add()

    ucb = self.__LinUCB()
    arm_pulls_pair.arm.id = int(np.argmax(ucb, axis=0))

    arm_pulls_pair.pulls = 1
    return actions

  def update(self, feedback: Feedback):
    arm_rewards_pair = feedback.arm_rewards_pairs[0]

    pulled_arm_index = arm_rewards_pair.arm.id
    Xt = np.array(arm_rewards_pair.rewards)

    At = self.__feature_matrix[:, pulled_arm_index].reshape(-1, 1)
    self.__Vt += (At @ At.T)
    self.__summation_AtXt += At * Xt
    self.__theta_hat_t = np.linalg.pinv(self.__Vt) @ self.__summation_AtXt

    self.__time += 1

  @property
  def goal(self) -> Goal:
    return MaxReward()

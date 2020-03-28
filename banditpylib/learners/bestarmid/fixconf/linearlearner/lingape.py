import numpy as np

from absl import flags

from .utils import LinearLearner, mat_norm

FLAGS = flags.FLAGS


class LinGapE(LinearLearner):
  """LinGapE policy :cite:`xu2017fully`.
  """

  def __init__(self, pars):
    super().__init__(pars)

  @property
  def _name(self):
    return 'LinGapE'

  def _learner_update(self, action, feedback):
    if isinstance(action, list):
      for (i, tup) in enumerate(action):
        self.__A = self.__A + np.outer(self._em_arms[tup[0]].feature,
                                       self._em_arms[tup[0]].feature)
        self.__b = self.__b + feedback[0][i] * self._em_arms[tup[0]].feature
    else:
      self.__A = self.__A + np.outer(self._em_arms[action].feature,
                                     self._em_arms[action].feature)
      self.__b = self.__b + feedback[0] * self._em_arms[action].feature
    self.__th = np.dot(np.linalg.inv(self.__A), self.__b)

  @property
  def __C_n(self):
    # eq 3 from the paper
    return 1 * \
           np.sqrt(2 * np.log(np.sqrt(np.linalg.det(self.__A)) /
                              (np.sqrt(self.__lambda) *
                               self.__delta))) + \
           np.sqrt(self.__lambda) * 1

  def __beta(self, i, j):
    x_diff = self._em_arms[i].feature - self._em_arms[j].feature
    return mat_norm(x_diff, np.linalg.inv(self.__A)) * self.__C_n

  def __Delta(self, i, j):
    x_diff = self._em_arms[i].feature - self._em_arms[j].feature
    return np.dot(x_diff, self.__th)

  def __select_direction(self):
    i_t = np.argmax([np.dot(a.feature, self.__th) for a in self._em_arms])
    gap_conf = [self.__Delta(j, i_t) +
                self.__beta(j, i_t)
                for j in range(self._arm_num)]
    j_t = np.argmax(gap_conf)
    B_t = max(gap_conf)
    return i_t, j_t, B_t

  def _learner_reset(self):
    # alg parameters suggested by the paper
    self.__lambda = 1.0
    self.__delta = self._fail_prob / 5
    self.__eps = 0.0
    # total number of pulls used
    self.__t = 0
    # dim of problem
    self.__d = self._bandit.features[0].shape[0]
    # confidence estimator
    self.__A = self.__lambda * np.eye(self.__d)
    self.__b = np.zeros(self.__d)
    self.__th = np.zeros(self.__d)

  def learner_round(self):
    # sample each arm once for the initialization step
    action = [(ind, 1) for ind in range(self._arm_num)]
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)
    self.__t = self._arm_num

    while True:
      i_t, j_t, B_t = self.__select_direction()
      if B_t <= self.__eps:
        return

      self.__t += 1
      x_diff = self._em_arms[i_t].feature - self._em_arms[j_t].feature
      greedy = [mat_norm(x_diff,
                         np.linalg.inv(self.__A + np.outer(a.feature,
                                                           a.feature)))
                for a in self._em_arms]
      action = np.argmin(greedy)
      feedback = self._bandit.feed(action)
      self._model_update(action, feedback)
      self._learner_update(action, feedback)

  def best_arm(self):
    i_t, _, _ = self.__select_direction()
    return int(i_t)

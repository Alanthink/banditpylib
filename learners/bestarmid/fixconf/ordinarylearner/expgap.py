import math

import copy
import numpy as np

from absl import flags

from arms import EmArm
from .utils import OrdinaryLearner

FLAGS = flags.FLAGS

__all__ = ['ExpGap']


class ExpGap(OrdinaryLearner):
  """Exponential-Gap Elimination"""

  def __init__(self):
    pass

  @property
  def name(self):
    return 'ExpGap'

  def _learner_init(self):
    pass

  def __median_elimination(self, active_arms, eps, log_delta):
    active_arms = copy.deepcopy(active_arms)
    ell = 1
    eps_ell = eps/4
    log_delta_ell = log_delta-math.log(2)
    eps_left = eps
    delta_left = math.exp(log_delta)
    while True:
      # threshold to be tuned
      if len(active_arms) <= 10:
        # uniform sampling
        t_ell = int(0.5/(eps_left**2)*(math.log(2/delta_left)))
        action = [(ind, t_ell) for ind in active_arms]
        feedback = self._bandit.feed(action)
        self._model_update(action, feedback)
        return max([(ind, self._em_arms[ind].em_mean)
            for ind in active_arms], key=lambda x:x[1])[0]

      t_ell = int(4/(eps_ell**2)*(math.log(3)-log_delta_ell))
      em_arms = [EmArm() for ind in range(self._arm_num)]
      action = [(ind, t_ell) for ind in active_arms]
      feedback = self._bandit.feed(action)
      for (i, tup) in enumerate(action):
        em_arms[tup[0]].update(feedback[0][i], tup[1])
      m_ell = np.median(np.array(
          [em_arms[ind].em_mean for ind in active_arms]))
      active_arms = [ind for ind in active_arms
          if em_arms[ind].em_mean >= m_ell]
      eps_left -= eps_ell
      delta_left -= math.exp(log_delta_ell)
      eps_ell *= 0.75
      log_delta_ell -= math.log(2)
      ell += 1

  def _learner_run(self):
    active_arms = list(range(self._arm_num))
    r = 1; eps_r = 0.25
    while len(active_arms) > 1:
      eps_r /= 2
      log_delta_r = math.log(self._fail_prob/50)-3*math.log(r)
      t_r = int(2/(eps_r**2)*(math.log(2)-log_delta_r))
      em_arms = [EmArm() for ind in range(self._arm_num)]
      action = [(ind, t_r) for ind in active_arms]
      feedback = self._bandit.feed(action)
      for (i, tup) in enumerate(action):
        em_arms[tup[0]].update(feedback[0][i], tup[1])
      best_arm_r = self.__median_elimination(active_arms, eps_r/2, log_delta_r)
      active_arms = [ind for ind in active_arms
          if em_arms[ind].em_mean >= em_arms[best_arm_r].em_mean-eps_r]
      r += 1

    self.__best_arm = active_arms[0]

  def _best_arm(self):
    return self.__best_arm

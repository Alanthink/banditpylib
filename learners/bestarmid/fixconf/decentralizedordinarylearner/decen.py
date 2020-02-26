import math

import numpy as np
import statistics

from .utils import DecentralizedOrdinaryLearner

__all__ = ['DECEN']


class DECEN(DecentralizedOrdinaryLearner):
  """Decentralized eliminiation"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'DECEN'

  def _learner_init(self):
    # alg parameters suggested by the paper
    self.__eps = 0.25
    self.__eta = 0.9
    self.__delta = 0.05
    self.__thresh = np.floor(self.__delta / self.__eta)
    # total number of pulls used
    self.__t = 0
    self.__K_global = range(self._arm_num)
    self.__K_local = range(self._arm_num)

    self.__l_global = 1
    self.__l_local = 1

    self.__message = []

  def broadcast_message(self, action, feedback):
    return [self.__K_global, self.__message]

  def __arm_selection(self, arm_set):
    # naive arm selection
    action = [(ind, int(np.ceil(1/((self.__eps/2)**2)*np.log(3/self.__delta))))
              for ind in range(self._arm_num)]
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)
    self.__t += np.ceil(1/((self.__eps/2)**2)*np.log(3/self.__delta))*self._arm_num

    med_arm_val = statistics.median([arm.em_mean for arm in self._em_arms])

    return [ind for (ind, arm) in enumerate(self._em_arms)
            if arm.em_mean < med_arm_val]

  def learner_choice(self, messages):
    """return an arm to pull"""

    # sample each arm once for the initialization step
    if self.__t == 0:
      action = [(ind, 1) for ind in range(self._arm_num)]
      feedback = self._bandit.feed(action)
      self._model_update(action, feedback)
      self.__t = self._arm_num

    self.__message = []
    messages = [list(m.values())[0] for m in messages]
    if len(messages) > 0:
      self.__K_global = messages[-1][0]
    K_local_hist = np.array([np.array(m[1]) for m in messages])

    if len(self.__K_global) > 0:
      sum_l = K_local_hist.sum(axis=0)
      arms_to_remove = sum_l[sum_l >= np.floor(np.log(self.__delta) /
                                               np.log(self.__eta))]
      arms_to_remove = np.argwhere(arms_to_remove > 0).flatten().tolist()

      self.__K_local = self.__K_global = [k for k in self.__K_global
                                          if k not in arms_to_remove]
    else:
      self.__K_local = self.__K_global

    K_local_hat = self.__arm_selection(self.__K_local)
    if len(K_local_hat) > 0:
      self.__l_local += 1
      self.__K_local = [k for k in self.__K_local if k not in K_local_hat]

      self.__message = [0 if k in K_local_hat
                        else 1 for k in range(self._arm_num)]

    em_means = np.array([self._em_arms[ind].em_mean for ind in self.__K_local])
    if (len(self.__K_local) <= 1) or 
       (np.all(np.isclose(em_means, em_means[0], 
                          rtol=1e-05, atol=1e-08))):
      return -1
    else:
      return self.best_arm()

  def learner_run(self):
    pass

  def best_arm(self):
    return max([(ind, self._em_arms[ind].em_mean)
                for ind in self.__K_local], key=lambda x: x[1])[0]

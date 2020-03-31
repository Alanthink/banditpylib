import math
import numpy as np

from .utils import OrdinaryLearner


class SH(OrdinaryLearner):
  """Sequential halving policy :cite:`karnin2013almost`

  .. inheritance-diagram:: SH
    :parts: 1
  """

  def __init__(self, pars):
    super().__init__(pars)
    self.__T0 = pars.get('T0', 2)

  @property
  def _name(self):
    return 'SH'

  def uniform_sampling(self, active_arms, budget_left):
    n = len(active_arms)
    rnd_array = np.random.multinomial(budget_left, np.ones(n)/n, size=1)[0]
    action = [(active_arms[i], rnd_array[i]) for i in range(n)]
    feedback = self._bandit.feed(action)
    self._model_update(action, feedback)
    self.__best_arm = max([
        (ind, self._em_arms[ind].em_mean) for ind in active_arms],
                          key=lambda x: x[1])[0]

  def _learner_reset(self):
    # calculate total number of rounds
    self.__r = math.ceil(math.log(self._arm_num, 2))

  def learner_round(self):
    # situation when budget is no greater than the number of arms
    if self._budget < self._arm_num:
      # randomly output an arm
      self.__best_arm = np.random.randint(self._arm_num)
      return
    if self._budget == self._arm_num:
      for ind in range(self._arm_num):
        feedback = self._bandit.feed(ind)
        self._model_update(ind, feedback)
      self.__best_arm = max([
          (ind, arm.em_mean) for (ind, arm) in enumerate(self._em_arms)],
                            key=lambda x: x[1])[0]
      return

    active_arms = list(range(self._arm_num))
    budget_left = self._budget
    for _ in range(self.__r):
      if len(active_arms) <= self.__T0:
        self.uniform_sampling(active_arms, budget_left)
        budget_left = 0
        return
      else:
        # pulls assigned to each arm
        t_r = math.floor(self._budget / (len(active_arms)*self.__r))
        if t_r > 0:
          action = [(ind, t_r) for ind in active_arms]
          feedback = self._bandit.feed(action)
          self._model_update(action, feedback)
          budget_left -= (t_r*len(active_arms))

        # sort according to empirical mean
        # past rewards are used
        active_arms.sort(key=lambda x: self._em_arms[x].em_mean, reverse=True)
        active_arms = active_arms[:math.ceil(len(active_arms)/2)]

    self.__best_arm = active_arms[0]
    return

  def best_arm(self):
    return self.__best_arm

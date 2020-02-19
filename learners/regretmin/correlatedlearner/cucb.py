import numpy as np
import cvxpy as cp

from .utils import CorrelatedLearner

__all__ = ['CUCB']


class CUCB(CorrelatedLearner):
  """CUCB"""

  def __init__(self, alpha=2):
    self.__alpha = alpha

  @property
  def name(self):
    return 'CUCB'

  def _learner_init(self):
    pass

  def learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    for k in range(self._arm_num):
      self._em_arms[k].action = self._bandit.actions[k]

    em_comp = list(range(self._arm_num)) # arm index
    # index  of most pulled arm
    k_max = np.argmax([arm.pulls for arm in self._em_arms])
    arm_kmax = self._em_arms[k_max]
    n_kmax = arm_kmax.pulls # number pulls for  most pulled arm
    mu_kmax = arm_kmax.em_mean # mean of most pulled arm
    dim = arm_kmax.action.shape[0] # action dimension

    th = cp.Variable(dim)
    # just want to check feasibility of constraints
    obj = cp.Minimize(cp.Constant(0))

    # recover confidence set & maximal mu over confidence set
    constr = [cp.abs(arm_kmax.action * th  - mu_kmax) <=
              np.sqrt(self.__alpha/n_kmax*np.log(self._t-1)),
              cp.norm(th, 2) <= 1]

    noncomp = []
    for k in em_comp:
      arm = self._em_arms[k]
      problem = cp.Problem(obj, constr+[arm.action * th >= a.action * th
                           for kk, a in enumerate(self._em_arms) if k != kk])
      problem.solve(warm_start=True)

      # if constraints not satisfied, not competitive arm
      if problem.value != 0.0:
        noncomp.append(k)

    # set  competitive  set & do UCB
    em_comp = [item for item in em_comp if item not in noncomp]
    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1))
           if i not in noncomp else float('-inf')
           for i, arm in enumerate(self._em_arms)]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass

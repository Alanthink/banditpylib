import numpy as np
import cvxpy as cp

from .utils import OrdinaryLearner

__all__ = ['CUCB']


class CUCB(OrdinaryLearner):
  """UCB"""

  def __init__(self, alpha=2):
    self.__alpha = alpha
    self.__th = cp.Variable(dim)
    self.__obj = cp.Minimize(cp.Constant(0)) # just want to check feasibility of constraints

  @property
  def name(self):
    return 'CUCB'

  def _learner_init(self):
    pass

  def _learner_choice(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    for k in range(self._arm_num):
      self._em_arms[k].action = self._bandit.arms[k].action


    em_comp = list(range(self._arm_num)) # arm index
    k_max = np.argmax([arm.pulls for arm in self._em_arms]) # index  of most pulled arm
    arm_kmax = self._em_arms[k_max]
    n_kmax = arm_kmax.pulls # number pulls for  most pulled arm
    mu_kmax = arm_kmax.em_mean # mean of most pulled arm
    dim = arm_kmax.action.shape[0] # action dimension

    # recover confidence set & maximal mu over confidence set
    constr = [cp.abs(arm_kmax.action * self.__th  - mu_kmax) <= np.sqrt(2*self.__alpha/n_kmax*np.log(self._t-1)), cp.norm(self.__th, 2) <= 1]

    noncomp = []
    for k in em_comp:
      arm = self._em_arms[k]
      problem = cp.Problem(__obj, constr+[arm.action * self.__th >= a.action * self.__th for kk,a in enumerate(self._em_arms) if k != kk])
      problem.solve(warm_start=True)
      if problem.value != 0.0:
        noncomp.append(k)

    em_comp = [item for item in em_comp if item not in noncomp]


    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1)) 
           for i, arm in enumerate(self._em_arms) if i in em_comp]
    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass

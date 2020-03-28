import numpy as np
import cvxpy as cp

from .utils import CorrelatedLearner


class CUCB(CorrelatedLearner):
  """CUCB policy :cite:`gupta2018exploiting`
  """

  def __init__(self, pars):
    super().__init__(pars)
    self.__alpha = float(pars['alpha']) if 'alpha' in pars else 2
    if self.__alpha <= 0:
      raise Exception('%s: alpha should be greater than 0!' % self.name)

  @property
  def _name(self):
    return 'CUCB'

  def _learner_reset(self):
    pass

  def learner_step(self, context):
    """return an arm to pull"""
    if self._t <= self._arm_num:
      return (self._t-1) % self._arm_num

    em_comp = list(range(self._arm_num)) # arm index
    # index of most pulled arm
    k_max = np.argmax([arm.pulls for arm in self._em_arms])
    arm_kmax = self._em_arms[k_max]
    n_kmax = arm_kmax.pulls # number pulls for  most pulled arm
    mu_kmax = arm_kmax.em_mean # mean of most pulled arm
    dim = arm_kmax.feature.shape[0] # feature dimension

    th = cp.Variable(dim)
    # just want to check feasibility of constraints
    obj = cp.Minimize(cp.Constant(0))

    # recover confidence set & maximal mu over confidence set
    constr = [cp.abs(arm_kmax.feature * th  - mu_kmax) <=
              np.sqrt(self.__alpha/n_kmax*np.log(self._t-1)),
              cp.norm(th, 2) <= 1]

    noncomp = []
    for k in em_comp:
      arm = self._em_arms[k]
      problem = cp.Problem(
          obj, constr+[arm.feature * th >= a.feature * th
                       for kk, a in enumerate(self._em_arms) if k != kk])
      problem.solve(warm_start=True)

      # if constraints not satisfied, not competitive arm
      if problem.value != 0.0:
        noncomp.append(k)

    # set competitive set & do UCB
    em_comp = [item for item in em_comp if item not in noncomp]
    ucb = [arm.em_mean+np.sqrt(self.__alpha/arm.pulls*np.log(self._t-1))
           if i not in noncomp else float('-inf')
           for i, arm in enumerate(self._em_arms)]

    return np.argmax(ucb)

  def _learner_update(self, context, action, feedback):
    pass

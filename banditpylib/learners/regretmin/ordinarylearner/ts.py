import numpy as np

from .utils import OrdinaryLearner

__all__ = ['TS']


class TS(OrdinaryLearner):
  """
  Thompson Sampling policy. Reward should be Bernoulli when Beta prior
  is chosen.

  At time ``self._t``, sample a virtual mean from the posterior distribution for
  every arm. Play the arm with the maximum sampled virtual mean.

  See :cite:`agrawal2017near` for more details.
  """

  def __init__(self, pars):
    """
    Args:
      ``pars``: a dictionary. Key ``'prior'`` has value ``'beta'`` or
       ``'gaussian'``. Default value is ``'beta'``.
    """
    super().__init__(pars)
    # set Beta prior by default
    self.__prior = pars.get('prior', 'beta')
    if self.__prior not in ['beta', 'gaussian']:
      raise Exception('Unknown prior!')

  @property
  def _name(self):
    return 'Thompson Sampling'

  def _learner_reset(self):
    pass

  def __beta_prior(self):
    # each arm has a uniform prior Beta(1, 1)
    vir_means = np.zeros(self._arm_num)
    for arm in range(self._arm_num):
      a = 1 + self._em_arms[arm].rewards
      b = 1 + self._em_arms[arm].pulls - self._em_arms[arm].rewards
      vir_means[arm] = np.random.beta(a, b)
    return np.argmax(vir_means)

  def __gaussian_prior(self):
    # each arm has a Gaussian prior N(0, 1)
    vir_means = np.zeros(self._arm_num)
    for arm in range(self._arm_num):
      mu = self._em_arms[arm].rewards / (self._em_arms[arm].pulls+1)
      sigma = 1.0 / (self._em_arms[arm].pulls+1)
      vir_means[arm] = np.random.normal(mu, sigma)
    return np.argmax(vir_means)

  def learner_step(self, context):
    """return an arm to play at time ``self._t``

    Return:
      an integer in [0, ``self._arm_num``)
    """
    if self.__prior == 'beta':
      return self.__beta_prior()
    return self.__gaussian_prior()

  def _learner_update(self, context, action, feedback):
    pass

r"""
Policies for ordinary multi-armed bandit with goal regret minimization.

We introduce notations in the following.

.. csv-table:: Notations

  :math:`T`, time horizon
  :math:`N`, total number of arms
  :math:`T_i(t)`,  number of times arm :math:`i` is played before time :math:`t`
  :math:`\hat{\mu}_i(t)`,  empirical mean of arm :math:`i` before time :math:`t`
  :math:`\hat{V}_i(t)`,empirical variance of arm :math:`i` before time :math:`t`
"""
from .eps_greedy import *
from .ucb import *
from .utils import *
from .ts import *
from .uniform import *
from .ucbv import *
from .moss import *

__all__ = [
    'OrdinaryLearner', 'EpsGreedy', 'UCB', 'ThompsonSampling', 'Uniform',
    'UCBV', 'MOSS'
]

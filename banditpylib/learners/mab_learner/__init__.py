r"""
Policies for ordinary multi-armed bandit with goal regret minimization.

We introduce notations in the following.

.. csv-table:: Notations

  :math:`T`, game horizon
  :math:`N`, total number of arms
  :math:`i_t`, arm pulled at time :math:`t`
  :math:`X_i^t`, "| empirical reward of arm :math:`i` at time :math:`t`
                  if arm :math:`i` is pulled"
  :math:`T_i(t)`,  number of times arm :math:`i` is played before time :math:`t`
  :math:`\bar{\mu}_i(t)`,  empirical mean of arm :math:`i` before time :math:`t`
  :math:`\bar{V}_i(t)`,empirical variance of arm :math:`i` before time :math:`t`
"""
from .eps_greedy import *
from .ucb import *
from .utils import *
from .ts import *
from .uniform import *
from .ucbv import *
from .moss import *
from .exp3 import *
from .explore_then_commit import *
from .softmax import *

__all__ = [
    'MABLearner', 'EpsGreedy', 'UCB', 'ThompsonSampling', 'Uniform', 'UCBV',
    'MOSS', 'EXP3', 'ExploreThenCommit', 'Softmax'
]

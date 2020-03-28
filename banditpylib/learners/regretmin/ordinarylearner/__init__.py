r"""
Policies for ordinary multi-armed bandit.

We introduce notations in the following.

.. csv-table::
  :header: notation, meaning

  :math:`T`, time horizon
  :math:`T_i(t)`,  number of times arm :math:`i` is played before time :math:`t`
  :math:`\hat{\mu}_i(t)`,  empirical mean of arm :math:`i` before time :math:`t`
  :math:`\hat{V}_i(t)`,empirical variance of arm :math:`i` before time :math:`t`
"""
from .utils import *
from .epsgreedy import *
from .moss import *
from .ts import *
from .ucb import *
from .ucbv import *
from .uniform import *

__all__ = [
    'OrdinaryLearner',
    'EpsGreedy',
    'MOSS',
    'TS',
    'UCB',
    'UCBV',
    'Uniform'
]

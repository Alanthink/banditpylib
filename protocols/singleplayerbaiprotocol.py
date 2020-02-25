from absl import flags
from absl import logging

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['SinglePlayerBAIProtocol']


class SinglePlayerBAIProtocol(Protocol):
  """single player best arm identification protocol
  """

  @property
  def type(self):
    return 'SinglePlayerBAIProtocol'

  @property
  def _budget(self):
    return self.__budget

  @property
  def _fail_prob(self):
    return self.__fail_prob

  def _one_trial(self, seed):
    np.random.seed(seed)

    if self._player.goal == 'FixedBudgetBAI':
      results = []
      for budget in self._pars['budgets']:
        self.__budget = budget

        ########################################################################
        # initialization
        self._bandit.init()
        self._player.init(self._bandit, self.__budget)
        ########################################################################

        self._player.learner_run()
        if self._bandit.tot_samples > budget:
          logging.fatal('%s uses more than the given budget!'
              % self._player.name)

        regret = self._bandit.best_arm_regret(self._player.best_arm())
        results.append(dict({self._player.name: [budget, regret]}))
      return results

    #  FixedConfidenceBAI
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__fail_prob = fail_prob

      ##########################################################################
      # initialization
      self._bandit.init()
      self._player.init(self._bandit, self.__fail_prob)
      ##########################################################################

      self._player.learner_run()
      regret = self._bandit.best_arm_regret(self._player.best_arm())
      results.append(
          dict({self._player.name:
                [fail_prob, self._bandit.tot_samples, regret]}))
    return results

from absl import flags
from absl import logging

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS

__all__ = ['SinglePlayerPEProtocol']


class SinglePlayerPEProtocol(Protocol):
  """single player pure exploration protocol
  """

  @property
  def type(self):
    return 'SinglePlayerPEProtocol'

  def __one_trial_fixbudget(self):
    results = []
    for budget in self._pars['budgets']:
      self.__budget = budget

      ##########################################################################
      # initialization
      self._bandit.init()
      self._player.init(self._bandit, self.__budget)
      ##########################################################################

      self._player.learner_run()
      if self._bandit.tot_samples > budget:
        logging.fatal(
            '%s uses more than the given budget!' % self._player.name)
      regret = getattr(
          self._bandit,
          '_'+type(self._bandit).__name__+'__'+self._regret_def)(
              self._player.rewards_def())
      results.append(dict({self._player.name: [budget, regret]}))
    return results

  def __one_trial_fixconf(self):
    results = []
    for fail_prob in self._pars['fail_probs']:
      self.__fail_prob = fail_prob

      ##########################################################################
      # initialization
      self._bandit.init()
      self._player.init(self._bandit, self.__fail_prob)
      ##########################################################################

      self._player.learner_run()
      regret = getattr(
          self._bandit,
          '_'+type(self._bandit).__name__+'__'+self._regret_def)(
              self._player.rewards_def())
      results.append(
          dict({self._player.name:
                [fail_prob, self._bandit.tot_samples, regret]}))
    return results

  def _one_trial(self, seed):
    np.random.seed(seed)
    if 'FixBudget' in self._player.goal:
      return self.__one_trial_fixbudget()
    return self.__one_trial_fixconf()

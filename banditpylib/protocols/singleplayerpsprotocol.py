from absl import flags

import numpy as np

from .utils import Protocol

FLAGS = flags.FLAGS


class SinglePlayerPEProtocol(Protocol):
  """Single player pure exploration protocol
  """

  def __init__(self, pars=None):
    """
    Args:
      pars:
        ``"fail_probs"`` ([float,], optional): fail probabilities.
        ``"budgets"`` ([int,], optional): budgets.
        ``"trials"`` (int): number of repetitions of the game.
        ``"processors"`` (int): maximum number of processors can be used. -1
        means trying to make use all available cpus.
    """

  @property
  def type(self):
    return 'SinglePlayerPEProtocol'

  def __one_trial_fixbudget(self, stop_cond):
    budget = stop_cond

    ############################################################################
    # initialization
    self._bandit.reset()
    self._player.reset(self._bandit, budget)
    ############################################################################

    self._player.learner_round()

    if self._bandit.tot_samples > budget:
      raise Exception(
          '%s uses more than the given budget!' % self._player.name)
    regret = getattr(self._bandit, self._regret_funcname)(
        getattr(self._player, self._rewards_funcname)())
    return dict({self._player.name: [budget, regret]})

  def __one_trial_fixconf(self, stop_cond):
    fail_prob = stop_cond

    ############################################################################
    # initialization
    self._bandit.reset()
    self._player.reset(self._bandit, fail_prob)
    ############################################################################

    self._player.learner_round()

    regret = getattr(self._bandit, self._regret_funcname)(
        getattr(self._player, self._rewards_funcname)())
    return dict(
        {self._player.name: [fail_prob, self._bandit.tot_samples, regret]})

  def _one_trial(self, seed, stop_cond):
    np.random.seed(seed)
    if 'Fix Budget' in self._player.goal:
      return self.__one_trial_fixbudget(stop_cond)
    return self.__one_trial_fixconf(stop_cond)

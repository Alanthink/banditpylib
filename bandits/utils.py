from abc import ABC, abstractmethod

import numpy as np

__all__ = ['Bandit', 'search', 'search_best_assortment']


class Bandit(ABC):
  """Abstract bandit environment"""

  @property
  @abstractmethod
  def type(self):
    pass

  # current state of the environment
  @property
  @abstractmethod
  def context(self):
    pass

  # full state of the environment (can not be fetched from outside)
  @property
  @abstractmethod
  def _oracle_context(self):
    pass

  @abstractmethod
  def init(self):
    pass

  @abstractmethod
  def _update_context(self):
    pass

  @abstractmethod
  def _take_action(self, action):
    """
    Input:
      action: an integer or a list of two-tuples; if it is a list of two-tuples,
      in each two-tuple, the first item is the arm index; the second item is
      the number of actions to be taken; if it is an integer, it is assumed the
      arm index and will be pulled one time.
    Output:
      reward or a list of rewards
    """

  @abstractmethod
  def regret(self, rewards):
    pass

  def feed(self, action):
    """
    Output:
      feedback: a tuple and feedback[0] denotes the reward
    """
    feedback = self._take_action(action)
    self._update_context()
    return feedback


def search(subsets, n, i, path, K=np.inf):
  if i == n:
    if path:
      subsets.append(path)
    return
  if len(path) < K:
    search(subsets, n, i+1, path+[i], K)
  search(subsets, n, i+1, path, K)


def search_best_assortment(abspar, revenue, K=np.inf):
  # non-purchase is assumed to have abstraction par 1
  subsets = []
  search(subsets, len(abspar), 1, [], K)
  sorted_assort = sorted( [ (sum([abspar[prod]/
      (sum([abspar[prod] for prod in subset])+1)*revenue[prod]
      for prod in subset]), subset) for subset in subsets], key=lambda x:x[0] )
  return sorted_assort[-1]

import numpy as np

from .pseudo_arm import PseudoArm

class PseudoCategArm(PseudoArm):
  """Pseudo categorical arm

  This class is used to store empirical information of an arm. Currently, In addition to
  the information regarding the first and second moments of empirical rewards
  i.e., mean and variance are stored, we also store the frequency of each category.
  """
  def __init__(self, prior=None, name: str = None):
    """
    Args:
      prior: defualt value [1,1] is the uniform prior distribution on [0,1]
      name: alias name
    """
    self.__name = 'pseudo_categ_arm' if name is None else name
    if prior is None:
      self.__prior = [1, 1]
    else:
      self.__prior = prior
    self.__n_categ = len(self.__prior)
    self.reset()

  def reset(self):
    """Clear information"""
    self.__total_pulls = 0
    self.__total_rewards = 0
    self.__sum_of_square_reward = 0
    self.__freq = self.__prior

  @property
  def freq(self):
    return self.__freq

  @property
  def sorted_freq(self):
    return sorted(self.__freq)

  @property
  def top_categ(self) -> int:
    """
    Return the index of the category with the highest frequency.
    If there are more than one, return the one with the smallest index.
    """
    return self.__freq.index(max(self.__freq))

  def update(self, categories: np.ndarray):
    """Update information

    Args:
      rewards: empirical rewards
    """
    self.__total_pulls += len(categories)
    self.__total_rewards += sum(categories)
    self.__sum_of_square_reward += sum(categories**2)
    for c in categories:
      self.__freq[c] += 1

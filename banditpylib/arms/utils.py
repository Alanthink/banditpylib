from abc import ABC, abstractmethod

import numpy as np


class Arm(ABC):
  """Arm class"""

  @abstractmethod
  def pull(self, pulls=1) -> np.ndarray:
    """
    Args:
      pulls: number of times to pull

    Return:
      rewards
    """

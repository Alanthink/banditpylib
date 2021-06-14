from typing import List, Tuple

import numpy as np


def argmax_or_min(values: List[float], find_min: bool = False) -> int:
  """Find index with the largest or smallest value

  Args:
    values: a list of values
    find_min: whether to select smallest value

  Returns:
    index with the largest or smallest value. When there is a tie, randomly
    output one of the indexes.
  """
  extremum = min(values) if find_min else max(values)
  indexes = [index for index, value in enumerate(values) if value == extremum]
  return np.random.choice(indexes)


def argmax_or_min_tuple(values: List[Tuple[float, int]],
                        find_min: bool = False) -> int:
  """Find the second element of the tuple with the largest or smallest value

  Args:
    values: a list of tuples
    find_min: whether to select smallest value

  Returns:
    the second element of the tuple with the largest or smallest value.
    When there is a tie, randomly output one of them.
  """
  extremum = min([value for value, _ in values]) if find_min else max(
      [value for value, _ in values])
  indexes = [index for (value, index) in values if value == extremum]
  return np.random.choice(indexes)

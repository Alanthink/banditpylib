from typing import List, Tuple

import numpy as np
import pandas as pd

import google.protobuf.json_format as json_format
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore

from banditpylib.data_pb2 import Trial


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


def parse_trials_from_bytes(data: bytes) -> List[Trial]:
  """Parse trials from bytes

  Args:
    data: bytes data

  Returns:
    trial protobuf messages
  """
  trials = []
  next_pos, pos = 0, 0
  while pos < len(data):
    trial = Trial()
    next_pos, pos = _DecodeVarint32(data, pos)
    trial.ParseFromString(data[pos:pos + next_pos])

    # Proceed to next message
    pos += next_pos
    trials.append(trial)
  return trials


def trials_to_dataframe(filename: str) -> pd.DataFrame:
  """Read bytes file storing trials and transform to pandas DataFrame

  Args:
    filename: file name

  Returns:
    pandas dataframe
  """
  with open(filename, 'rb') as f:
    data = []
    trials = parse_trials_from_bytes(f.read())
    for trial in trials:
      for result in trial.results:
        tmp_dict = json_format.MessageToDict(
            result,
            including_default_value_fields=True,
            preserving_proto_field_name=True)
        tmp_dict['bandit'] = trial.bandit
        tmp_dict['learner'] = trial.learner
        data.append(tmp_dict)
    data_df = pd.DataFrame.from_dict(data)
    return data_df

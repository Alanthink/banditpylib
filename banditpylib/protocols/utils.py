import multiprocessing
from multiprocessing import Pool
import time
from typing import List

from abc import ABC, abstractmethod
from absl import logging
import pandas as pd

import google.protobuf.json_format as json_format
from google.protobuf.internal.decoder import _DecodeVarint32  # type: ignore
from google.protobuf.internal.encoder import _VarintBytes  # type: ignore

from banditpylib.bandits import Bandit
from banditpylib.data_pb2 import Trial
from banditpylib.learners import Learner


def time_seed() -> int:
  """Generate random seed using current time

  Returns:
    random seed
  """
  tem_time = time.time()
  return int((tem_time - int(tem_time)) * 10000000)


def parse_trials_data(data: bytes) -> List[Trial]:
  """Retrieve trials data from bytes data"""
  trials_data = []
  next_pos, pos = 0, 0
  while pos < len(data):
    trial = Trial()
    next_pos, pos = _DecodeVarint32(data, pos)
    trial.ParseFromString(data[pos:pos + next_pos])

    # Proceed to next message
    pos += next_pos
    trials_data.append(trial)
  return trials_data


def trial_data_messages_to_dict(filename: str) -> pd.DataFrame:
  """Read file storing trials data and transform to pandas DataFrame"""
  with open(filename, 'rb') as f:
    data = []
    trials_data = parse_trials_data(f.read())
    for trial in trials_data:
      for data_item in trial.data_items:
        tmp_dict = json_format.MessageToDict(
            data_item,
            including_default_value_fields=True,
            preserving_proto_field_name=True)
        tmp_dict['bandit'] = trial.bandit
        tmp_dict['learner'] = trial.learner
        data.append(tmp_dict)
    data_df = pd.DataFrame.from_dict(data)
    return data_df


class Protocol(ABC):
  """Abstract class for a protocol which is used to coordinate the interactions
  between the learner and the bandit environment.
  """
  def __init__(self, bandit: Bandit, learners: List[Learner]):
    """
    Args:
      bandit: bandit environment
      learners: learners used to run simulations
    """
    for learner in learners:
      if not isinstance(learner.running_environment, List):
        if not isinstance(bandit, learner.running_environment):
          raise Exception('Learner %s can not recognize environment %s.' %
                          (learner.name, bandit.name))
      else:
        # learner.running_environment is a list
        environment_is_recognizable = False
        for env in learner.running_environment:
          if isinstance(bandit, env):
            environment_is_recognizable = True

        if not environment_is_recognizable:
          raise Exception('Learner %s can not recognize environment %s.' %
                          (learner.name, bandit.name))

    self.__bandit = bandit
    self.__learners = learners
    # The learner simulated currently
    self.__current_learner: Learner = None

  @property
  @abstractmethod
  def name(self) -> str:
    """Protocol name"""

  @property
  def bandit(self) -> Bandit:
    """Bandit environment the simulator is using the learners to play with"""
    return self.__bandit

  @property
  def current_learner(self) -> Learner:
    """The learner used by the simulator currently"""
    return self.__current_learner

  @abstractmethod
  def _one_trial(self, random_seed: int, debug: bool) -> bytes:
    """One trial of the game

    This method defines how to run one trial of the game.

    Args:
      random_seed: random seed
      debug: whether to run the trial in debug mode

    Returns:
      one trial data
    """

  def __write_to_file(self, data: bytes):
    """Write the result of one trial to file

    Args:
      data: one trial data
    """
    with open(self.__output_filename, 'ab') as f:
      f.write(_VarintBytes(len(data)))
      f.write(data)
      f.flush()

  def play(self, trials: int, output_filename: str, processes=-1, debug=False):
    """Start playing the game

    Args:
      trials: number of repetitions
      output_filename: file used to dump the results
      processes: maximum number of processes to run. -1 means no limit
      debug: debug mode. When it is set to `True`, `trials` will be
        automatically set to 1 and debug information of the trial will be
        printed out.

    .. warning::
      By default, `output_filename` will be opened with mode `a`.
    """
    if debug:
      trials = 1

    for learner in self.__learners:
      # Set current learner
      self.__current_learner = learner

      logging.info('start %s\'s play with %s', self.__current_learner.name,
                   self.__bandit.name)

      start_time = time.time()
      self.__output_filename = output_filename
      pool = Pool(processes=(
          multiprocessing.cpu_count() if processes < 0 else processes))

      trial_results = []
      for _ in range(trials):
        result = pool.apply_async(self._one_trial,
                                  args=[time_seed(), debug],
                                  callback=self.__write_to_file)

        trial_results.append(result)

      # Can not apply for processes any more
      pool.close()
      pool.join()

      # Check if there are exceptions during the trials
      for result in trial_results:
        result.get()

      logging.info('%s\'s play with %s runs %.2f seconds.',
                   self.__current_learner.name, self.__bandit.name,
                   time.time() - start_time)

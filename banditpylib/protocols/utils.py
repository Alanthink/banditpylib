import multiprocessing
from multiprocessing import Pool
import time
from typing import List

from abc import ABC, abstractmethod
from absl import logging

from google.protobuf.internal.encoder import _VarintBytes  # type: ignore

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner


def time_seed() -> int:
  """Generate random seed using current time

  Returns:
    random seed
  """
  tem_time = time.time()
  return int((tem_time - int(tem_time)) * 10000000)


class Protocol(ABC):
  """Abstract class for a protocol which is used to coordinate the interactions
  between the learner and the bandit environment.

  :param Bandit bandit: bandit environment
  :param List[Learner] learners: learners used to run simulations
  """
  def __init__(self, bandit: Bandit, learners: List[Learner]):
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
      output_filename: name of the file used to dump the results
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

import json
import multiprocessing
from multiprocessing import Pool
import time
from typing import List, Dict

from abc import ABC, abstractmethod
from absl import logging

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
  """
  Abstract class for the protocol which is used to coordinate the interactions
  between the learner and the bandit environment.

  For each setup, just call :func:`play` to start the game. A protocol should
  implement :func:`_one_trial` method to define how to run one trial of the
  game.
  """
  def __init__(self, bandit: Bandit, learners: List[Learner]):
    """
    Args:
      bandit: bandit environment
      learners: learners to be compared with
    """
    for learner in learners:
      if not isinstance(bandit, learner.running_environment):
        raise Exception('Learner %s can not recognize environment %s!' %
                        (learner.name, bandit.name))
    self.__bandit = bandit
    self.__learners = learners
    # learner the simulator is currently running
    self.__current_learner = None

  @property
  @abstractmethod
  def name(self) -> str:
    """protocol name"""

  @property
  def bandit(self) -> Bandit:
    """bandit environment the simulator is running in"""
    return self.__bandit

  @property
  def current_learner(self) -> Learner:
    """current learner the simulator is using"""
    return self.__current_learner

  @abstractmethod
  def _one_trial(self, random_seed: int) -> Dict or List[Dict]:
    """One trial of the game

    Args:
      random_seed: random seed

    Returns:
      result of one trial
    """

  def __write_to_file(self, data: Dict or List[Dict]):
    """Write the result of one trial to file

    Args:
      data: result of one trial
    """
    with open(self.__output_filename, 'a') as f:
      if isinstance(data, list):
        for data_point in data:
          json.dump(data_point, f)
          f.write('\n')
      else:
        json.dump(data, f)
        f.write('\n')
      f.flush()

  def play(self, trials: int, output_filename: str, processes=-1, debug=False):
    """Start playing the game

    Args:
      trials: number of repetitions
      output_filename: file used to dump the results
      processes: maximum number of processes to run. -1 means no limit
      debug: debug mode. When it is set to `True`, the game will stop if there
        is any error in subprocesses. Otherwise, the errors of subprocesses will
        be silenced.

    .. warning::
      By default, `output_filename` will be opened with mode `a`.
    """
    for learner in self.__learners:
      # set current learner
      self.__current_learner = learner

      start_time = time.time()
      self.__output_filename = output_filename
      pool = Pool(processes=(
          multiprocessing.cpu_count() if processes < 0 else processes))

      for _ in range(trials):
        results = pool.apply_async(self._one_trial,
                                   args=[time_seed()],
                                   callback=self.__write_to_file)

        # for debugging purpose
        if debug:
          results.get()

      # can not apply for processes any more
      pool.close()
      pool.join()
      logging.info('%s\'s play with %s runs %.2f seconds.',
                   self.__current_learner.name, self.__bandit.name,
                   time.time() - start_time)

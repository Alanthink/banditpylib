import json
import multiprocessing
from multiprocessing import Pool
import time
from typing import Dict

from abc import ABC, abstractmethod

from banditpylib.bandits import Bandit
from banditpylib.learners import Learner


def time_seed() -> int:
  """Generate random seed using current time

  Return:
    int: random seed
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
  def __init__(self, bandit: Bandit, learner: Learner):
    """
    Args:
      bandit: bandit environment
      learner: learner
    """
    self.__bandit = bandit
    self.__learner = learner

  @property
  @abstractmethod
  def name(self) -> str:
    """
    Return:
      protocol name
    """

  @abstractmethod
  def _one_trial(self, bandit: Bandit, learner: Learner,
                 random_seed: int) -> Dict:
    """One trial of the game

    Args:
      bandit: bandit environment
      learner: learner
      random_seed: random seed

    Return:
      result in dictionary
    """

  def __write_to_file(self, data: Dict):
    """Write the result of one trial to file

    Args:
      data: the result of one trial
    """
    with open(self.__output_filename, 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def play(self, trials: int, output_filename: str, processes=-1, debug=False):
    """Start playing the game

    Args:
      trials: number of repetitions
      output_filename: file used to dump the results
      processes: maximum number of processes to run. -1 means no limit
      debug: play the game in debugging mode
    """
    self.__output_filename = output_filename
    pool = Pool(processes=(
        multiprocessing.cpu_count() if processes < 0 else processes))

    for _ in range(trials):
      results = pool.apply_async(self._one_trial,
                                 args=(
                                     # bandit
                                     self.__bandit,
                                     # learner
                                     self.__learner,
                                     # random_seed
                                     time_seed(),
                                 ),
                                 callback=self.__write_to_file)

      # for debugging purpose
      if debug:
        results.get()

    # can not apply for processes any more
    pool.close()
    pool.join()

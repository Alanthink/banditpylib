"""
Abstract class of the protocol which is used to coordinate the interactions
between the learner and the specified bandit environment.

For each setup, just call `play` to start simulation. A protocol should
implement `_one_trial` method to define how to run one trial of experiment.
"""
import json
import time
import multiprocessing
from multiprocessing import Pool

from abc import ABC, abstractmethod

from absl import flags
from absl import logging

FLAGS = flags.FLAGS

__all__ = ['Protocol', 'current_time']


def get_funcname_of_regret_rewards(goal):
  if goal == 'Best Arm Identification':
    return ('best_arm_regret', 'best_arm')
  elif goal == 'Regret Minimization':
    return ('regret', 'rewards')
  else:
    raise Exception('Unknown goal!')


# for generating random seeds
def current_time():
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """abstract protocol class"""

  @property
  @abstractmethod
  def type(self):
    """type of the protocol"""

  @property
  def __trials(self):
    """# of trials of the experiment"""
    return self._pars['trials']

  @property
  def __processors(self):
    """maximum number of processors can be used"""
    if 'processors' in self._pars and self._pars['processors'] >= 1:
      return int(self._pars['processors'])
    return multiprocessing.cpu_count()

  @abstractmethod
  def _one_trial(self, seed):
    """one trial experiment"""

  def __write_to_file(self, data):
    """write the result of one trial to file

    input:
      data: the result of one trial
    """
    with open(self.__output_file, 'a') as f:
      if isinstance(data, list):
        for item in data:
          json.dump(item, f)
          f.write('\n')
      else:
        json.dump(data, f)
        f.write('\n')
      f.flush()

  def __multi_proc(self):
    """multi-processes helper"""
    pool = Pool(processes=self.__processors)

    for _ in range(self.__trials):
      result = pool.apply_async(
          self._one_trial, args=(current_time(), ),
          callback=self.__write_to_file)

      if FLAGS.debug:
        # for debugging purposes
        # to make sure error info of subprocesses will be reported
        # this flag could heavily increase the running time
        result.get()

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandit, learner, running_pars, output_file):
    """
    input:
      bandit: bandit of the setup
      learner: learner of the setup
      running_pars: parameters used for this experiment
      output_file: file to store the results of the experiment
    """
    if isinstance(bandit, list):
      # multiple players
      self._bandits = bandit
      self._players = learner
      logging.info(
          'run %s using protocol %s' %
          (self._players[0].name, self.type))
      self._regret_funcname, self._rewards_funcname = \
          get_funcname_of_regret_rewards(self._players[0].goal)
    else:
      # single player
      self._bandit = bandit
      self._player = learner
      logging.info(
          'run %s with protocol %s' % (learner.name, self.type))
      self._regret_funcname, self._rewards_funcname = \
          get_funcname_of_regret_rewards(self._player.goal)

    self.__output_file = output_file
    self._pars = running_pars

    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

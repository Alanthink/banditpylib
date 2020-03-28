import json
import time
import multiprocessing
from multiprocessing import Pool

from abc import ABC, abstractmethod

from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def get_funcname_of_regret_rewards(goal):
  if 'Best Arm Identification' in goal:
    return ('best_arm_regret', 'best_arm')
  elif goal == 'Regret Minimization':
    return ('regret', 'rewards')
  else:
    raise Exception('Unknown goal!')


def get_stop_cond(goal):
  if goal == 'Fix Budget Best Arm Identification':
    return 'budgets'
  elif goal == 'Fix Confidence Best Arm Identification':
    return 'fail_probs'
  elif goal == 'Regret Minimization':
    return 'horizons'
  else:
    raise Exception('Unknown goal!')


def current_time():
  """Generate random seed using current time

  Return:
    int: random seed
  """
  tem_time = time.time()
  return int((tem_time-int(tem_time))*10000000)


class Protocol(ABC):
  """
  Abstract class for the protocol which is used to coordinate the interactions
  between the learner and the environment.

  For each setup, just call ``play`` to start the game. A protocol should
  implement ``_one_trial`` method to define how to run one trial of the game.
  """

  @property
  @abstractmethod
  def type(self):
    """Type of the protocol"""

  @property
  def __trials(self):
    """Number of trials of the game"""
    return self._pars['trials']

  @property
  def __processors(self):
    """Maximum number of processors can be used"""
    if 'processors' in self._pars and self._pars['processors'] >= 1:
      return int(self._pars['processors'])
    return multiprocessing.cpu_count()

  @abstractmethod
  def _one_trial(self, seed, stop_cond):
    """One trial of the game"""

  def __write_to_file(self, data):
    """Write the result of one trial to file

    Args:
      data (dict): the result of one trial
    """
    with open(self.__output_file, 'a') as f:
      json.dump(data, f)
      f.write('\n')
      f.flush()

  def __multi_proc(self, stop_cond):
    """Multi-processes helper"""
    pool = Pool(processes=self.__processors)

    for _ in range(self.__trials):
      result = pool.apply_async(
          self._one_trial, args=(current_time(), stop_cond, ),
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
    """Start the game

    Args:
      bandit (object or [object,]): environment
      learner (object or [object,]): learner
      running_pars (dict): running parameters
      output_file (str): file used to store the results
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
      self.__stop_cond = running_pars[get_stop_cond(self._players[0].goal)]
    else:
      # single player
      self._bandit = bandit
      self._player = learner
      logging.info(
          'run %s with protocol %s' % (learner.name, self.type))
      self._regret_funcname, self._rewards_funcname = \
          get_funcname_of_regret_rewards(self._player.goal)
      self.__stop_cond = running_pars[get_stop_cond(self._player.goal)]

    self.__output_file = output_file
    self._pars = running_pars

    start_time = time.time()
    for stop_cond in self.__stop_cond:
      self.__multi_proc(stop_cond)
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

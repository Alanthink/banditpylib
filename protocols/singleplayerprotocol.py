import json
import time
from multiprocessing import Pool

from absl import flags
from absl import logging

import numpy as np

from bandits import Bandit
from .utils import Protocol, current_time

FLAGS = flags.FLAGS

__all__ = ['SinglePlayerProtocol']


class SinglePlayerProtocol(Protocol):
  """Single Player Protocol
  """

  @property
  def type(self):
    return 'SinglePlayerProtocol'

  @property
  def __horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  @property
  def __trials(self):
    # # of repetitions of the play
    return self._pars['trials']

  @property
  def __processors(self):
    # maximum number of processors can be used
    return self._pars['processors']

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    self._bandit.init()
    self._player.init(self._bandit, self.__horizon)
    ############################################################################

    agg_regret = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        # simulation starts from t = 1
        context = self._bandit.context
        action = self._player.learner_choice(context)
        feedback = self._bandit.feed(action)
        self._player.update(context, action, feedback)
      if t % self.__frequency == 0:
        agg_regret[t] = self._bandit.regret(self._player.rewards)
    return dict({self._player.name: agg_regret})

  def __write_to_file(self, data):
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
    pool = Pool(processes = self.__processors)

    for _ in range(self.__trials):
      result = pool.apply_async(self._one_trial, args=(current_time(), ),
          callback=self.__write_to_file)
      if FLAGS.debug:
        # for debugging purposes
        # to make sure error info of subprocesses will be reported
        # this flag could heavily increase the running time
        result.get()

    # can not apply for processes any more
    pool.close()
    pool.join()

  def play(self, bandit, player, output_file, pars):
    if not isinstance(bandit, Bandit):
      logging.fatal('Not a legimate bandit!')

    self._bandit = bandit
    self._player = player
    self.__output_file = output_file
    self._pars = pars

    logging.info('run learner %s with goal %s under protocol %s' %
        (player.name, player.goal, self.type))
    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time()-start_time))

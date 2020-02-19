import json
import time
from random import randint
from multiprocessing import Pool

from absl import flags
from absl import logging
import numpy as np

from bandits import Bandit
from .utils import Protocol, current_time

FLAGS = flags.FLAGS

__all__ = ['DecentralizedRegretMinProtocol']


class DecentralizedRegretMinProtocol(Protocol):
  """Decentralized Regret Minimization Protocol
  """

  def __init__(self, pars):
    self.__messages = []
    self.__num_players = pars['num_players']

  @property
  def type(self):
    return 'DecentralizedRegretMinProtocol'

  @property
  def _num_players(self):
    return self.__num_players

  @property
  def __trials(self):
    # # of repetitions of the play
    return self._pars['trials']

  @property
  def __processors(self):
    # maximum number of processors can be used
    return self._pars['processors']

  @property
  def __horizon(self):
    return self._pars['horizon']

  @property
  def __frequency(self):
    # frequency to record intermediate regret results
    return self._pars['freq']

  def _one_round(self):
    # sample player
    k = randint(0, self._num_players-1)
    player = self._players[k]
    bandit = self._bandits[k]

    # play player and broadcast message
    context = bandit.context
    action = player.learner_choice(context, self.__messages)
    feedback = bandit.feed(action)
    player.update(context, action, feedback)
    message = player._broadcast_message(context, action, feedback)
    self.__messages.append({player.name: message})

  def _one_trial(self, seed):
    np.random.seed(seed)

    ############################################################################
    # initialization
    for k in range(self._num_players):
      player = self._players[k]
      bandit = self._bandits[k]
      bandit.init()
      player.init(bandit, self.__horizon)
    ############################################################################

    agg_regret = dict()
    for t in range(self.__horizon + 1):
      if t > 0:
        self._one_round()
      if t > 0 and t % self.__frequency == 0:
        agg_regret[t] = self.regret()
    return dict({self._players[0].name: agg_regret})

  def regret(self):
    return sum([self._bandits[k].regret(self._players[k].rewards)
                for k in range(self._num_players)])

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
    pool = Pool(processes=self.__processors)

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

  # pylint: disable=arguments-differ
  def play(self, bandits, players, output_file, pars):
    for bandit in bandits:
      if not isinstance(bandit, Bandit):
        logging.fatal('Not a legimate bandit!')

    self._bandits = bandits
    self._players = players
    self.__output_file = output_file
    self._pars = pars

    logging.info('run learner %s with goal %s under protocol %s' %
        (players[0].name, players[0].goal, self.type))
    start_time = time.time()
    self.__multi_proc()
    logging.info('%.2f seconds elapsed' % (time.time() - start_time))

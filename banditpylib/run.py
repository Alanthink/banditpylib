"""
main file
"""
import json
import os

from importlib import import_module
import tempfile

from absl import logging
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'output runtime debug info')
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_boolean('novar', False, 'do not show std in the output figure')

BANDIT_PKG = 'banditpylib.bandits'
LEARNER_PKG = 'banditpylib.learners'
PROTOCOL_PKG = 'banditpylib.protocols'

__all__ = ['run']


def parse(config, new_policies):
  setups = []

  bandit_name = config['environment']['bandit']
  Bandit = getattr(import_module(BANDIT_PKG), bandit_name)
  bandit_pars = config['environment']['params']
  learner_goal = config['learners']['goal']
  # initialize learners and their corresponding protocols
  for learner_config in config['learners']['policies']:
    learner_package = '%s.%s.%s' % \
        (LEARNER_PKG, learner_goal, learner_config['params']['type'])
    learner_name = learner_config['policy']
    Learner = getattr(import_module(learner_package), learner_name)

    Protocol = getattr(import_module(PROTOCOL_PKG), Learner.protocol)
    protocol = Protocol(learner_config['params'])

    if 'SinglePlayer' in Learner.protocol:
      # single player
      bandit = Bandit(bandit_pars)
      learner = Learner(learner_config['params'])
    else:
      # multiple players
      if 'num_players' not in learner_config['params']:
        logging.fatal(
            '%s: please specify the number of players!' % learner_name)
      num_players = learner_config['params']['num_players']
      bandit = [Bandit(bandit_pars) for _ in range(num_players)]
      learner = [Learner(learner_config['params']) for _ in range(num_players)]
    setups.append((bandit, protocol, learner))

  if new_policies:
    for learner in new_policies:
      if isinstance(learner, list):
        # multi-players
        num_players = len(learner)
        bandit = [Bandit(bandit_pars) for _ in range(num_players)]
        Protocol = getattr(import_module(PROTOCOL_PKG), learner[0].protocol)
        protocol = Protocol({'num_players': num_players})
      else:
        # single player
        bandit = Bandit(bandit_pars)
        protocol = getattr(import_module(PROTOCOL_PKG), learner.protocol)()
      setups.append((bandit, protocol, learner))

  goals = [setup[2][0].goal if isinstance(learner, list) else setup[2].goal
           for setup in setups]
  if any(goal != goals[0] for goal in goals):
    logging.fatal('Learners must have the same goal!')
  logging.info('run with goal %s' % goals[0])
  return setups, config['running']


def run(config, new_policies=None, debug=False):
  # initialization of flags
  FLAGS([''])
  FLAGS.debug = debug
  if FLAGS.debug:
    # DEBUG, INFO, WARN, ERROR, FATAL
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.INFO)

  setups, running_pars = parse(config, new_policies)

  try:
    temp_dir = tempfile.mkdtemp()
    data_file = os.path.join(temp_dir, FLAGS.out)
    for (bandit, protocol, learner) in setups:
      protocol.play(bandit, learner, running_pars, data_file)
    data = []
    with open(data_file, 'r') as file:
      for line in file:
        data.append(json.loads(line))
    os.remove(data_file)
  finally:
    os.rmdir(temp_dir)

  return data

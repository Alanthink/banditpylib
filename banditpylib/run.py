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


def parse(config):
  setups = []

  # initialize bandit
  bandit_type = config['environment']['bandit']
  Bandit = getattr(import_module(BANDIT_PKG), bandit_type)
  bandit_pars = config['environment']['params']
  # initialize learners and their corresponding protocols
  learner_goal = config['learners']['goal']
  for learner_config in config['learners']['policies']:
    if 'protocol' in learner_config:
      protocol_type = learner_config['protocol']['type']
      Protocol = getattr(import_module(PROTOCOL_PKG), protocol_type)
      protocol_pars = learner_config['protocol']['params']
      protocol = Protocol(protocol_pars)
    else:
      if learner_goal == 'regretmin':
        protocol_type = 'SinglePlayerRegretMinProtocol'
      elif learner_goal in ['bestarmid.fixbudget', 'bestarmid.fixconf']:
        protocol_type = 'SinglePlayerPEProtocol'
      else:
        logging.fatal('%s: no specified protocol!' % learner_config['policy'])
      Protocol = getattr(import_module(PROTOCOL_PKG), protocol_type)
      protocol_pars = dict()
      protocol = Protocol()

    policy = learner_config['policy']
    learner_package = '%s.%s.%s' % \
        (LEARNER_PKG, learner_goal, learner_config['params']['type'])
    policy_pars = learner_config['params'] \
        if 'params' in learner_config else dict()
    Learner = getattr(import_module(learner_package), policy)

    if 'SinglePlayer' in protocol_type:
      # single player
      bandit = Bandit(bandit_pars)
      learner = Learner(policy_pars)
      goal = learner.goal
    else:
      # multiple players
      if 'num_players' not in protocol_pars:
        logging.fatal('%s: please specify the number of players!' % policy)
      num_players = protocol_pars['num_players']
      bandit = [Bandit(bandit_pars) for _ in range(num_players)]
      learner = [Learner(policy_pars) for _ in range(num_players)]
      goal = learner[0].goal

    setups.append((bandit, protocol, learner))

  logging.info('run with goal %s' % goal)

  return setups, config['running']


def run(config, debug=False):
  # initialization of flags
  FLAGS([''])
  FLAGS.debug = debug
  if FLAGS.debug:
    # DEBUG, INFO, WARN, ERROR, FATAL
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.INFO)

  setups, running_pars = parse(config)

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

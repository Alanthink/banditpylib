"""
To run, try `python3 main.py` under `banditpylib` root directory.
The results are put under `out/` by default.
"""

import json
import os

from importlib import import_module

from absl import app
from absl import logging
from absl import flags

from draw import draw_figure

FLAGS = flags.FLAGS

flags.DEFINE_string('config', 'config.json', 'config file')
flags.DEFINE_string('dir', 'out', 'output directory')
flags.DEFINE_string('out', 'data.out', 'file for generated data')
flags.DEFINE_boolean('debug', False, 'output runtime debug info')
flags.DEFINE_boolean('novar', False, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_boolean('fig', False, 'generate figure only')

BANDIT_PKG = 'bandits'
LEARNER_PKG = 'learners'
PROTOCOL_PKG = 'protocols'


def parse(config):
  setups = []

  # initialize bandit
  bandit_type = config['environment']['bandit']
  Bandit = getattr(import_module(BANDIT_PKG), bandit_type)
  bandit_pars = config['environment'][bandit_type]
  # initialize learners and their corresponding protocols
  learner_goal = config['learners']['goal']
  for learner_config in config['learners']['policies']:
    if 'protocol' in learner_config:
      protocol_type = learner_config['protocol']['type']
      Protocol = getattr(import_module(PROTOCOL_PKG), protocol_type)
      protocol_pars = learner_config['protocol'][protocol_type]
      protocol = Protocol(protocol_pars)
    else:
      if learner_goal == 'regretmin':
        protocol_type = 'SinglePlayerRegretMinProtocol'
      elif learner_goal in ['bestarmid.fixbudget', 'bestarmid.fixconf']:
        protocol_type = 'SinglePlayerBAIProtocol'
      else:
        logging.fatal('%s: no specified protocol!' % learner_config['policy'])
      Protocol = getattr(import_module(PROTOCOL_PKG), protocol_type)
      protocol_pars = dict()
      protocol = Protocol()

    policy = learner_config['policy']
    learner_package = '%s.%s.%s' % \
        (LEARNER_PKG, learner_goal, learner_config[policy]['type'])
    policy_pars = learner_config[policy] if policy in learner_config else dict()
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

  return setups, config['running_pars']


def main(argv):
  del argv

  if FLAGS.debug:
    # DEBUG, INFO, WARN, ERROR, FATAL
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.INFO)

  data_file = os.path.join(FLAGS.dir, FLAGS.out)

  # load config
  with open(FLAGS.config, 'r') as json_file:
    config = json.load(json_file)

  if not FLAGS.fig:
    setups, running_pars = parse(config)

    os.makedirs(os.path.dirname(data_file), exist_ok=True)
    prev_files = os.listdir(FLAGS.dir)
    if FLAGS.rm:
      for file in prev_files:
        os.remove(os.path.join(FLAGS.dir, file))
    else:
      if os.listdir(FLAGS.dir):
        logging.fatal(('%s/ is not empty. Make sure you have'
                       ' archived previously generated data. '
                       'Try --rm flag which will automatically'
                       ' delete previous data.') % FLAGS.dir)

    for (bandit, protocol, learner) in setups:
      protocol.play(bandit, learner, running_pars, data_file)

  # figure generation
  draw_figure(config['learners']['goal'])


if __name__ == '__main__':
  app.run(main)

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

flags.DEFINE_string('config_filename', 'config.json', 'config filename')
flags.DEFINE_string('dir', 'out', 'output directory')
flags.DEFINE_string('data_filename', 'data.out', 'filename for generated data')
flags.DEFINE_boolean('debug', False, 'output runtime debug info')
flags.DEFINE_boolean('novar', False, 'do not show std in the output figure')
flags.DEFINE_boolean('rm', False, 'remove previously generated data')
flags.DEFINE_boolean('fig', False, 'generate figure only')

ARM_PKG = 'arms'
BANDIT_PKG = 'bandits'
LEARNER_PKG = 'learners'


def parse(config):
  means = config['environment']['arm']['means']
  Arm = getattr(import_module(ARM_PKG),
      config['environment']['arm']['type'])
  arms = [Arm(mean) for mean in means]
  Bandit = getattr(import_module(BANDIT_PKG),
      config['environment']['bandit'])
  bandit = Bandit(arms)
  learner_package = '%s.%s.%s' % \
      (LEARNER_PKG, config['learner']['goal'], config['learner']['type'])
  learners = [getattr(import_module(learner_package), learner)()
      for learner in config['learner']['policy']]
  pars = config['parameters']
  return learners, bandit, pars


def main(argv):
  del argv

  if FLAGS.debug:
    # DEBUG, INFO, WARN, ERROR, FATAL
    logging.set_verbosity(logging.DEBUG)
  else:
    logging.set_verbosity(logging.INFO)

  data_file = os.path.join(FLAGS.dir, FLAGS.data_filename)

  # load config
  with open(FLAGS.config_filename, 'r') as json_file:
    config = json.load(json_file)

  if not FLAGS.fig:
    learners, bandit, pars = parse(config)

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

    for learner in learners:
      learner.play(bandit, data_file, pars)

  # figure generation
  draw_figure(config['learner']['goal'])


if __name__ == '__main__':
  app.run(main)

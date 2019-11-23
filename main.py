"""
A simple example.

To run, try `python3 main.py` under `banditpylib` root directory.
The result is output to `out/figure.pdf` by default.
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
flags.DEFINE_string('data_filename', 'data.out', 'output data filename')
flags.DEFINE_string('figure_filename', 'figure.pdf', 'output figure filename')
flags.DEFINE_boolean('debug', False, 'run a simple setup for debug')
flags.DEFINE_boolean('novar', False, 'do not show std in the output figure')
flags.DEFINE_enum('do', 'all', ['all', 'd', 'f', 'r'],
    ('d:generate the data,',
     'f:generate the figure,',
     'r:remove the data, all:do everything'))

# DEBUG, INFO, WARN, ERROR, FATAL
logging.set_verbosity(logging.INFO)

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

  data_file = os.path.join(FLAGS.dir, FLAGS.data_filename)
  figure_file = os.path.join(FLAGS.dir, FLAGS.figure_filename)

  # load config
  with open(FLAGS.config_filename, 'r') as json_file:
    config = json.load(json_file)

  if FLAGS.do in ['d', 'all']:
    # data generation

    learners, bandit, pars = parse(config)

    # clean file
    open(data_file, 'w').close()

    for learner in learners:
      learner.play(bandit, data_file, pars)

  if FLAGS.do in ['f', 'all']:
    # figure generation
    draw_figure(data_file, figure_file, config['learner']['goal'], FLAGS.novar)
  if FLAGS.do in ['r', 'all']:
    # remove generated data
    os.remove(data_file)


if __name__ == '__main__':
  app.run(main)
